"""
- action-dim: 6 / 26 actions
- agent.reset_optimizer: leraning rate decay
- reward: projection reward
- add target model, update its weights every target_update epochs
- add force reward for each step by envoking traj_mwpts function in generate_trajectory
- local sensing input: global to local, residual link
- stochastic action reinitialization (bound collision)
- add obst-generation-mode: voxel_random, voxel_constraint, plane_random, radom, gazebo_random
"""

import os,sys,math,csv,argparse,logging,random,time,torch,pdb
from datetime import datetime
import numpy as np
from numpy import *
from torch import nn
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from environment import Env
from generate_trajectory import traj_mwpts
from network_model import LinearNetwork,ConvNetwork
from plot import plot_env

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, item):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(item)

    def sample(self, batch_size):
        return zip(*random.sample(self.buffer, batch_size))

    def size(self):
        return len(self.buffer)

class Agent(object):
    def __init__(self, args):
        self.args=args
        self.is_training = not args.eval
        self.load_pretrained = args.load_pretrained
        assert args.buffer_size >= args.batch_size
        self.batch_size = args.batch_size
        self.buffer = ReplayBuffer(args.buffer_size)
        self.action_dim = args.action_dim
        self.gamma = args.gamma
        self.lr = args.lr
        self.model = None
        self.target_model = None
        if args.mode == "linear":
            if torch.cuda.is_available():
                self.model = LinearNetwork(args.state_dim, args.action_dim).cuda()
                self.target_model = LinearNetwork(args.state_dim, args.action_dim).cuda()
            else:
                self.model = LinearNetwork(args.state_dim, args.action_dim)
                self.target_model = LinearNetwork(args.state_dim, args.action_dim)
        elif args.mode == "conv":
            if torch.cuda.is_available():
                self.model = ConvNetwork(args.sensing_range, args.action_dim).cuda()
                self.target_model = ConvNetwork(args.sensing_range, args.action_dim).cuda()
            else:
                self.model = ConvNetwork(args.sensing_range, args.action_dim)
                self.target_model = ConvNetwork(args.sensing_range, args.action_dim)
        assert self.model is not None
        assert self.target_model is not None
        if args.load_pretrained:
            pre_weight_path = './model/saved_weights_%s_local.pth.tar'%args.mode
            if os.path.isfile(pre_weight_path):
                print("=> loading checkpoint '{}'".format(pre_weight_path))
                checkpoint = torch.load(pre_weight_path,map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                self.model.load_state_dict(checkpoint['state_dict'])

            else:
                raise ValueError('Weight path does not exist.')
        self.update_target()
        self.model.train()
        self.target_model.eval()
        self.reset_optimizer(self.lr)


    def print_model_weight(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    def reset_optimizer(self, lr):
        """reset optimizer learning rate.
        """
        if self.args.optimizer == 'adam':
            self.model_optim = torch.optim.Adam(
                self.model.parameters(), lr=lr)
        elif self.args.optimizer == 'sgd':
            self.model_optim = torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.5,
                weight_decay=args.weight_decay)
        return

    def update_target(self):
        print("=> updating target network weights...")
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, epsilon=0.0, prev_is_hit_bound=False, topk_rand=1):
        """Output an action.
        """
        if not self.is_training:
            epsilon = 0.0
        if random.random() >= epsilon:
            if isinstance(state, tuple):
                state_var = []
                for temp in state:
                    if torch.cuda.is_available():
                        state_var.append(torch.tensor(temp, dtype=torch.float).unsqueeze(0).cuda())
                    else:
                        state_var.append(torch.tensor(temp, dtype=torch.float).unsqueeze(0))
                state_var = tuple(state_var)
            else:
                if torch.cuda.is_available():
                    state_var = torch.tensor(state, dtype=torch.float).unsqueeze(0).cuda()
                else:
                    state_var = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            # self.model.eval()
            logits = self.model(state_var).detach().cpu().numpy()
            if not prev_is_hit_bound:
                actions_sort = np.argsort(logits[0], -1)
                rand_idx = np.random.randint(topk_rand)
                action = actions_sort[-1 * rand_idx - 1]
            else:
                action = random.randrange(self.action_dim)
        else:
            assert self.is_training == True
            action = random.randrange(self.action_dim)
        return action

    def learning(self):
        """Extract from buffer and train for one epoch.
        """
        data_list = self.buffer.sample(self.batch_size)
        (states_curt, action_curt, rewards_curt, states_next, is_dones) = \
            self._stack_to_numpy(data_list)
        if isinstance(states_curt, tuple):
            states_curt_var = []
            for temp in states_curt:
                states_curt_var.append(
                    torch.tensor(
                        temp, dtype=torch.float).cuda()
                )
            states_curt_var = tuple(states_curt_var)
        else:
            states_curt_var = torch.tensor(
                states_curt, dtype=torch.float).cuda()
        action_curt_var = torch.tensor(
            action_curt, dtype=torch.long).cuda()
        rewards_curt_var = torch.tensor(
            rewards_curt, dtype=torch.float).cuda()
        if isinstance(states_next, tuple):
            states_next_var = []
            for temp in states_next:
                states_next_var.append(
                    torch.tensor(
                        temp, dtype=torch.float).cuda()
                )
            states_next_var = tuple(states_next_var)
        else:
            states_next_var = torch.tensor(
                states_next, dtype=torch.float).cuda()
        is_dones_var = torch.tensor(
            is_dones, dtype=torch.float).cuda()
        # if self.is_training and not self.load_pretrained:
        if self.is_training:
            self.model.train()
        else:
            self.model.eval()
        logits_curt_var = self.model(states_curt_var)
        q_value = logits_curt_var.gather(1, action_curt_var.unsqueeze(1)).squeeze(1)
        logits_next_var = self.target_model(states_next_var)
        next_q_value = logits_next_var.max(1)[0]
        expected_q_value = rewards_curt_var + \
            self.gamma * next_q_value * (1 - is_dones_var)

        loss_mse = (q_value - expected_q_value.detach()).pow(2).mean()
        loss_mae = torch.abs(q_value - expected_q_value.detach()).mean()
        loss = torch.max(loss_mse, loss_mae)
        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()
        return loss.detach().item()

    def _stack_to_numpy(self, data_list):
        ret = []
        for temp_data in data_list:
            if isinstance(temp_data[0], tuple):
                temp_list = []
                tuple_size = len(temp_data[0])
                for _ in range(tuple_size):
                    temp_list.append([])
                for curt_tup in temp_data:
                    for idx in range(tuple_size):
                        temp_list[idx].append(curt_tup[idx])
                temp_ret_list = []
                for temp in temp_list:
                    temp_ret_list.append(np.array(temp))
                ret.append(tuple(temp_ret_list))
            else:
                temp_np = np.array(temp_data)
                ret.append(temp_np)
        return ret


class Trainer(object):
    def __init__(self, agent, env, args):
        self.args = args
        self.agent = agent
        self.env = env
        self.max_steps = args.max_steps
        self.batch_size = args.batch_size
        self.save_epochs = args.save_epochs
        self.save_weights_dir = args.save_weights_dir
        self.num_obst = args.num_obst
        self.num_objs = args.num_objs
        # non-Linear epsilon decay
        epsilon_final = args.epsilon_min
        epsilon_start = args.epsilon
        epsilon_decay = args.epsilon_decay
        if args.enable_epsilon:
            self.epsilon_by_frame = \
                lambda frame_idx: epsilon_final + \
                    (epsilon_start - epsilon_final) * math.exp(
                        -1. * (frame_idx // epsilon_decay))
        else:
            self.epsilon_by_frame = lambda frame_idx: 0.0

    def train(self):
        timestamp = datetime.now()
        time_str = timestamp.strftime("%H_%M_%S")
        loss_reward_filepath = os.path.join('.', 'loss_reward_{}.csv'.format(time_str))
        if os.path.exists(loss_reward_filepath):
            os.remove(loss_reward_filepath)
        lr = self.agent.lr
        episode = 0
        while True:
            episode += 1
            episode_reward = 0
            is_done = False
            is_goal = False
            prev_is_hit_bound = False
            steps = 0
            num_obst = 0
            num_outbound = 0
            epsilon = self.epsilon_by_frame(episode)
            logging.info('epsilon: {0:.04f}'.format(epsilon))
            actions = []
            loss_list = []

            rewards = []
            # Intialize environment with different number of obstacles
            self.num_obst = random.randint(0, 100)
            self.num_objs = random.randint(5, 10)
            self.env.reset(self.num_obst, self.num_objs)
            state_curt = self.env.get_state()
            segment  = np.array(state_curt[1] * self.args.env_size)
            velocity_curt = np.array((0, 0, 0.001))
            acceler_curt = np.array((0, 0, 0))
            gerk_curt = np.array((0, 0, 0))
            waypoints = []
            while (not is_done) and (steps <= self.max_steps):
                action_curt = self.agent.act(state_curt, epsilon=epsilon, prev_is_hit_bound=prev_is_hit_bound, topk_rand=2)
                actions.append(action_curt)
                reward_curt, is_done, reward_info = self.env.step(action_curt)
                num_obst += int(reward_info['is_obst'])
                num_outbound += int(reward_info['is_bound'])
                prev_is_hit_bound = reward_info['is_bound']
                if reward_info['is_goal']:
                    is_goal = True
                waypoints.append(list(self.env.objs_info['drone_pos']))
                state_next = self.env.get_state()
                # calculate force reward
                if self.args.thrust_reward:
                    segment = vstack((segment, np.array(state_next[1] * self.args.env_size)))
                    num = segment.shape[0]
                    t = np.asarray([0])
                    for i in range(num - 1):
                        t = hstack((t, 6 * (i + 1)))
                    path, f, norm_f, velocity_next, acceler_next, gerk_next = \
                        traj_mwpts(t, segment.T, np.array([velocity_curt]).T,
                                   np.array([acceler_curt]).T, np.array([gerk_curt]).T)
                    force_reward = 1 / (1 + math.exp(-1 * np.sum(norm_f)/norm_f.shape[1])) / \
                        self.args.grid_resolution / self.args.env_size
                    reward_curt -= force_reward
                self.agent.buffer.add((state_curt, action_curt, reward_curt, state_next, is_done))
                state_curt = state_next
                episode_reward += reward_curt
                rewards.append(reward_curt)
            #     loss = 0.0
            #     if self.agent.buffer.size() >= self.batch_size:
            #         loss = self.agent.learning()
            #         loss_list.append(loss)
                steps += 1

            if self.agent.buffer.size() >= self.batch_size:
                loss = self.agent.learning()
                loss_list.append(loss)

            loss_avg = sum(loss_list) / max(len(loss_list), 1)
            waypoints.append(list(self.env.objs_info['drone_pos']))

            # update target model weights
            if episode % args.target_update == 0:
                self.agent.update_target()

            if int(args.verbose) >= 2:
                print('actions: ', actions)
            logging.info('loss_avg: {0:.04f}'.format(loss_avg))

            print('episode: {0:05d}, step: {1:03d}, reward: {2:.04f}, num_obst: {3:01d}, num_outbound: {7:01d}, is_goal: {4}, start: {5}, target: {6}'.format(
                episode,
                steps,
                episode_reward,
                num_obst,
                is_goal,
                self.env.objs_info['drone_pos_start'],
                self.env.objs_info['goal'],
                num_outbound
            ))
            if episode % 100 == 0:
                print('actions: \n', actions)

            # evaluating
            if episode % 2000 == 0:
                print("Evaluating the performance...")
                self.eval()
                time.sleep(5)

            # learning decay
            # if episode % 5000 == 0:
            #     lr *= 0.8
            #     self.agent.reset_optimizer(lr)

            # plot reward and loss
            with open(loss_reward_filepath, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([loss_avg, episode_reward, num_obst, int(is_goal)])

            if episode % self.save_epochs == 0:
                save_dic = {
                    'args' : args,
                    'episode' : episode,
                    'state_dict' : self.agent.model.state_dict()
                }
                if not os.path.exists(self.save_weights_dir):
                    os.mkdir(self.save_weights_dir)
                torch.save(save_dic, os.path.join(
                    self.save_weights_dir, 'saved_weights_{}_local_demo.pth.tar'.format(args.mode)))

    def eval(self, env, source, destination):
        episode = 0
        success = 0
        achieve = 0
        while True:
            episode += 1
            episode_reward = 0
            is_done = False
            is_goal = False
            steps = 0
            num_obst = 0
            actions = []
            loss_list = []
            rewards = []
            # Intialize environment
            #self.num_obst = random.randint(0, 100) # grids num
            #self.num_objs = random.randint(5, 10)  # obs num
            #self.env.reset(env=env, source=source, destination=destination)
            #self.env.reset(self.num_obst, self.num_objs)
            state_curt = self.env.get_state()
            segment  = np.array(state_curt[1] * self.args.env_size)
            velocity_curt = np.array((0, 0, 0.001))
            acceler_curt = np.array((0, 0, 0))
            gerk_curt = np.array((0, 0, 0))
            waypoints = []
            waypoints.append(list(self.env.objs_info['drone_pos']))
            while (not is_done) and (steps <= self.max_steps):
                action_curt = self.agent.act(state_curt, epsilon=0.0)
                actions.append(action_curt)
                reward_curt, is_done, reward_info = self.env.step(action_curt)
                num_obst += int(reward_info['is_obst'])
                if reward_info['is_goal']:
                    is_goal = True
                    achieve += 1
                    if num_obst == 0:
                        success += 1
                state_next = self.env.get_state()
                waypoints.append(list(self.env.objs_info['drone_pos']))
                #calculate force reward
                if self.args.thrust_reward:
                    segment = vstack((segment, np.array(state_next[1] * self.args.env_size)))
                    num = segment.shape[0]
                    t = np.asarray([0])
                    for i in range(num - 1):
                        t = hstack((t, 6 * (i + 1)))
                    path, f, norm_f, velocity_next, acceler_next, gerk_next = \
                        traj_mwpts(t, segment.T, np.array([velocity_curt]).T,
                                np.array([acceler_curt]).T, np.array([gerk_curt]).T)
                    force_reward = 1 / (1 + math.exp(-1 * np.sum(norm_f)/norm_f.shape[1])) / \
                        self.args.grid_resolution / self.args.env_size
                    reward_curt -= force_reward
                state_curt = state_next
                episode_reward += reward_curt
                rewards.append(reward_curt)
                steps += 1

            # waypoints.append(list(self.env.objs_info['drone_pos']))
            if episode % 100 == 0:
                print('Evaluating success ratio: {0:.03f}'.format(float(success / episode)))
                print('Evaluating achieve ratio: {0:.03f}'.format(float(achieve / episode)))

            if (self.args.eval):
                print('episode: {0:05d}, step: {1:03d}, reward: {2:.01f}, num_obst: {3:03d}, is_goal: {4}, start: {5}, target: {6}'
                .format(episode,steps,episode_reward,num_obst,is_goal,self.env.objs_info['drone_pos_start'],self.env.objs_info['goal']))

            if reward_info['is_goal'] and num_obst == 0:
                print(waypoints)
                print(self.env.objs_info['obst_list'])
                #plot_env(self.env, waypoints)
            print(waypoints)
            plot_env(self.env, waypoints)
            return waypoints

def trajectory_main(obs_list, source, goal):
    args = parse_args()
    #setup_logging(args)
    env = Env(args, obs_list, source, goal)
    agent = Agent(args)
    trainer = Trainer(agent, env, args)
    if not args.eval:
        trainer.train()
    else:
        return trainer.eval(env, source, goal)

def main(args):
    env = Env(args)
    agent = Agent(args)
    trainer = Trainer(agent, env, args)
    if not args.eval:
        trainer.train()
    else:
        trainer.eval()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--mode", default='conv', choices=['linear', 'conv'], type=str)
    parser.add_argument("--batch-size", default=200, type=int)
    parser.add_argument("--optimizer", default='adam', choices=['adam', 'sgd'], type=str)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--env-size", default=10, type=int)
    parser.add_argument("--sensing-range", default=5, type=int)
    parser.add_argument("--grid-resolution", default=0.7, type=float)
    parser.add_argument("--num-obst", default=50, type=int, help='number of uccupied grids')
    parser.add_argument("--num-objs", default=15, type=int, help='number of shaped obstacles')
    parser.add_argument("--state-dim", default=204, type=int, help='maximum obs number n: 50 for linear model, n*4+4')
    parser.add_argument("--action-dim", default=26, choices=[6, 26], type=int)
    parser.add_argument("--eval", default=True)
    parser.add_argument("--buffer-size", default=2000, type=int)
    parser.add_argument("--gamma", default=0.9, type=float)
    parser.add_argument("--enable-epsilon", action='store_true')
    parser.add_argument("--epsilon", default=1.0, type=float)
    parser.add_argument("--epsilon-min", default=0.1, type=float)
    parser.add_argument("--epsilon-decay", default=200, type=int)
    parser.add_argument("--max-steps", default=200, type=int)
    parser.add_argument("--save-epochs", default=1000, type=int)
    parser.add_argument("--save-weights-dir", default='./saved_weights', type=str)
    parser.add_argument("--load-pretrained", default=True)
    parser.add_argument("--thrust-reward", action='store_true')
    parser.add_argument("--target-update", default=30, type=int)
    parser.add_argument("--obst-generation-mode",default="voxel_random",choices=['voxel_random', 'plane_random', 'voxel_constrain', 'test', 'random', 'gazebo_random', 'demo'],type=str)
    parser.add_argument("--verbose", default='2', type=str)

    return parser.parse_args()

def setup_logging(args, log_path=None):
    """Setup logging module
    """
    lvl = {
        '0': logging.ERROR,
        '1': logging.WARN,
        '2': logging.INFO
    }

    logging.basicConfig(
        level=lvl[args.verbose],
        filename=log_path)


if __name__ == "__main__":
    '''
    args = parse_args()
    setup_logging(args)
    main(args)
    '''
    obs_list=np.array([[4,4,i] for i in range(10)])

    source=np.array([3,1,2])
    goal=np.array([6,6,2])
    trajectory_main(obs_list, source, goal)
    #'''
