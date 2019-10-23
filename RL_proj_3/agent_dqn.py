#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
import collections
from collections import deque
import os
import sys

import time



import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""
#Hyper parameters for training

REPLAY_SIZE = 100000

GAMMA = 0.99
BATCH_SIZE = 32
        
LEARNING_RATE = 1e-4
TARGET_UPDATE_INTERVAL = 1000
LEARNING_STARTS = 10000
SAVE_INTERVAL = 200

DEVICE = 'gpu'

EPSILON_DECAY = 10**4
EPSILON_START = 1.0
EPSILON_END   = 0.02
        

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # initializations for replay memory
        self.env = env
        self.buffer = collections.deque(maxlen=REPLAY_SIZE)      # initializing a replay memory buffer

        #initializations of agent
        self._reset()
        self.last_action = 0

        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        
        ###########################
        pass

    def push(self, experience):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        """
        ###########################
        self.buffer.append(experience)
        ###########################
        
        
    def replay_buffer(self, batch_size):

        """ You can add additional arguments as you need.
        Select batch from buffer.

        sample a batch of 32 from the experience collected
        """
        ###########################
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        ########################### 
        # The 'states' below are already in the transposed form because they are sampled from experience
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)


    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0
    

    def make_action(self, net, epsilon=0.0, device='gpu'):
        """
        select action
        """
        
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()

        else:
            state_a = np.array([self.state.transpose(2,0,1)], copy=False)
            #torch.tensor opperation appends a '1' at the start of the numpy array
            state_v = torch.tensor(state_a).to(device)
            Q_values_v = net(state_v)
            _, action_v = torch.max(Q_values_v, dim=1)
            action = int(action_v.item())

        ###########################
        return action

    def play_step(self, net, epsilon=0.0, device='gpu'):

        """
        execute action and take a step in the environment
        add the state,action,rewards to the experience replay
        return the total_reward
        """
        done_reward = None
        action_for_exp = self.make_action(net, epsilon, DEVICE)

        new_state, reward, is_done, _ = self.env.step(action_for_exp)
        self.total_reward += reward
        new_state = new_state
        #remember that the state that comes in from taking a step in our environment
        # will be in the form of width X height X depth

        # But whatever state goes into experience will be in the form of depth X height X width
        # i.e the experience buffer will have state in the transposed format
        # because this is the format that pytorch input should look like
        exp = Experience(self.state.transpose(2,0,1), action_for_exp, reward, is_done, new_state)

        #adding experiences in our replay memory
        self.push(exp)
        self.state = new_state

        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


    def loss_function(batch, net, target_net, device="gpu"):

        states, actions, rewards, dones, new_states = batch

        state_v = torch.tensor(states).to(device)
        print(state_v.shape)
        next_states_v = torch.tensor(next_states_v).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done = torch.ByteTensor(dones).to(device)

        state_action_values =net(states_v).gather(1, actions_v.long().unsqueeze(-1)).squeeze(-1)
        next_state_values  = target_net(next_states_v).max(1)[0]
        next_state_values[done] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values*GAMMA +rewards_v

        loss = nn.MSELoss()(state_action_values, expected_state_action_values)

        return loss



    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        device = torch.device("gpu")

        net = DQN((4, 84, 84), self.env.action_space.n).to(device)
        target_net = DQN((4, 84, 84), self.env.action_space.n).to(device)

        epsilon = EPSILON_START

        #defining the optimizer for your neural network
        optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE)

        #empty list of total rewards
        total_rewards = []
        best_mean_reward = None
        # initializations for time and speed calculation 
        frame_idx = 0
        timestep_frame = 0
        timestep = time.time()


        while True:

            frame_idx += 1
            epsilon = max(EPSILON_END, ((EPSILON_START-frame_idx)/EPSILON_DECAY))

            reward = self.play_step(net, epsilon, device=device)

            if reward is not None:
                #appending rewards in an empty list of total_rewards
                total_rewards.append(reward)

                # not asked to calculate speed 
                speed = (frame_idx-timestep_frame)/(time.time()-timestep)
                timestep_frame = frame_idx
                timestep = time.time()

                #calculating mean of last(recent) 100 rewards
                mean_reward = np.mean(total_rewards[-100:])

                print("{} frames: done {} games, mean reward {}, epsilon {}, speed {} frames/s".format(frame_idx, len(total_rewards), round(mean_reward, 3), round(epsilon,2), round(speed, 2)))

                if best_mean_reward is None or best_mean_reward < mean_reward or len(total_rewards)%25 == 0:
                    
                    if best_mean_reward is not None:
                        print("New best mean reward {} -> {}, model saved".format(round(best_mean_reward, 3), round(mean_reward, 3)))


            if frame_idx % SAVE_INTERVAL == 0
                torch.save(net.state_dict(), "breakoutNoFrameSkip"  + ".dat")

            #checking the replay memory
            if len(self.buffer) < LEARNING_STARTS:
                continue

            #check if we need to update our target function
            if frame_idx % TARGET_UPDATE_INTERVAL == 0:
                target_net.load_state_dict(net.state_dict())

            # we dont wanna accumilate our gradients
            # hence it is importent to make them zero at every iteration

            optimizer.zero_grad()
            # sampling a batch from buffer
            batch = replay_buffer(BATCH_SIZE)
            loss_t = loss_function(batch, net, target_net, device= device)
            print(loss_t)
            loss_t.backward()
            optimizer.step()

        self.env.close()        
        ###########################
