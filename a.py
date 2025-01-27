import pandas as pd
import pickle
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import gymnasium as gym
from stable_baselines3 import A2C
import tensorboard

class tccEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, network: nx.Graph, actions_amout: int, stop_function = None, reward_function = None):
        self.network = network 
        self.position = list(self.network.nodes)[random.randint(0, self.network.number_of_nodes()-1)]
        self.n = self.network.number_of_nodes()
        self.stop_function = self.default_stop if stop_function is None else stop_function 
        self.reward_function = self.default_reward if reward_function is None else reward_function 

        self.action_space = gym.spaces.Discrete(actions_amout)
        self.observation_space = gym.spaces.Box(low=0, high=np.array([self.n]), shape=(1,), dtype=np.int32)

        self.count = 0
        
    def reset(self,seed=None):
        super().reset(seed=seed)
        self.position = list(self.network.nodes)[random.randint(0, self.network.number_of_nodes()-1)]
        self.count = 0

        obs = self.position
        return obs, {}
    
    def default_reward(self, action):
        possible_actions = list(self.network.neighbors(self.position))
        #print(possible_actions)
        if action < len(possible_actions):
            self.position = possible_actions[action]
            reward = possible_actions[action]// 10000
        else:
            reward = -100000
        
        return reward
    
    def default_stop(self):
        return self.count > 50
    

    def step(self,action):

        reward = self.reward_function(action)
        
        self.count += 1
        
        terminated = self.stop_function()
        
        obs = self.position

        return obs, reward, terminated, False, {"count" : self.count}



with open('./sunt/graph_designer/graph_gtfs.gpickle', 'rb') as f:
    G = pickle.load(f)
map = {}
for node in G.nodes:
    map[node] = int(node)
G = nx.relabel_nodes(G, map)

env = tccEnv(G, 9)


def train_sb3():
    # Where to store trained model and logs
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = tccEnv(G, 9)

    # Use Advantage Actor Critic (A2C) algorithm.
    # Use MlpPolicy for observation space 1D vector.
    model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
   
    # This loop will keep training until you stop it with Ctr-C.
    # Start another cmd prompt and launch Tensorboard: tensorboard --logdir logs
    # Once Tensorboard is loaded, it will print a URL. Follow the URL to see the status of the training.
    # Stop the training when you're satisfied with the status.
    TIMESTEPS = 1000
    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False) # train
        model.save(f"{model_dir}/a2c_{TIMESTEPS*iters}") # Save a trained model every TIMESTEPS


train_sb3()