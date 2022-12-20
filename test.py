from model import Q_net
import torch
import os
import torch.nn as nn
import argparse
import gym
import numpy as np
from tqdm import tqdm
'''
####################################################
    DO NOT CHANGE THIS CODE. YOUR MODEL WILL BE EVALUED USING THIS EXACT CODE.
    You are free to change any other files as you want.
    CONSTRUCT YOUR MODEL, THEN TEST YOUR CODE RUNNING THIS FILE
####################################################
'''
def validate(model:nn.Module, iteration):
    iteration = iteration
    with torch.no_grad():
        total_t = 0
        for episode in tqdm(range(iteration)):
            done = False
            state_0 = env.reset()
            while done == False:
                total_t += 1
                actiontable = model(torch.Tensor(state_0).unsqueeze(0))
                action = np.argmax(actiontable).item()
                state, _, done, _ = env.step(action)
                state_0 = state
    return total_t/iteration

if __name__  == '__main__':
    parser = argparse.ArgumentParser(description='2022 AI Project')
    parser.add_argument('--iteration', type = int, default= 1000)

    args = parser.parse_args()

    THE_model = Q_net()
    abspath = os.path.dirname(os.path.abspath(__file__))+'/modelparam_final.pt'
    THE_model.load_state_dict(torch.load(abspath))
    env = gym.make('CartPole-v1')
    score = str(validate(THE_model, args.iteration))
    with open('result.txt', 'w') as f:
        f.writelines(score)