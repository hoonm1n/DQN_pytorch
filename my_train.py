from torch.optim import SGD, Adam #Default optimizer. you may use other optimizers
from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn as nn
import gym
from utils import get_explore_rate, select_action, replay_buffer
from model import Q_net
import copy

import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
update_num = 0
global_step = 0



def simulate(model, args): #model: the neural network
    #optimizer and loss for the neural network
    optimizer = Adam(model.parameters(), lr=0.0005, eps=1e-7)
    criterion = nn.MSELoss()
    model_target = copy.deepcopy(model)
    global_step = 0

    ## Instantiating the learning related parameters
    explore_rate = get_explore_rate(0, args.decay_constant, args.min_explore_rate)

    memory = replay_buffer(args.max_memory)
    num_streaks = 0
    for episode in range(args.num_episodes):
        # Reset the environment
        state = env.reset()
        loss = 0
        
        for t in range(args.max_timestep):
            # env.render()#you may want to comment this line out, to run code silently without rendering
            
            # Selecting an action. the action MUST be choosed from the neural network's output.
            with torch.no_grad():
                actiontable = model(torch.tensor(state))
                action = select_action(actiontable, explore_rate, env)

            # Execute the action then collect outputs
            next_state, reward, done, _ = env.step(action)
            writer.add_scalar("reward", reward, global_step)
            #Update the memory
            memory.insert_memory((state, action, next_state, reward, done))

            # Train
            if len(memory) > 1000:
                loss = replay(model, model_target, memory, args, criterion, optimizer)
                writer.add_scalar("loss", loss, global_step)

            state = next_state
            
            
            global_step += 1
            #done: the cart failed to maintain balance
            if done == True:
                break

        writer.flush()
        writer.add_scalar("timestep", t, episode)
        # Update parameters
        explore_rate = get_explore_rate(episode, args.decay_constant, args.min_explore_rate)

        #update target network: soft-update
        for param, target_param in zip(model.parameters(), model_target.parameters()):
            target_param.data.copy_(
                0.9*param.data + (1-0.9)*target_param.data
                )
        
        if episode % 100 == 0:
            torch.save(model.state_dict(), f'model/modelparam_{episode}.pt')
        print(f'Episode: {episode}, timestep: {t+1:3}/250, total_step: {global_step}, explore: {explore_rate}')
        # print("Episode %d finished after %f time steps." % (episode, t))
        

        #The episode is considered as a success if timestep >SOLVED_TIMESTEP 
        if (t >= args.solved_timestep):
            num_streaks += 1
        else:
            num_streaks = 0
            
        #  when the agent 'solves' the environment: steak over 120 times consecutively
        if num_streaks > args.streak_to_end:
            print("The Environment is solved")
            #torch.save(model.state_dict(), 'modelparam_2.pt')
            break

    writer.close()##########
    env.close()#closes window
    

def replay(model, model_target, memory, args, criterion, optimizer, iteration = 1):
    batch = memory.sample(args.batchsize)
    state, action, next_state, reward, done = batch
    
    y = model(state).max(dim=1, keepdim=True)[0]

    q_next = model_target(next_state).max(dim=1, keepdim=True)[0]
    target = reward + args.discount_factor * (1-done) * q_next

    loss = criterion(y, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model!')
    parser.add_argument('--num_episodes', type = int, default= 10000)
    parser.add_argument('--max_timestep', type = int, default= 250)
    parser.add_argument('--solved_timestep', type = int, default= 99) #199
    parser.add_argument('--streak_to_end', type = int, default= 120) ## 쌓이는거 횟수
    parser.add_argument('--batchsize', type = int, default= 512)
    parser.add_argument('--min_explore_rate', type = float, default= 0.01)
    parser.add_argument('--discount_factor', type = float, default= 0.99)
    parser.add_argument('--decay_constant', type = int, default= 500)#25
    parser.add_argument('--max_memory', type = int, default = 10000)
    train_args = parser.parse_args()

    env = gym.make('CartPole-v1')
    qnet = Q_net()
    simulate(qnet, train_args)