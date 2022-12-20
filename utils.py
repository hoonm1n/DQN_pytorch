import math
import random
import numpy as np
import torch

def get_explore_rate(episode, DECAY_CONSTANT, MIN_EXPLORE_RATE):
    if episode >= DECAY_CONSTANT-1:
        return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((episode+1)/DECAY_CONSTANT)))
    else:
        return 1.0

def select_action(actiontable, explore_rate, env):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        # Select the action with the highest q value
        action = np.argmax(actiontable).item()
    return action


from collections import deque
class replay_buffer():
    def __init__(self, memory_size):
        self.state      = deque([], maxlen=memory_size)
        self.action     = deque([], maxlen=memory_size)
        self.next_state = deque([], maxlen=memory_size)
        self.reward     = deque([], maxlen=memory_size)
        self.done       = deque([], maxlen=memory_size)

    def __len__(self):
        return len(self.state)
        
    def insert_memory(self, rec):
        self.state.append(rec[0])
        self.action.append(rec[1])
        self.next_state.append(rec[2])
        self.reward.append(rec[3])
        self.done.append(rec[4])

    def sample(self, batch_size=64):
        indices = np.random.randint(0, len(self.state), batch_size)

        state_batch      = torch.FloatTensor(np.array(self.state)[indices].reshape(batch_size, -1))
        action_batch     = torch.FloatTensor(np.array(self.action)[indices].reshape(batch_size, -1))
        next_state_batch = torch.FloatTensor(np.array(self.next_state)[indices].reshape(batch_size, -1))
        reward_batch     = torch.FloatTensor(np.array(self.reward)[indices].reshape(batch_size, -1))
        done_batch       = torch.FloatTensor(np.array(self.done)[indices].reshape(batch_size, -1))

        return state_batch, action_batch, next_state_batch, reward_batch, done_batch