import time
import torch
from unityagents import UnityEnvironment
import numpy as np
from ddpg_agent import Agent
from collections import deque
import torch

from model import *
from config import Config

device = torch.device("cpu")

env = UnityEnvironment(file_name="Tennis.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

config = Config(num_workers=num_agents)

def run(env, agent, max_step=10000, train_mode=True):
    env_info = env.reset(train_mode=train_mode)[brain_name]
    states = env_info.vector_observations
    scores = np.zeros(num_agents)
    agent.reset()

    for t in range(max_step): 
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations 
        rewards = env_info.rewards
        dones = env_info.local_done
        if train_mode:
            agent.step(states, actions, rewards, next_states, dones)

        states = next_states
        scores += rewards 
        if np.any(dones):
            break

    return np.mean(scores)
    
    
def ddpg(agent, train_mode=True):
    scores_window = deque(maxlen=100)
    scores = []
        
    for i in range(1, config.episode_count+1):
        begin = time.time()
        score = run(env=env, agent=agent, train_mode=train_mode)
        scores_window.append(score)
        score_average = np.mean(scores_window)
        scores.append(score)
        
        if i % 10 == 0:
            print('\rEpisode {} Average score: {:.2f} Min: {:.2f} Max: {:.2f} Time: {:.2f} Epsilon: {:.2f}'.format(
                i, 
                score_average, 
                np.min(scores), 
                np.max(scores), 
                time.time() - begin,
                agent.epsilon
            ))        
                    
        if score_average >= 0.5:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\nSolve in {:d} episodes. Average score: {:.2f}'.format(i, score_average))            
            break            
            
    return scores


if __name__ == '__main__':
    rand_seed = 0
    agent = Agent(config=config,
                  state_size=state_size, 
                  action_size=action_size, 
                  num_agents=num_agents, 
                  random_seed=rand_seed,
                  device=torch.device('cpu'),
                  actor_trained_weight_filename="checkpoint_actor.pth", 
                  critic_trained_weight_filename="checkpoint_critic.pth")

    scores = ddpg(agent, train_mode=False)

    env.close()