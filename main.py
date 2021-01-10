"""
The main file to run the algorithms
"""

import gym
from algorithms.DQN import DQN


env_name = 'CartPole-v0'
episode = 10000 # Episode limitation
max_steps = 300 # Step limitation in an episode

def main():
  # initialize gym
    env = gym.make(env_name)

    agent = DQN(time_step = 1,
              s_dim = env.observation_space.shape[0],
              a_dim = env.action_space.n,
              memory_capacity = 5000,
              steps_update_target = 200,
              epsilon = 0.1,
              batch_size = 64,
              gamma = 0.9,
              n_hidden_layers = 1,
              n_hidden_nodes = [128],
              act_funcs = ['relu', 'softmax'])

    total_reward = 0
    for n_epi in range(episode):
        step, done = 0, False
        state = env.reset()
        while not done and step < max_steps:
            action = agent.greedy_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            if agent.pointer >= agent.memory_capacity:
                agent.learn()
        if not n_epi % 200:
            print('-' * 20)
            print('total reward in the %d 200:' %(n_epi // 200))
            print(total_reward / 200)
            total_reward = 0
if __name__ == '__main__':
  main()
