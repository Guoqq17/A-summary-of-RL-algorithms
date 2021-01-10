"""
The main file to run the algorithms
Each algorithm is tested with an environment in GYM, e.g., CartPole-v0 is used to test the original DQN
I will keep improving this file to make it more integrated
Qiangqiang Guo, Jan 9, 2021
"""

import gym
from algorithms.DQN import DQN


def main():
    ## DQN
    # Initialize gym environment
    env_name = 'CartPole-v0'
    max_episode = 10000         # maximum episode limitation
    max_steps = 300             # step limitation in an episode
    env = gym.make(env_name)

    agent = DQN(s_dim = env.observation_space.shape[0],
              a_dim = env.action_space.n,
              memory_capacity = 5000,
              steps_update_target = 200,
              epsilon = 0.1,
              batch_size = 64,
              gamma = 0.9,
              n_hidden_layers = 1,
              n_hidden_units = [128],
              act_funcs = ['relu', 'softmax'])

    total_reward = 0
    for n_epi in range(max_episode):
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

        # print to see the average reward every 200 episodes
        if not n_epi % 200:
            print('-' * 20)
            print('total reward in the %d 200:' %(n_epi // 200))
            print(total_reward / 200)
            total_reward = 0
if __name__ == '__main__':
  main()
