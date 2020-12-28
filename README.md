# A summary of RL algorithms
In this repo, I will summarize and implement some key algorithms in RL. This serves as a self-learning notes on my way to RL.

## 1. Deep Q-Learning

This section follows the flow of this paper: [Rainbow: Combining Improvements in Deep Reinforcement Learning (Hessel et al., 2017)](https://arxiv.org/pdf/1710.02298.pdf), which gives a good summary of Deep Q-learning algorithms as well as test different combinations of those algorithms.

### 1.1 Original deep Q-Networks (DQN)

Let's start from the very beginning, [Mnih et al. (2015)](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf) proposed the Deep Q-Networks (DQN) algorithm at Nature. This will be used as the baseline of this section. A convolutional neural network is used to learn the Q values using experience replay. The bellman equation for the state-value function for policy $\pi$ (i.e., $v_{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t = s]$) is
$$
\begin{aligned}
v_{\pi}(s) &= \mathbb{E}_{\pi}[G_t | S_t = s]\\
& = \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} | S_t = s]\\
& =\sum_{a} \pi(a|s) \sum_{s^{\prime}} \sum_{r} p(s^{\prime}, r | s, a) [r + \gamma \mathbb{E}_{\pi}[G_{t+1} | S_t = s]]\\
& =\sum_{a} \pi(a|s) \sum_{s^{\prime}, r} p(s^{\prime}, r | s, a) [r +\gamma v_{\pi}(s^{\prime})]\\
\end{aligned}
$$
sf
