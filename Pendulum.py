import gym
from DoubleDQN import DoubleDQN
import numpy as np
import tensorflow as tf

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 11

sess = tf.Session()
with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess)

sess.run(tf.global_variables_initializer())

def train(model):
    total_steps = 0
    observation = env.reset()
    while True:
        action = model.choose_action(observation)

        f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4) # # convert to [-2 ~ 2] float actions
        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10 # normalize to a range of (-1, 0)
        model.store_transition(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:
            model.learn()

        if total_steps - MEMORY_SIZE > 10000:
            break

        observation = observation_
        total_steps += 1



q_double = train(double_DQN)
