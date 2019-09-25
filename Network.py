# -*- coding: utf-8 -*
import numpy as np
import tensorflow as tf

class DQN:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        # e_increment 从0开始加
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()
        # Update `ref` by assigning `value` to it
        # making replace_target_p a sentence, not a function
        t_p = tf.get_collection('target_net_p')
        e_p = tf.get_collection('eval_net_p')

        # replace_target_op, one line of code
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_p, e_p)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.cost_all = []

    def _build_net(self):
        with tf.variable_scope('eval_net'):
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
            self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
            # num_steps_between_gradient_updates : n_l
            c_names, n_l1, w_init, b_init = \
                ['eval_net_p', tf.GraphKeys.GLOBAL_VARIABLES], \
                10, tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

            # s = state
            w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_init, collections=c_names)
            b1 = tf.get_variable('b1', [1, n_l1], initializer=b_init, collections=c_names)
            l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_init, collections=c_names)
            b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_init, collections=c_names)
            self.q_eval = tf.matmul(l1, w2) + b2

            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # target_net
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_p', tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w3', [self.n_features, n_l1], initializer=w_init)
                b1 = tf.get_variable('b3', [1, n_l1], initializer=b_init)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w4', [n_l1, self.n_actions], initializer=w_init)
                b2 = tf.get_variable('b4', [1, self.n_actions], initializer=b_init)
                self.q_next = tf.matmul(l1, w2) + b2

    # samples need to update one time gradient
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):

        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation}) # recursively fill placeholder
            # return the indices
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:

            self.sess.run(self.replace_target_op)

        if self.memory_counter > self.memory_size:
            # randomly pick previous learning experience
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:], # fixted params
                self.s: batch_memory[:, :self.n_features] # newest params
            })

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target}) # s -> q_val
        self.cost_all.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon.max else self.epsilon.max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_all)), self.cost_all)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


