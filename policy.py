import tensorflow as tf


class PolicyNetwork:
    def __init__(self, config, name='policy_network'):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, config['state_size']], name="state")
            self.action = tf.placeholder(tf.int32, [config['action_size']], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            kernel_initializer = tf.contrib.layers.xavier_initializer(seed=config['seed'])
            dense = tf.layers.dense(units=config['policy_units'][0],
                                    inputs=self.state,
                                    kernel_initializer=kernel_initializer,
                                    activation=tf.nn.relu)
            bn = tf.layers.batch_normalization(inputs=dense)

            for units in config['policy_units'][1:]:
                dense = tf.layers.dense(units=units,
                                        inputs=bn,
                                        kernel_initializer=kernel_initializer,
                                        activation=tf.nn.relu)
                bn = tf.layers.batch_normalization(inputs=dense)

            self.output = tf.layers.dense(units=config['action_size'],
                                          inputs=bn,
                                          kernel_initializer=kernel_initializer,
                                          activation=None)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)

            self.global_step = tf.Variable(0, trainable=False)
            decayed_lr = tf.train.exponential_decay(learning_rate=config['learning_rate_policy'],
                                                    global_step=self.global_step,
                                                    decay_steps=config['learning_rate_decay_steps_policy'],
                                                    decay_rate=config['learning_rate_decay_rate_policy'],
                                                    staircase=True)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=decayed_lr).minimize(self.loss, global_step=self.global_step)
