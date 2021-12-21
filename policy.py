import tensorflow as tf


class PolicyNetwork:
    def __init__(self, config, name='policy_network'):
        with tf.variable_scope(name):
            self.state = tf.placeholder(
                tf.float32, [None, config['state_size']], name="state")
            self.action = tf.placeholder(
                tf.int32, [config['action_size']], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.lr = tf.placeholder(tf.float32, name="lr")

            kernel_initializer = tf.contrib.layers.xavier_initializer(
                seed=config['seed'])
            dense = tf.layers.dense(units=config['policy_units'][0],
                                    inputs=self.state,
                                    kernel_initializer=kernel_initializer,
                                    activation=tf.nn.relu)

            for units in config['policy_units'][1:]:
                dense = tf.layers.dense(units=units,
                                        inputs=dense,
                                        kernel_initializer=kernel_initializer,
                                        activation=tf.nn.relu)

            self.output = tf.layers.dense(units=config['action_size'],
                                          inputs=dense,
                                          kernel_initializer=kernel_initializer,
                                          activation=None)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)

            if config['type'] == 'actor_critic':
                self.optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.lr).minimize(self.loss)
            else:
                global_step = tf.Variable(0, trainable=False)
                decayed_lr = tf.train.exponential_decay(learning_rate=config['lr_decay_rate_policy'],
                                                        global_step=global_step,
                                                        decay_steps=config['lr_decay_steps_policy'],
                                                        decay_rate=config['lr_decay_rate_policy'],
                                                        staircase=True)
                self.optimizer = tf.train.AdamOptimizer(
                    decayed_lr).minimize(loss, global_step=global_step)
