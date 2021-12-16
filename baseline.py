import tensorflow as tf


class BaselineNetwork:
    def __init__(self, config, name='baseline'):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, config['state_size']], name="state")
            self.state_value = tf.placeholder(tf.float32,  name="state_value")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            kernel_initializer = tf.contrib.layers.xavier_initializer(seed=config['seed'])
            dense = tf.layers.dense(units=config['baseline_units'][0],
                                    inputs=self.state,
                                    kernel_initializer=kernel_initializer,
                                    activation=tf.nn.relu)
            bn = tf.layers.batch_normalization(inputs=dense)

            for units in config['baseline_units'][1:]:
                dense = tf.layers.dense(units=units,
                                        inputs=bn,
                                        kernel_initializer=kernel_initializer,
                                        activation=tf.nn.relu)
                bn = tf.layers.batch_normalization(inputs=dense)

            self.output = tf.layers.dense(units=1,
                                          inputs=bn,
                                          kernel_initializer=kernel_initializer,
                                          activation=None)

            self.diff = tf.squared_difference(self.output, self.state_value)
            self.loss = self.diff
            self.global_step = tf.Variable(0, trainable=False)
            decayed_lr = tf.train.exponential_decay(learning_rate=config['learning_rate_baseline'],
                                                    global_step=self.global_step,
                                                    decay_steps=config['learning_rate_decay_steps_baseline'],
                                                    decay_rate=config['learning_rate_decay_rate_baseline'],
                                                    staircase=True)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=decayed_lr).minimize(self.loss, global_step=self.global_step)
