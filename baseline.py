import tensorflow as tf


class BaselineNetwork:
    def __init__(self, config, name='baseline'):
        with tf.variable_scope(name):
            self.state = tf.placeholder(
                tf.float32, [None, config['state_size']], name="state")
            self.state_value = tf.placeholder(tf.float32,  name="state_value")
            self.td_error = tf.placeholder(tf.float32,  name="td_error")
            self.lr = tf.placeholder(tf.float32, name="lr")

            kernel_initializer = tf.contrib.layers.xavier_initializer(
                seed=config['seed'])
            dense = tf.layers.dense(units=config['baseline_units'][0],
                                    inputs=self.state,
                                    kernel_initializer=kernel_initializer,
                                    activation=tf.nn.relu)

            for units in config['baseline_units'][1:]:
                dense = tf.layers.dense(units=units,
                                        inputs=dense,
                                        kernel_initializer=kernel_initializer,
                                        activation=tf.nn.relu)

            self.output = tf.layers.dense(units=1,
                                          inputs=dense,
                                          kernel_initializer=kernel_initializer,
                                          activation=None)

            self.loss = tf.squared_difference(self.output, self.state_value)
            if config['type'] == 'actor_critic':
                self.lr = self.lr*self.td_error
                self.optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.lr).minimize(self.loss)
            else:
                global_step = tf.Variable(0, trainable=False)
                decayed_lr = tf.train.exponential_decay(learning_rate=config['lr_baseline'],
                                                        global_step=global_step,
                                                        decay_steps=config['lr_decay_steps_baseline'],
                                                        decay_rate=config['lr_decay_rate_baseline'],
                                                        staircase=True)
                self.optimizer = tf.train.AdamOptimizer(
                    decayed_lr).minimize(self.loss, global_step)
