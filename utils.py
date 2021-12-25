import gym
import tensorflow as tf
from policy import PolicyNetwork
from baseline import BaselineNetwork
import collections
import numpy as np
import pandas as pd
import os
import datetime
from pathlib import Path


def save_config(config):
    config['cur_run'] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    new_config = pd.DataFrame(
        pd.Series(config)).transpose().set_index('cur_run')
    new_config.to_csv('config_run_mapper.csv', mode='a',
                      header=not os.path.isfile('config_run_mapper.csv'))


def set_seed(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    tf.random.set_random_seed(seed_value)


def train(config):
    outpt_fotmat = "Episode {} Reward: {} Average over 100 episodes: {}"
    tf.disable_eager_execution()
    set_seed(config['seed'])
    env = gym.make(config['env_name'])

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    config['state_size'] = state_size
    config['action_size'] = action_size

    # Initialize the policy network
    tf.reset_default_graph()
    policy = PolicyNetwork(config)

    if config['type'] in ['reinforce_with_baseline', 'actor_critic']:
        baseline = BaselineNetwork(config)

    solved = False
    Transition = collections.namedtuple(
        "Transition", ["state", "action", "reward", "next_state", "done"])
    episode_rewards = np.zeros(config['max_episodes'])
    average_rewards = 0.0

    Path(config['log_dir']).mkdir(parents=True, exist_ok=True)
    summary_writer = tf.summary.FileWriter(
        os.path.join(config['log_dir'], config['cur_run']))

    # Start training the agent with REINFORCE algorithm
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for episode in range(config['max_episodes']):
            state = env.reset()
            state = state.reshape([1, state_size])
            episode_transitions = []

            for step in range(config['max_steps']):
                actions_distribution = sess.run(policy.actions_distribution,
                                                {policy.state: state})
                action = np.random.choice(np.arange(len(actions_distribution)),
                                          p=actions_distribution)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])

                if config['render']:
                    env.render()

                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1

                if config['type'] == 'actor_critic':
                    update_actor_critic(sess, config, policy, baseline,
                                        done, state, next_state, reward,
                                        action_one_hot)

                episode_transitions.append(
                    Transition(state=state, action=action_one_hot,
                               reward=reward, next_state=next_state,
                               done=done))
                episode_rewards[episode] += reward

                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(
                            episode_rewards[(episode - 99):episode + 1])
                    print(
                        outpt_fotmat.format(episode, episode_rewards[episode],
                                            round(average_rewards, 2)))
                    if average_rewards > 475:
                        print(' Solved at episode: ' + str(episode))
                        solved = True

                    write_custom_scalar_to_tensorboard(
                        summary_writer, 'avg_reward', average_rewards, episode)
                    write_custom_scalar_to_tensorboard(
                        summary_writer, 'steps', step, episode)
                    break

                state = next_state

            if solved:
                break

            # Weights update
            if config['type'] == 'reinforce':
                update_reinforce(sess, config, policy, episode_transitions)
            elif config['type'] == 'reinforce_with_baseline':
                update_reinforce_with_baseline(
                    sess, config, policy, baseline, episode_transitions)


def update_reinforce(sess, config, policy, episode_transitions):
    # Compute Rt for each time-step t and update the network's weights
    for t, transition in enumerate(episode_transitions):
        total_discounted_return = sum(
            config['discount_factor'] ** i * t.reward for i, t in enumerate(episode_transitions[t:]))  # Rt
        feed_dict_policy = {policy.state: transition.state, policy.R_t: total_discounted_return,
                            policy.action: transition.action}
        _, policy_loss = sess.run(
            [policy.optimizer, policy.loss], feed_dict_policy)


def update_reinforce_with_baseline(sess, config, policy, baseline,
                                   episode_transitions):
    for t, transition in enumerate(episode_transitions):
        total_discounted_return = sum(
            config['discount_factor'] ** i * t.reward
            for i, t in enumerate(episode_transitions[t:]))  # Rt

        value_function = sess.run(
            baseline.output, {baseline.state: transition.state})
        advantage = total_discounted_return - value_function

        feed_dict_baseline = {
            baseline.state: transition.state, baseline.state_value: advantage}
        _, baseline_loss = sess.run(
            [baseline.optimizer, baseline.loss], feed_dict_baseline)

        feed_dict_policy = {policy.state: transition.state,
                            policy.R_t: advantage,
                            policy.action: transition.action}
        _, policy_loss = sess.run(
            [policy.optimizer, policy.loss], feed_dict_policy)


def update_actor_critic(sess, config, policy, baseline,
                        done, state, next_state, reward, action):
    value_next_state = 0 if done else sess.run(
        baseline.output, {baseline.state: next_state})

    td_target = reward + config['discount_factor'] * value_next_state
    td_error = td_target - \
        sess.run(baseline.output, {baseline.state: state})

    feed_dict = {baseline.state: state, baseline.state_value: td_target}
    _, loss = sess.run(
        [baseline.optimizer, baseline.loss], feed_dict)

    feed_dict = {policy.state: state, policy.R_t: td_error,
                 policy.action: action}
    _, loss = sess.run(
        [policy.optimizer, policy.loss], feed_dict)


def write_custom_scalar_to_tensorboard(summary_writer, tag, value, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, step)
    summary_writer.flush()
