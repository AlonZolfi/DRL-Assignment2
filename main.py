import argparse
import utils
import datetime


def main(config):
    utils.save_config(config)
    utils.train(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Policy gradients for gym-ai environments')
    # Environment parameters
    parser.add_argument('--env_name', type=str, default='CartPole-v1',
                        help='See options in: https://gym.openai.com/envs')
    parser.add_argument('--render', type=bool, default=False,
                        help='Whether game graphics should be shown')

    # Training parameters
    parser.add_argument('--type', type=str, default='reinforce_with_baseline',
                        help='', choices=['reinforce', 'reinforce_with_baseline', 'actor_critic'])
    parser.add_argument('--seed', type=int, default=1,
                        help='The seed for reproducibility')
    parser.add_argument('--max_episodes', type=int, default=2000,
                        help='Maximum number of steps in each episode')
    parser.add_argument('--max_steps', type=int, default=501,
                        help='Maximum number of steps in each episode')
    parser.add_argument('--discount_factor', type=float, default=0.99,
                        help='The discount factor')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='The path to save to logs to')

    # Policy network parameters
    parser.add_argument('--learning_rate_policy', type=float, default=1e-4,
                        help='The policy gradient network learning rate')
    parser.add_argument('--learning_rate_decay_rate_policy', type=float, default=0.99,
                        help='The policy network learning rate decay value')
    parser.add_argument('--learning_rate_decay_steps_policy', type=int, default=300,
                        help='The policy network learning rate decay steps')
    parser.add_argument('--policy_units', metavar='N', type=int, nargs='+', default=[64, 64, 64, 64],
                        help='Number of units in each dense layer of the policy network')

    # Baseline network parameters
    parser.add_argument('--learning_rate_baseline', type=float, default=1e-4,
                        help='The baseline network learning rate')
    parser.add_argument('--learning_rate_decay_rate_baseline', type=float, default=0.99,
                        help='The policy network learning rate decay value')
    parser.add_argument('--learning_rate_decay_steps_baseline', type=int, default=300,
                        help='The policy network learning rate decay steps')
    parser.add_argument('--baseline_units', metavar='N', type=int, nargs='+', default=[32, 32, 32],
                        help='Number of units in each dense layer of the baseline network')
    main(vars(parser.parse_args()))
