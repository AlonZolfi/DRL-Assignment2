import argparse
import utils


def main(config):
    utils.train(config)
    # utils.write_log(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Q-learning for gym-ai environments')
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
    parser.add_argument('--max_episodes', type=int, default=1000,
                        help='Maximum number of steps in each episode')
    parser.add_argument('--max_steps', type=int, default=501,
                        help='Maximum number of steps in each episode')
    parser.add_argument('--discount_factor', type=float, default=0.95,
                        help='The discount factor')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='The path to save to logs to')

    # Policy network parameters
    parser.add_argument('--learning_rate_policy', type=float, default=5e-4,
                        help='The policy gradient network learning rate')
    parser.add_argument('--learning_rate_decay_rate_policy', type=float, default=0.95,
                        help='The policy network learning rate decay value')
    parser.add_argument('--learning_rate_decay_steps_policy', type=int, default=50,
                        help='The policy network learning rate decay steps')
    parser.add_argument('--policy_units', metavar='N', type=int, nargs='+', default=[128, 64],
                        help='Number of units in each dense layer of the policy network')

    # Baseline network parameters
    parser.add_argument('--learning_rate_baseline', type=float, default=5e-4,
                        help='The baseline network learning rate')
    parser.add_argument('--learning_rate_decay_rate_baseline', type=float, default=0.95,
                        help='The policy network learning rate decay value')
    parser.add_argument('--learning_rate_decay_steps_baseline', type=int, default=50,
                        help='The policy network learning rate decay steps')
    parser.add_argument('--baseline_units', metavar='N', type=int, nargs='+', default=[128, 64],
                        help='Number of units in each dense layer of the baseline network')
    main(vars(parser.parse_args()))
