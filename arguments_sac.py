import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='FetchReach-v1', help='the environment name')
    parser.add_argument('--n-epochs', type=int, default=50, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=40, help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--noise-eps', type=float, default=0.25, help='noise eps')
    parser.add_argument('--target-noise-eps', type=float, default=0.1, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=1e-4, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=1e-4, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.995, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=50, help='the number of tests')
    parser.add_argument('--target-update-interval', type=int, default=2, help='Delay in the update of the Actor target network')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, help='the rollouts per mpi')
    parser.add_argument('--init-exploration-steps', type=int, default=int(1e4), help='the steps of the initial exploration')
    parser.add_argument('--init-exploration-policy', type=str, default='uniform', help='the inital exploration policy')
    parser.add_argument('--log-std-max', type=float, default=2, help='the maximum log std value')
    parser.add_argument('--log-std-min', type=float, default=-6, help='the minimum log std value')
    parser.add_argument('--entropy-weights', type=float, default=0.2, help='the entropy weights')
    parser.add_argument('--action-repeat', type=int, default=10, help='repeat the action for n times')
    parser.add_argument('--n-updates', type=int, default=16, help='The times to update the network but during an episode itself (different from n-batches)')
    parser.add_argument('--num-envs', type=int, default=8, help='Number of environments')
    parser.add_argument("--disable_wandb", action="store_true", help="Use this flag to disable wandb logging")
    parser.add_argument("--use_bc_with_rl", action="store_true", help="Use this flag to do RLfD")
    parser.add_argument('--num_bc_demos', type=int, default=5, help='the number of expert demos to train the agent for RLfD')



    args = parser.parse_args()

    return args
