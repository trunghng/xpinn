import argparse
import os
import json
from datetime import datetime as dt

from utils import set_seed, load_default_dataset, load_dataset, load_training_data, poisson_exp, pde
from xpinn import train, test


def main():
    parser = argparse.ArgumentParser(description='Extended Physics Informed Neural Network')
    mode_parsers = parser.add_subparsers(title='Modes')

    train_parser = mode_parsers.add_parser('train')
    train_parser.set_defaults(mode='train')
    train_parser.add_argument('--N-bs', nargs='+', type=int, default=[200],
                                help='Number of boundary points in subdomains')
    train_parser.add_argument('--N-Fs', nargs='+', type=int, default=[5000, 1800, 1200],
                                help='Number of residual points in subdomains')
    train_parser.add_argument('--N-Is', nargs='+', type=int, default=[100, 100],
                                help='Number of interface points in interfaces')
    train_parser.add_argument('--interfaces', nargs='+', type=int, action='append',
                                default=[[0, 1], [0, 2]], help='Interface list.\
                                E.g., [[sd1_idx, sd2_idx], [sd1_idx, sd3_idx], [sd3_idx, sd4_idx]]]')
    train_parser.add_argument('--W-u', type=float, default=20, help='Data mismatch weight')
    train_parser.add_argument('--W-F', type=float, default=1, help='Residual weight')
    train_parser.add_argument('--W-I', type=float, default=20, help='Average solution continuity weight')
    train_parser.add_argument('--W-IF', type=float, default=1, help='Residual continuity weight along the interface')
    train_parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
    train_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    train_parser.add_argument('--verbose', action='store_true', help='Whether to display training log')
    train_parser.add_argument('--log-freq', type=int, default=10, help='Logging frequency')
    train_parser.add_argument('--save-model', action='store_true', help='Whether to save the model')

    test_parser = mode_parsers.add_parser('test')
    test_parser.set_defaults(mode='test')
    test_parser.add_argument('--model-path', type=str, help='Model path to load')

    for p in [train_parser, test_parser]:
        p.add_argument('--exp-name', type=str, help='Experiment name')
        p.add_argument('--seed', type=int, default=0, help='Seed for RNG')
        p.add_argument('--nondefault-dataset', action='store_true', help='Whether to use the default dataset')
        p.add_argument('--plot-dataset', action='store_true', help='Whether to plot the dataset')
        p.add_argument('--layers', nargs='+', type=int, action='append',
                        default=[[2, 30, 30, 1], [2, 20, 20, 20, 20, 1], [2, 25, 25, 25, 1]],
                        help='MLP architectures of subnets')

    args = parser.parse_args()
    set_seed(args.seed)

    log_dir = os.path.join(os.getcwd(), 'data', args.exp_name)\
        if args.exp_name else f'/tmp/experiments/{str(dt.now())}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if args.nondefault_dataset:
        Xb, ub, Xf, Xi, x_total, y_total, u_exact = load_dataset()
        f = pde
    else:
        Xb, ub, Xf, Xi, x_total, y_total, u_exact = load_default_dataset(log_dir, args.plot_dataset)
        f = poisson_exp
    del args.nondefault_dataset

    output = json.dumps(vars(args), separators=(',', ':\t'), indent=4)
    print('Experiment config:\n', output)
    with open(os.path.join(log_dir, 'config.json'), 'w') as out:
        out.write(output)

    if args.mode == 'train':
        Xb_train, ub_train, Xf_train, Xi_train = load_training_data(Xb, ub, Xf, Xi, args.N_bs, args.N_Fs, args.N_Is)

        train(Xb_train, ub_train, Xf_train, Xi_train, args.interfaces, f,
            args.layers, args.W_u, args.W_F, args.W_I, args.W_IF, args.epochs,
            args.lr, args.verbose, args.log_freq, args.save_model, log_dir)
    else:
        test(Xb, Xf, Xi, x_total, y_total, u_exact, args.layers, args.model_path, log_dir)


if __name__ == '__main__':
    main()
