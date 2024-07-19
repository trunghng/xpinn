import random, argparse, os, json
from datetime import datetime as dt

import numpy as np
import torch

from utils import load_dataset, load_training_data, poisson_exp
from xpinn import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extended Physics Informed Neural Network')
    mode_parsers = parser.add_subparsers(title='Modes')

    train_parser = mode_parsers.add_parser('train')
    train_parser.set_defaults(mode='train')
    train_parser.add_argument('--N-b', type=int, default=200, help='Number of boundary points in each subdomain')
    train_parser.add_argument('--N-F', type=int, default=1000, help='Number of residual points in each subdomain')
    train_parser.add_argument('--N-I', type=int, default=100, help='Number of interface points in each interface')
    train_parser.add_argument('--interfaces', nargs='+', type=int, action='append',
                                default=[[0, 1], [0, 2]], help='Interface list.\
                                E.g. [[sd1_idx, sd2_idx], [sd1_idx, sd3_idx], [sd3_idx, sd4_idx]]]')
    train_parser.add_argument('--W-u', type=float, default=20, help='Data mismatch weight')
    train_parser.add_argument('--W-F', type=float, default=1, help='Residual weight')
    train_parser.add_argument('--W-I', type=float, default=20, help='Average solution continuity weight')
    train_parser.add_argument('--W-IF', type=float, default=1, help='Residual continuity weight along the interface')
    train_parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    train_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    train_parser.add_argument('--verbose', action='store_true', help='Whether to display training log')
    train_parser.add_argument('--save-model', action='store_true', help='Whether to save the model')

    test_parser = mode_parsers.add_parser('test')
    test_parser.set_defaults(mode='test')
    test_parser.add_argument('--model-path', type=str, help='Model path to load')

    for p in [train_parser, test_parser]:
        p.add_argument('--exp-name', type=str, help='Experiment name')
        p.add_argument('--seed', type=int, default=0, help='Seed for RNG')
        p.add_argument('--layers', nargs='+', type=int, action='append',
                    default=[[2, 30, 30, 1], [2, 20, 20, 20, 20, 1], [2, 25, 25, 25, 1]],
                    help='MLP architectures of subnets')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    xb, yb, ub, x_f1, y_f1, x_f2, y_f2, x_f3, y_f3, x_i1, y_i1, x_i2,\
        y_i2, u_exact, x_total, y_total = load_dataset()

    log_dir = os.path.join(os.getcwd(), 'data', args.exp_name) if args.exp_name else f'/tmp/experiments/{str(dt.now())}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    output = json.dumps(vars(args), separators=(',',':\t'), indent=4)
    print('Experiment config:\n', output)
    with open(os.path.join(log_dir, 'config.json'), 'w') as out:
        out.write(output)

    if args.mode == 'train':
        xb_train, yb_train, ub_train, x_f1_train, y_f1_train, x_f2_train, y_f2_train, x_f3_train, y_f3_train,\
            x_i1_train, y_i1_train, x_i2_train, y_i2_train = load_training_data(xb, yb, ub, x_f1, y_f1, x_f2,\
                y_f2, x_f3, y_f3, x_i1, y_i1, x_i2, y_i2, args.N_b, args.N_F, args.N_I)

        train([(xb_train, yb_train), None, None], [ub_train, None, None],
            [(x_f1_train, y_f1_train), (x_f2_train, y_f2_train), (x_f3_train, y_f3_train)],
            [(x_i1_train, y_i1_train), (x_i2_train, y_i2_train)], args.interfaces, [poisson_exp] * len(args.layers),
            args.layers, args.W_u, args.W_F, args.W_I, args.W_IF, args.epochs, args.lr, args.verbose, args.save_model, log_dir)
    else:
        test(xb, yb, x_f1, y_f1, x_f2, y_f2, x_f3, y_f3, x_i1, y_i1, x_i2, y_i2, u_exact, x_total, y_total,
                args.layers, args.model_path, log_dir)
