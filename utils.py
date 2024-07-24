from typing import List, Tuple
import random

import numpy as np
import torch
from torch.autograd import grad
import scipy
import matplotlib.pyplot as plt


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def ftensor(x, requires_grad: bool=False):
    return torch.tensor(x, dtype=torch.float32, requires_grad=requires_grad)


def load_default_dataset():
    data = scipy.io.loadmat('./dataset/XPINN_2D_PoissonEqn.mat')

    xb = data['xb'].T
    yb = data['yb'].T
    ub = data['ub'].T
    x_f1 = data['x_f1'].T
    y_f1 = data['y_f1'].T
    x_f2 = data['x_f2'].T
    y_f2 = data['y_f2'].T
    x_f3 = data['x_f3'].T
    y_f3 = data['y_f3'].T
    x_i1 = data['xi1'].T
    y_i1 = data['yi1'].T
    x_i2 = data['xi2'].T
    y_i2 = data['yi2'].T
    u_exact = data['u_exact'].T
    x_total = data['x_total'].T
    y_total = data['y_total'].T

    Xb = [(xb, yb), None, None]
    ub = [ub, None, None]
    Xf = [(x_f1, y_f1), (x_f2, y_f2), (x_f3, y_f3)]
    Xi = [(x_i1, y_i1), (x_i2, y_i2)]

    return Xb, ub, Xf, Xi, x_total, y_total, u_exact


def load_dataset():
    raise NotImplementedError("load_dataset() method not implemented")


def load_training_data(
        Xb: List[Tuple[np.ndarray, np.ndarray]],
        ub: List[np.ndarray],
        Xf: List[Tuple[np.ndarray, np.ndarray]],
        Xi: List[Tuple[np.ndarray, np.ndarray]],
        N_b: int, N_F: int, N_I: int
    ):
    Xb_train, ub_train, Xf_train, Xi_train = [], [], [], []

    for i in range(len(Xb)):
        if Xb[i]:
            xb, yb = Xb[i]
            idx = np.random.choice(xb.shape[0], N_b, replace=False)
            Xb_train.append((ftensor(xb[idx, :]), ftensor(yb[idx, :])))
            ub_train.append(ftensor(ub[i][idx, :]))
        else:
            Xb_train.append(None)
            ub_train.append(None)

    for i in range(len(Xf)):
        x_f, y_f = Xf[i]
        idx = np.random.choice(x_f.shape[0], N_F, replace=False)
        x_f_train = ftensor(x_f[idx, :], requires_grad=True)
        y_f_train = ftensor(y_f[idx, :], requires_grad=True)
        Xf_train.append((x_f_train, y_f_train))

    for i in range(len(Xi)):
        x_i, y_i = Xi[i]
        idx = np.random.choice(x_i.shape[0], N_I, replace=False)
        x_i_train = ftensor(x_i[idx, :], requires_grad=True)
        y_i_train = ftensor(y_i[idx, :], requires_grad=True)
        Xi_train.append((x_i_train, y_i_train))

    return Xb_train, ub_train, Xf_train, Xi_train


def flat(x):
    return [x[i] for i in range(x.shape[0])]


def poisson_exp(model, x: torch.Tensor, y: torch.Tensor):
    """
    Poisson's equation:
        u_xx + u_yy = e^x + e^y
    """
    u = model([x, y])
    u_x = grad(flat(u), x, create_graph=True, allow_unused=True)[0]
    u_y = grad(flat(u), y, create_graph=True, allow_unused=True)[0]
    u_xx = grad(flat(u_x), x, create_graph=True, allow_unused=True)[0]
    u_yy = grad(flat(u_y), y, create_graph=True, allow_unused=True)[0]
    return u_xx + u_yy - torch.exp(x) - torch.exp(y)


def plot_loss(losses: List[List[float]], lossnames: List[str], plot_path: str):
    for loss, name in zip(losses, lossnames):
        plt.plot(np.arange(1, len(loss) + 1), loss, label=name)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig(plot_path)
