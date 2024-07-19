from typing import List

import numpy as np
import torch
from torch.autograd import grad
import scipy
import matplotlib.pyplot as plt


def ftensor(x, requires_grad: bool=False):
    return torch.tensor(x, dtype=torch.float32, requires_grad=requires_grad)


def load_dataset():
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
    x_total =  data['x_total'].T
    y_total =  data['y_total'].T

    return xb, yb, ub, x_f1, y_f1, x_f2, y_f2, x_f3, y_f3, x_i1,\
            y_i1, x_i2, y_i2, u_exact, x_total, y_total


def load_training_data(xb, yb, ub, x_f1, y_f1, x_f2, y_f2, x_f3, y_f3, x_i1,\
                        y_i1, x_i2, y_i2, N_b: int, N_F: int, N_I: int):
    idx1 = np.random.choice(xb.shape[0], N_b, replace=False)
    xb_train = ftensor(xb[idx1, :])
    yb_train = ftensor(yb[idx1, :])
    ub_train = ftensor(ub[idx1, :])

    idx2 = np.random.choice(x_f1.shape[0], N_F, replace=False)
    x_f1_train = ftensor(x_f1[idx2, :], requires_grad=True)
    y_f1_train = ftensor(y_f1[idx2, :], requires_grad=True)

    idx3 = np.random.choice(x_f2.shape[0], N_F, replace=False)
    x_f2_train = ftensor(x_f2[idx3, :], requires_grad=True)
    y_f2_train = ftensor(y_f2[idx3, :], requires_grad=True)

    idx4 = np.random.choice(x_f3.shape[0], N_F, replace=False)
    x_f3_train = ftensor(x_f3[idx4, :], requires_grad=True)
    y_f3_train = ftensor(y_f3[idx4, :], requires_grad=True)

    idx5 = np.random.choice(x_i1.shape[0], N_I, replace=False)
    x_i1_train = ftensor(x_i1[idx5, :])
    y_i1_train = ftensor(y_i1[idx5, :])

    idx6 = np.random.choice(x_i2.shape[0], N_I, replace=False)
    x_i2_train = ftensor(x_i2[idx6, :])
    y_i2_train = ftensor(y_i2[idx6, :])

    return xb_train, yb_train, ub_train, x_f1_train, y_f1_train, x_f2_train, y_f2_train,\
                x_f3_train, y_f3_train, x_i1_train, y_i1_train, x_i2_train, y_i2_train


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
