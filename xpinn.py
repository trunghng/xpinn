from typing import List, Tuple, Callable
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon

from utils import ftensor, plot_loss
from network import XPINN


def train(Xb: List[Tuple[torch.Tensor, torch.Tensor]], ub: List[torch.Tensor],
        Xf: List[Tuple[torch.Tensor, torch.Tensor]], Xi: List[Tuple[torch.Tensor, torch.Tensor]],
        interfaces: List[Tuple[int, int]], f: Callable, f_aug_args: List[any],
        layers: List[List[int]], W_u: float, W_F: float, W_I: float, W_IF: float,
        epochs: int, lr: float, verbose: bool, save_model: bool, log_dir: str):
    model = XPINN(layers, [nn.Tanh] * len(layers))
    opt = Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    N_sd = len(model.subnets)

    # Convert from [[sd1_idx, sd2_idx], [sd1_idx, sd3_idx], [sd3_idx, sd4_idx]]
    # to [[0, 1], [0], [1, 2], [2]]
    interfaces_ = []
    for sd in range(N_sd):
        interfaces_.append([i for i in range(len(interfaces)) if sd in interfaces[i]])

    # Convert from [[sd1_idx, sd2_idx], [sd1_idx, sd3_idx], [sd3_idx, sd4_idx]]
    # to [[sd2_idx, sd3_idx], [sd1_idx], [sd1_idx, sd4_idx], [sd3_idx]]
    interfaces__ = []
    for sd in range(N_sd):
        interfaces__.append([itf[1 - itf.index(sd)] for itf in interfaces if sd in itf])

    for ep in range(epochs):
        u_preds = [model.subnets[q](Xb[q]) if Xb[q] else None for q in range(N_sd)]
        f_preds = [f(model.subnets[q], Xf[q][0], Xf[q][1], *f_aug_args) for q in range(N_sd)]
        ui_preds = [{
            p: model.subnets[p](Xi[i]),
            q: model.subnets[q](Xi[i])
        } for i, (p, q) in enumerate(interfaces)]
        u_avgs = [sum(uips.values()) / 2 for uips in ui_preds]
        losses_ = []

        for q in range(N_sd):
            mse_u = loss_fn(ub[q], u_preds[q]) if ub[q] is not None else 0
            mse_f = loss_fn(f_preds[q], torch.zeros_like(f_preds[q]))
            mse_uavg = sum([loss_fn(ui_preds[i][q], u_avgs[i]) for i in interfaces_[q]])
            mse_r = sum([loss_fn(f_preds[q], f_preds[q_plus]) for q_plus in interfaces__[q]])
            sd_loss = W_u * mse_u + W_F * mse_f + W_I * mse_uavg + W_IF * mse_r
            losses_.append(sd_loss)

        opt.zero_grad()
        loss = sum(losses_)
        loss.backward(retain_graph=True)
        opt.step()
        losses.append(list(map(lambda x: x.item(), losses_)))

        if verbose:
            log = f'Epoch {ep + 1:3d}'
            for q in range(N_sd):
                log += f' | Subnet{q + 1} loss {losses[-1][q]:,.5f}'
            print(log)

    plot_loss(np.asarray(losses).T, [f'subnet{i + 1} loss' for i in range(N_sd)] + ['total loss'],
                osp.join(log_dir, 'loss.png'))
    if save_model:
        torch.save(model.state_dict(), osp.join(log_dir, 'model.pth'))


def test(Xb, Xf, Xi, x_total, y_total, u_exact, layers: List[List[int]], model_path: str, log_dir: str):
    triang_total = tri.Triangulation(x_total.squeeze(), y_total.squeeze())
    X_fi1_train_Plot = np.concatenate(Xi[0], axis=-1)
    X_fi2_train_Plot = np.concatenate(Xi[1], axis=-1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # Draw exact solution
    tcf = ax1.tricontourf(triang_total, u_exact.squeeze(), 100 ,cmap='jet')
    tcbar = fig.colorbar(tcf)
    tcbar.ax.tick_params(labelsize=10)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('u (exact)')
    ax1.plot(X_fi1_train_Plot[:,0:1], X_fi1_train_Plot[:,1:2], 'w-', markersize=2)
    ax1.plot(X_fi2_train_Plot[:,0:1], X_fi2_train_Plot[:,1:2], 'w-', markersize=2)

    # Draw predict solution
    model = XPINN(layers, [nn.Tanh] * len(layers))
    model.load_state_dict(torch.load(model_path))
    u_preds = model.predict(Xf)
    u_preds = np.concatenate(u_preds)

    tcf = ax2.tricontourf(triang_total, u_preds.squeeze(), 100 ,cmap='jet')
    tcbar = fig.colorbar(tcf)
    tcbar.ax.tick_params(labelsize=10)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('u (predict)')
    ax2.plot(X_fi1_train_Plot[:,0:1], X_fi1_train_Plot[:,1:2], 'w-', markersize=2)
    ax2.plot(X_fi2_train_Plot[:,0:1], X_fi2_train_Plot[:,1:2], 'w-', markersize=2)
    plt.savefig(osp.join(log_dir, 'solution.png'))