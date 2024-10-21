# -*- coding: utf-8 -*-
#
#    Copyright (C) 2024 Radu Ioan Bot (radu.bot@univie.ac.at)
#                       Enis Chenchene (enis.chenchene@univie.ac.at)
#
#    This file is part of the example code repository for the paper:
#
#      R. I. Bot and E. Chenchene.
#      Extra-Gradient Method with Flexible Anchoring: Strong Convergence and
#      Fast Residual Decay,
#      2024. DOI: 10.48550/arXiv.2410.14369.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
This file contains the implementation of our numerical experiments in Section 3
of the paper:

R. I. Bot and E. Chenchene.
Extra-Gradient Method with Flexible Anchoring: Strong Convergence and
Fast Residual Decay,
2024. DOI: 10.48550/arXiv.2410.14369.

"""

import numpy as np
import matplotlib.pyplot as plt
import structures as st
import optimize as opt
from matplotlib import rc
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

fonts = 20
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],
              'size': fonts})
rc('text', usetex=True)


def experiment_1(maxit):

    np.random.seed(1)

    # defining operator
    dim = 10
    Mat = np.zeros((dim, dim))
    Mat[int(dim / 2):, :int(dim / 2)] = -np.eye(int(dim / 2))
    Mat[:int(dim / 2), int(dim / 2):] = np.eye(int(dim / 2))
    x_opt = 10 * np.random.rand(dim)
    b = Mat @ x_opt
    center = x_opt
    M = st.Operator(Mat, b, center, False)

    # step-size for g-eag
    the = (1 - 1e-9) / M.L

    # initial point
    init = 10 * np.random.rand(M.dim)

    # testing the following alphas
    alphas_low = np.linspace(1e-3, 1, 10)
    alphas_lar = np.linspace(1, 2, 10)[1:]
    alphas = np.hstack((alphas_low, alphas_lar))
    alphas.sort()

    # testing the following betas
    Betas = np.array([[1, 2],
                      [5, 50]])

    # intialize plots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    cmap = LinearSegmentedColormap.from_list("WhiteBlue", [(1, 1, 0),
                                                           (1, 0.5, 0),
                                                           (1, 0, 0)])
    norm = Normalize(alphas[0], alphas[-1])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for i in range(2):
        for j in range(2):

            beta = Betas[i, j]
            axs[i, j].loglog([1 / (k + 1) for k in range(maxit)],
                             '--', color='k', alpha=0.3)
            axs[i, j].loglog([1 / (k + 1) ** 2 for k in range(maxit)],
                             '--', color='k', alpha=0.3)

            # competitors
            Rs_eag_c = opt.eag_c(init, M, maxit)
            axs[i, j].loglog(Rs_eag_c, linewidth=3, label='EAG-C', color='k')

            Rs_eag_v = opt.eag_v(init, M, maxit)
            axs[i, j].loglog(Rs_eag_v, linewidth=3, label='EAG-V', color='gray')

            Rs_feg = opt.feg(init, M, maxit)
            axs[i, j].loglog(Rs_feg, linewidth=3, label='FEG', color='b')

            Rs_ppv = opt.ppv(init, M, maxit)
            axs[i, j].loglog(Rs_ppv,  linewidth=3, label='APV', color='violet')

            # proposed method
            for case, alpha in enumerate(alphas):

                # regularization parameter
                eps_f = lambda k: alpha / (the * (k + beta))
                Rs_g_feg = opt.g_feg(init, M, eps_f, the, maxit)

                color = cmap(norm(alpha))

                if alpha == 1:
                    axs[i, j].loglog(Rs_g_feg, color=color, linewidth=3)
                else:
                    axs[i, j].loglog(Rs_g_feg, color=color, alpha=0.5)

            if i == 1:
                axs[i, j].set_xlabel('Iteration number ' + r'$(k)$')

            if j == 0:
                axs[i, j].set_ylabel('Residual ' + r'$\|M(x^k)\|^2$')

            axs[i, j].set_title(rf'$\beta = {int(beta)}$')
            axs[i, j].set_xlim(1, maxit)
            axs[i, j].set_ylim(1e-6, 1e3)

            if i == 1 and j == 1:
                axs[i, j].legend(ncols=1, bbox_to_anchor=(1.7, 1.4))

    cbar_ax = fig.add_axes([0.25, 0.03, 0.5, 0.01])
    cbar = plt.colorbar(sm, ticks=[0.01] + list(np.round(alphas[0:-1:3], 2))
                        + [np.round(alphas[-1], 2)],
                        anchor=(2.0, 0), cax=cbar_ax, orientation="horizontal")
    cbar.set_label('Algorithm 1 with different choices of ' + r'$\alpha $')
    plt.savefig('results/experiment_1.pdf',  bbox_inches='tight')

    plt.show()


def experiment_2(maxit):

    np.random.seed(1)

    # defining operator
    dim = 10
    Mat = np.zeros((dim, dim))
    Mat[int(dim / 2):, :int(dim / 2)] = -np.eye(int(dim / 2))
    Mat[:int(dim / 2), int(dim / 2):] = np.eye(int(dim / 2))
    x_opt = 10 * np.random.rand(dim)
    b = Mat @ x_opt
    center = x_opt
    M = st.Operator(Mat, b, center, False)

    # lipschitz constant
    if M.isLinear:
        L = np.linalg.norm(M.Mat)
    else:
        L = np.linalg.norm(M.Mat) + 1

    # step-size for g-eag
    the = (1 - 1e-9) / L

    # initial point
    init = 10 * np.random.rand(M.dim)

    # testing the following alphas
    alphas_low = np.linspace(1e-3, 1 / the, 10)
    alphas_lar = np.linspace(1 / the, 2 / the, 10)[1:]
    alphas = np.hstack((alphas_low, alphas_lar))
    alphas.sort()

    # testing the following betas
    Betas = np.array([[1, 2],
                      [5, 50]])

    # intialize plots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    for i in range(2):
        for j in range(2):

            beta = Betas[i, j]
            axs[i, j].loglog([1 / (k + 1) for k in range(maxit)],
                             '--', color='k', alpha=0.3)
            axs[i, j].loglog([1 / (k + 1) ** 2 for k in range(maxit)],
                             '--', color='k', alpha=0.3)

            # competitors
            Rs_eag_c = opt.eag_c(init, M, maxit)
            axs[i, j].loglog(Rs_eag_c, linewidth=3, label='EAG-C', color='k')

            Rs_eag_v = opt.eag_v(init, M, maxit)
            axs[i, j].loglog(Rs_eag_v, linewidth=3, label='EAG-V', color='gray')

            Rs_feg = opt.feg(init, M, maxit)
            axs[i, j].loglog(Rs_feg, linewidth=3, label='FEG', color='b')

            Rs_ppv = opt.ppv(init, M, maxit)
            axs[i, j].loglog(Rs_ppv,  linewidth=3, label='APV', color='violet')

            eps_f = lambda k: (2 / np.pi * np.arctan(1e-3 * k) / the) / \
                (k + beta) + np.random.normal(0, 1 / (k + 1))
            Rs_g_feg = opt.g_feg(init, M, eps_f, the, maxit)
            axs[i, j].loglog(Rs_g_feg,  linewidth=1, label='G-EAG-C2',
                             color='orange')

            # proposed method
            eps_f = lambda k : (2  / np.pi * np.arctan(1e-3 * k) / the) / \
                (k + beta)
            Rs_g_feg = opt.g_feg(init, M, eps_f, the, maxit)
            axs[i, j].loglog(Rs_g_feg,  linewidth=3, label='G-EAG-C1',
                             color='red')

            if i == 1:
                axs[i, j].set_xlabel('Iteration number ' + r'$(k)$')

            if j == 0:
                axs[i, j].set_ylabel('Residual ' + r'$\|M(x^k)\|^2$')

            axs[i, j].set_title(rf'$\beta = {int(beta)}$')
            axs[i, j].set_xlim(1, maxit)
            axs[i, j].set_ylim(1e-5, 1e3)
            # axs[i, j].grid(which='both')

            if i == 1 and j == 1:
                axs[i, j].legend(ncols=1, bbox_to_anchor=(1, 1.5))

    plt.savefig('results/experiment_2.pdf', bbox_inches='tight')
    plt.show()


def experiment_3(maxit):
    '''
    Infinte dimensional example
    '''
    np.random.seed(2)

    # operator
    off_set = 1
    dim = 2 * maxit + 2
    constant = np.zeros(dim)
    constant[0] = 1
    constant[1] = - 1
    M = st.Inifinite_dimensional_operator(constant, off_set, dim)
    x_opt = np.zeros(M.dim)
    x_opt[0] = 1

    # step-size for g-eag
    the = 0.8 / M.L

    # initial point
    init = np.zeros(M.dim)
    init[0] = 0.9
    init[2] = 1

    # testing the following alphas
    alphas_low = np.linspace(1e-3, 1, 10)
    alphas_lar = np.linspace(1, 2, 10)[1:]
    alphas = np.hstack((alphas_low, alphas_lar))
    alphas.sort()

    # testing the following betas
    Betas = np.array([[1, 2],
                      [5, 50]])

    # intialize plots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    fig_d, axs_d = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    cmap = LinearSegmentedColormap.from_list("WhiteBlue", [(1, 1, 0),
                                                           (1, 0.5, 0),
                                                           (1, 0, 0)])
    norm = Normalize(alphas[0], alphas[-1])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for i in range(2):
        for j in range(2):

            beta = Betas[i, j]
            axs[i, j].loglog([1 / (k + 1) for k in range(maxit)],
                             '--', color='k', alpha=0.3)
            axs[i, j].loglog([1 / (k + 1) ** 2 for k in range(maxit)],
                             '--', color='k', alpha=0.3)

            axs_d[i, j].loglog([1 / (k + 1) for k in range(maxit)], '--',
                               color='k', alpha=0.3)
            axs_d[i, j].loglog([1 / (k + 1) ** 2 for k in range(maxit)], '--',
                               color='k', alpha=0.3)

            # competitors
            Rs_eg, Ds_eg = opt.eg(init, M, the, maxit, x_opt)
            axs[i, j].loglog(Rs_eg, linewidth=3, label='EG', color='green')
            axs_d[i, j].loglog(Ds_eg, linewidth=3, label='EG', color='green')

            Rs_eag_c, Ds_eag_c = opt.eag_c(init, M, maxit, x_opt)
            axs[i, j].loglog(Rs_eag_c, linewidth=3, label='EAG-C', color='k')
            axs_d[i, j].loglog(Ds_eag_c, linewidth=3, label='EAG-C', color='k')

            Rs_eag_v, Ds_eag_v = opt.eag_v(init, M, maxit, x_opt)
            axs[i, j].loglog(Rs_eag_v, linewidth=3, label='EAG-V', color='gray')
            axs_d[i, j].loglog(Ds_eag_v, linewidth=3, label='EAG-V', color='gray')

            Rs_feg, Ds_feg = opt.feg(init, M, maxit, x_opt)
            axs[i, j].loglog(Rs_feg, linewidth=3, label='FEG', color='b')
            axs_d[i, j].loglog(Ds_feg, linewidth=3, label='FEG', color='b')

            Rs_ppv, Ds_ppv = opt.ppv(init, M, maxit, x_opt)
            axs[i, j].loglog(Rs_ppv,  linewidth=3, label='APV', color='violet')
            axs_d[i, j].loglog(Ds_ppv, linewidth=3, label='APV', color='violet')

            # proposed method
            for case, alpha in enumerate(alphas):

                # regularization parameter
                eps_f = lambda k: alpha / (k + beta)
                Rs_g_feg, Ds_g_feg = opt.g_feg(init, M, eps_f, the, maxit, x_opt)

                color = cmap(norm(alpha))

                if alpha == 1:
                    axs[i, j].loglog(Rs_g_feg, color=color, linewidth=3, alpha=1)
                    axs_d[i, j].loglog(Ds_g_feg, color=color, linewidth=3, alpha=1)
                else:
                    axs[i, j].loglog(Rs_g_feg, color=color, alpha=0.5)
                    axs_d[i, j].loglog(Ds_g_feg, color=color, alpha=0.5)

            if i == 1:
                axs[i, j].set_xlabel('Iteration number ' + r'$(k)$')
                axs_d[i, j].set_xlabel('Iteration number ' + r'$(k)$')

            if j == 0:
                axs[i, j].set_ylabel('Residual ' + r'$\|M(x^k)\|^2$')
                axs_d[i, j].set_ylabel('Distance ' + r'$\|x^k - x^*\|^2$')

            axs[i, j].set_title(rf'$\beta = {int(beta)}$')
            axs[i, j].set_xlim(1, maxit)
            axs[i, j].set_ylim(1e-6, 1e1)

            if i == 1 and j == 1:
                axs[i, j].legend(ncols=1, bbox_to_anchor=(2.1, 1.4))

            axs_d[i, j].set_title(rf'$\beta = {int(beta)}$')
            axs_d[i, j].set_xlim(1, maxit)
            axs_d[i, j].set_ylim(1e-3, 1e0)
            # axs_d[i, j].grid(which='both')

    cbar_ax = fig.add_axes([0.95, 0.2, 0.01, 0.5])
    cbar = plt.colorbar(sm, ticks=[0.01] + list(np.round(alphas[0:-1:3], 2))
                        + [alphas[-1]],
                        anchor=(2.0, 0), cax=cbar_ax, orientation="vertical")
    cbar.set_label('Algorithm 1 with different choices of ' + r'$\alpha $')
    fig_d.savefig('results/experiment_3_dists.pdf',  bbox_inches='tight')
    fig.savefig('results/experiment_3_resid.pdf',  bbox_inches='tight')

    fig_d.show()
    fig.show()

    return


def test_rate():

    maxit = 100000

    Eps = [lambda k: (10 / np.pi * np.arctan(1e-1 * k + 1e-3)) / (k + 2),
           lambda k: 0.9 / (k + 10) * np.log(k + 2),
           lambda k: 3 / ((k + 10) * np.log(k + 2)),
           lambda k: 1 / np.log(1e2 * k + 2)]

    Titles = [r'$\varepsilon^k \simeq \frac{2\alpha}{\pi}\frac{\arctan(k)}{\theta k}$',
              r'$\varepsilon^k \simeq \frac{1}{k}\log(k)$',
              r'$\varepsilon^k \simeq \frac{1}{k\log(k)}$',
              r'$\varepsilon^k \simeq \frac{1}{\log(k)}$']

    fig, axs = plt.subplots(2, 4, figsize=(15, 5), sharex=True, sharey='row',
                            height_ratios=[1, 3])
    domain = np.arange(maxit)

    # plotting functions
    for num_case in range(4):

        eps_f = Eps[num_case]
        axs[0, num_case].loglog(domain, eps_f(domain), color='r', linewidth=3)
        axs[0, num_case].set_title(Titles[num_case], pad=20)
        axs[0, num_case].set_xlim(1, maxit)

    for num_case in range(4):

        eps_f = Eps[num_case]
        Rs = np.zeros(maxit)
        r = 0

        for it in range(maxit):

            h = ((eps_f(it + 1) - eps_f(it)) / eps_f(it)) ** 2
            r = r * (1 - eps_f(it)) + eps_f(it) * h

            Rs[it] = r

        axs[1, num_case].loglog(Rs, linewidth=3, color='b')
        axs[1, num_case].loglog([1 / (k + 1) for k in range(maxit)], '--',
                                color='k', alpha=0.5, label=r'$O(k^{-1})$')
        axs[1, num_case].loglog([1 / (k + 1) ** 2 for k in range(maxit)], '--',
                                color='k', alpha=0.5, label=r'$O(k^{-2})$')
        axs[1, num_case].loglog([1 / np.log(k + 2) for k in range(maxit)],
                                '--', color='k', alpha=0.5,
                                label=r'$O(\log(k)^{-2})$')
        axs[1, num_case].set_ylim(1e-9, 1e2)
        axs[1, num_case].set_xlim(1, maxit)
        axs[1, num_case].set_xlabel('Iteration number ' + r'$(k)$')

    axs[0, 0].set_ylabel(r'$\varepsilon^k$')
    axs[1, 0].set_ylabel(r'$r^k$')

    # builiding legend
    axs[1, 3].text(4 * 1e3, 1 / (4 * 1e3), r'$k^{-1}$', rotation=-25,
                   fontsize=15)
    axs[1, 3].text(5 * 1e3, 1 / (5 * 1e3) ** 2, r'$k^{-2}$', rotation=-30,
                   fontsize=15)
    axs[1, 3].text(1e3, 2 / np.log(1e3), r'$\log(k)^{-1}$', rotation=-2,
                   fontsize=15)

    plt.savefig('results/rates.pdf', bbox_inches='tight')
    plt.show()
