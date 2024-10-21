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
This file contains an implementation of various anchored schemes utilized in
our numerical experiments in Section 3 of:

R. I. Bot and E. Chenchene.
Extra-Gradient Method with Flexible Anchoring: Strong Convergence and
Fast Residual Decay,
2024. DOI: 10.48550/arXiv.2410.14369.

"""

import numpy as np


def eg(init, M, the, maxit=100, x_opt=[]):

    # initialize
    x = np.copy(init)

    # storage
    if len(x_opt) != 0:
        Ds = np.zeros(maxit)
    Rs = np.zeros(maxit)

    for it in range(maxit):

        y = x - the * M.apply(x)
        x = x - the * M.apply(y)

        if len(x_opt) != 0:
            Ds[it] = np.sum((x - x_opt) ** 2)

        # measure residual
        res = np.sum(M.apply(x) ** 2)
        Rs[it] = res

    if len(x_opt) != 0:
        return Rs, Ds
    else:
        return Rs


def g_feg(init, M, eps_f, the, maxit=100, x_opt=[]):

    # initialize
    x = np.copy(init)

    # storage
    if len(x_opt) != 0:
        Ds = np.zeros(maxit)

    Rs = np.zeros(maxit)

    for it in range(maxit):

        eps = eps_f(it)
        eps_plus = eps_f(it + 1)

        y = (1 - the * eps) * x - the * M.apply(x)
        x = (x - the * M.apply(y)) / (1 + the * eps_plus)

        if len(x_opt) != 0:
            Ds[it] = np.sum((x - x_opt) ** 2)

        # measure residual
        res = np.sum(M.apply(x) ** 2)
        Rs[it] = res

    if len(x_opt) != 0:
        return Rs, Ds
    else:
        return Rs


def eag_c(init, M, maxit=100, x_opt=[]):

    the = 1 / (8 * M.L)

    # initialize
    x = np.copy(init)

    # storage
    if len(x_opt) != 0:
        Ds = np.zeros(maxit)

    Rs = np.zeros(maxit)

    for it in range(maxit):

        eps = 1 / (it + 2)

        y = (1 - eps) * x - the * M.apply(x)
        x = (1 - eps) * x - the * M.apply(y)

        if len(x_opt) != 0:
            Ds[it] = np.sum((x - x_opt) ** 2)

        # measure residual
        res = np.sum(M.apply(x) ** 2)
        Rs[it] = res

    if len(x_opt) != 0:
        return Rs, Ds
    else:
        return Rs


def eag_v(init, M, maxit=100, x_opt=[]):

    the = 0.5 / M.L

    # initialize
    x = np.copy(init)

    # storage
    if len(x_opt) != 0:
        Ds = np.zeros(maxit)

    Rs = np.zeros(maxit)

    for it in range(maxit):

        eps = 1 / (it + 2)

        y = (1 - eps) * x - the * M.apply(x)
        x = (1 - eps) * x - the * M.apply(y)

        if len(x_opt) != 0:
            Ds[it] = np.sum((x - x_opt) ** 2)

        # update step-size
        the = the * (1 - (the ** 2 * M.L ** 2) /
                     ((it + 1) * (it + 3) * (1 - the ** 2 * M.L ** 2)))

        # measure residual
        res = np.sum(M.apply(x) ** 2)
        Rs[it] = res

    if len(x_opt) != 0:
        return Rs, Ds
    else:
        return Rs


def feg(init, M, maxit=100, x_opt=[]):

    # epsilon
    eps_f = lambda k: 1 / (k + 1)

    # initialize
    x = np.copy(init)

    # storage
    if len(x_opt) != 0:
        Ds = np.zeros(maxit)

    Rs = np.zeros(maxit)

    for it in range(maxit):

        eps = eps_f(it)

        y = (1 - eps) * x - (1 - eps) * M.apply(x) / M.L
        x = (1 - eps) * x - M.apply(y) / M.L

        if len(x_opt) != 0:
            Ds[it] = np.sum((x - x_opt) ** 2)

        # measure residual
        res = np.sum(M.apply(x) ** 2)
        Rs[it] = res

    if len(x_opt) != 0:
        return Rs, Ds
    else:
        return Rs


def ppv(init, M, maxit=100, x_opt=[]):

    eta = 1 / (2 * M.L * np.sqrt(3))
    BigM = 4 * M.L ** 2

    # epsilon
    eps_f = lambda k: 1 / (k + 2)

    # initialize
    x = np.copy(init)
    y = np.copy(init)

    # storage
    if len(x_opt) != 0:
        Ds = np.zeros(maxit)
    Rs = np.zeros(maxit)

    for it in range(maxit):

        y_old = np.copy(y)
        eps = eps_f(it)

        y = (1 - eps) * x - (1 - eps) * eta * M.apply(y_old)
        x = (1 - eps) * x - eta * M.apply(y)

        # updating step-size
        eta = ((1 - eps ** 2 - BigM * eta ** 2) * eps_f(it + 1) * eta) / \
            ((1 - BigM * eta ** 2) * (1 - eps) * eps)

        if len(x_opt) != 0:
            Ds[it] = np.sum((x - x_opt) ** 2)

        # measure residual
        res = np.sum(M.apply(x) ** 2)
        Rs[it] = res

    if len(x_opt) != 0:
        return Rs, Ds
    else:
        return Rs
