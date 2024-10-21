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
This file contains useful functions to run the numerical experiments in Section
3 of:

R. I. Bot and E. Chenchene.
Extra-Gradient Method with Flexible Anchoring: Strong Convergence and
Fast Residual Decay,
2024. DOI: 10.48550/arXiv.2410.14369.

"""

import numpy as np
from scipy.ndimage import shift


def clip(x, maxim, minim):

    return np.minimum(np.maximum(x, -minim), maxim)


def proj_ball(x, radius, center):

    norm = np.linalg.norm(x - center)

    if norm > radius:
        return center + (x - center) / norm

    return x


def soft(w, lm):

    return np.sign(w) * np.maximum(np.abs(w) - lm, 0)


class Operator:

    def __init__(self, Mat, b, center, isLinear):

        self.Mat = Mat
        self.b = b
        self.dim = np.shape(Mat)[0]
        self.isLinear = isLinear

        if isLinear:
            self.L = np.linalg.norm(Mat)
        else:
            self.L = np.linalg.norm(Mat) + 1

        self.center = center

    def apply(self, x):

        if self.isLinear:
            return self.Mat @ x - self.b
        else:
            return self.Mat @ x - self.b + proj_ball(x, 1, self.center)


class Inifinite_dimensional_operator:

    def __init__(self, constant, off_set, dim):

        self.constant = constant
        self.off_set = off_set
        self.dim = dim
        self.L = 2

    def apply(self, x):

        return x - shift(x, self.off_set) - self.constant


class Resolvent:

    def __init__(self, prox_1, prox_2):

        self.prox_1 = prox_1
        self.prox_2 = prox_2

    def apply(self, w):

        x1 = self.prox1(w)
        x2 = self.prox2(2 * x1 - w)

        return w + x2 - x1
