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
#      2024. DOI: XX.YYYYY/arXiv.XXXX.YYYYY.
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
Run this script to obtain all Figures of:

R. I. Bot and E. Chenchene.
Extra-Gradient Method with Flexible Anchoring: Strong Convergence and
Fast Residual Decay,
2024. DOI: XX.YYYYY/arXiv.XXXX.YYYYY.

"""

import experiments as exp

if __name__ == '__main__':

    print('Testing rates numerically, generating Figure 2.1')
    exp.test_rate()

    print('Running Experiment 1, which generates Figure 3.1')
    exp.experiment_1(1000)

    print('Running Experiment 2, which generates Figure 3.2')
    exp.experiment_2(10000)

    print('Running Experiment 3, which generates Figure 3.3')
    exp.experiment_3(1000)
