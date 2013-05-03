#!/usr/bin/env python
#
# Copyright (c) 2012-2013 Oleksandr Sviridenko
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from les.problems import BILPProblem
from les.decomposers.finkelstein_qb_decomposer import FinkelsteinQBDecomposer
from les.ext.coin import OsiSymSolverInterfaceFactory

from les.solvers.local_elimination_solver.local_elimination_solver import LocalEliminationSolver

class _OsiSymSolverInterfaceFactory(OsiSymSolverInterfaceFactory):

  def build(self):
    si = OsiSymSolverInterfaceFactory.build(self)
    si.set_sym_param("verbosity", -2)
    return si

class LocalEliminationSolverTest(unittest.TestCase):

  def test_solve1(self):
    problem = BILPProblem.build_from_scratch(
      [8, 2, 5, 5, 8, 3, 9, 7, 6],
      [[2., 3., 4., 1., 0., 0., 0., 0., 0.],
       [1., 2., 3., 2., 0., 0., 0., 0., 0.],
       [0., 0., 1., 4., 3., 4., 2., 0., 0.],
       [0., 0., 2., 1., 1., 2., 5., 0., 0.],
       [0., 0., 0., 0., 0., 0., 2., 1., 2.],
       [0., 0., 0., 0., 0., 0., 3., 4., 1.]],
      [7, 6, 9, 7, 3, 5]
    )
    decomposer = FinkelsteinQBDecomposer()
    decomposer.decompose(problem)
    solver = LocalEliminationSolver()
    solver.params.master_solver_factory = _OsiSymSolverInterfaceFactory()
    solver.load_problem(problem, decomposer.get_decomposition_tree())
    solver.solve()
    self.assertEqual(39.0, solver.get_obj_value())
    self.assertEqual([1., 0., 1., 1., 1., 0., 0., 1., 1.], solver.get_col_solution())

  def test_solve2(self):
    problem = BILPProblem.build_from_scratch(
      [2, 3, 1, 5, 4, 6, 1],
      [[3., 4., 1., 0., 0., 0., 0.],
       [0., 2., 3., 3., 0., 0., 0.],
       [0., 2., 0., 0., 3., 0., 0.],
       [0., 0., 2., 0., 0., 3., 2.]],
      [6, 5, 4, 5]
    )
    decomposer = FinkelsteinQBDecomposer()
    decomposer.decompose(problem)
    tree = decomposer.get_decomposition_tree()
    subproblems = tree.get_subproblems()
    self.assertEqual(2, len(subproblems))
    solver = LocalEliminationSolver()
    solver.params.master_solver_factory = _OsiSymSolverInterfaceFactory()
    solver.load_problem(problem, decomposer.get_decomposition_tree())
    solver.solve()
    self.assertEqual(18.0, solver.get_obj_value())
    self.assertEqual([1., 0., 0., 1., 1., 1., 1.], solver.get_col_solution())

if __name__ == '__main__':
  unittest.main()
