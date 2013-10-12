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

from __future__ import with_statement

from les import mp_model
from les import generator
from les.utils import unittest

class GeneratorTest(unittest.TestCase):

  def test_generate_models(self):
    model = mp_model.build(
      [8, 2, 5, 5],
      [[2, 3, 4, 1],
       [1, 2, 3, 2]],
      ['L'] * 2,
      [7, 6]
    )
    with self.assert_raises(generator.Error):
      generator.Generator(model, (u'x3', u'x5'), (u'x0', u'x2'))
    g = generator.Generator(model, (u'x3', u'x4'), (u'x1', u'x2'))
    model_solution_pairs = list(g)
    self.assert_equal(4, len(model_solution_pairs))
    # Since we're doing maximization the first values for the given shared
    # variables should be (1.0, 1.0).
    relaxed_model, solution = model_solution_pairs[0]
    self.assert_equal([0.0, 0.0, 1.0, 1.0], solution.get_variables_values().tolist())

if __name__ == '__main__':
  unittest.main()
