# -*- coding: utf-8; -*-
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

"""The :class:`ProblemBase` represents base model class that is used by all
available models, such as :class:`~les.problems.bilp_problem.BILPProblem`,
:class:`~les.problems.knapsack_problem.KnapsackProblem`, etc.
"""

import types

class ProblemBase(object):
  """Base problem class for all available problem models.

  :param name: Optional string that represents problem name.
  """

  DEFAULT_NAME = "UNKNOWN"

  subproblem_name_format = "Z%d"

  def __init__(self, name=None):
    self._name = None
    if name:
      self.set_name(name)

  def set_name(self, name):
    """Sets problem name.

    :param name: A string that represents problem name.

    :raises: TypeError
    """
    if not type(name) is types.StringType:
      raise TypeError("name must be a string: %s" % type(name))
    self._name = name

  def get_name(self):
    """Returns the problem name. Returns `DEFAULT_NAME` if name wasn't defined.

    :returns: A string that represents problem's name.
    """
    return self._name or self.__class__.DEFAULT_NAME

  def build_subproblem(self, *args, **kwargs):
    """Builds and returns subproblem for the current problem instance. Pass all
    the arguments to the subproblem's constructor.

    :returns: A :class:`SubproblemBase` instance.
    """
    raise NotImplementedError()

class SubproblemBase(ProblemBase):
  """Base subproblem class for all available subproblem models."""

  pass
