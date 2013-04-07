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

import logging
import time

from les import runtime
from les.runtime import thread_pool
from les.solvers.local_elimination_solver.distributors.distributor import Distributor
from les.solvers.local_elimination_solver.local_solver import LocalSolver

class ThreadDistributor(Distributor):

  logger = logging.getLogger()

  def __init__(self, local_solver_settings, max_num_threads=0,
               reporter=None):
    Distributor.__init__(self, local_solver_settings)
    self._max_num_threads = 0
    self._reporter = reporter or self._default_reporter
    self.set_max_num_threads(max_num_threads or runtime.get_num_cpus())
    self._subproblems = []

  def __str__(self):
    return "%s[max_num_threads=%d, num_tasks=%d]" % (self.__class__.__name__,
                                                     self.get_max_num_threads(),
                                                     len(self._subproblems))

  def set_max_num_threads(self, n):
    if not isinstance(n, (int, long)):
      raise TypeError()
    self._max_num_threads = n

  def get_max_num_threads(self):
    return self._max_num_threads

  @classmethod
  def _default_reporter(cls, request, elapsed_time):
    subproblem = request.args[0]
    cls.logger.info("T%d: %s solved in %6.4f sec(s)"
                     % (request.requestID, subproblem, elapsed_time))

  def put(self, subproblem):
    """Put subproblems in order they come."""
    self._subproblems.append(subproblem)

  def _callback(self, subproblem):
    start = time.clock()
    solver = LocalSolver(self.get_local_solver_settings())
    solver.solve(subproblem)
    return time.clock() - start

  def run(self):
    self.logger.info("Running %s..." % self)
    pool = thread_pool.ThreadPool(self._max_num_threads)
    reqs = thread_pool.make_requests(self._callback, self._subproblems,
                                     self._reporter)
    for req in reqs:
      pool.put_request(req)
    pool.wait()
