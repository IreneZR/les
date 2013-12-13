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

import timeit

from les import backend_solvers as backend_solver_manager
from les import mp_model
from les.pipeline import Pipeline
from les.frontend_solver.driver import Driver
from les import mp_solver_base
from les import executors as executor_manager
from les import decomposers as decomposer_manager
from les import solution_tables as solution_table_manager
from les.utils import logging

class Error(Exception):
  pass

class FrontendSolver(mp_solver_base.MPSolverBase):
  '''This class implements the optimization logic of local elimination
  solver.
  '''

  def __init__(self):
    self._model = None
    self._optimization_params = None
    self._executor = None
    self._pipeline = None

  @classmethod
  def _finalize_optimization_parameters(self, params):
    if not params.HasField('default_backend_solver'):
      default_solver_id = backend_solver_manager.get_default_solver_id()
      if not default_solver_id:
        raise Error('Cannot define default backend solver id.')
      params.default_backend_solver = default_solver_id
    return params

  def _process_decomposition_tree(self, tree):
    # TODO(d2rk): merge nodes if necessary.
    for node in tree.get_nodes():
      if node.get_num_shared_variables() > self._optimization_params.max_num_shared_variables:
        logging.debug('Node %s has too many shared variables: %d > %d',
                      node.get_name(), node.get_num_shared_variables(),
                      self._optimization_params.max_num_shared_variables)
        return False
    return True

  def _solve_single_model(self):
    request = self._executor.build_request()
    request.set_model(mp_model_parameters.build(self._model))
    request.set_solver_id(self._optimization_params.default_backend_solver)
    response = self._executor.execute(request)
    self._set_solution(response.get_solution())

  def get_model(self):
    '''Returns model solved by this solver.

    :returns: A :class:`~les.mp_model.mp_model.MPModel` instance.
    '''
    return self._model

  def load_model(self, model):
    if not isinstance(model, mp_model.MPModel):
      raise TypeError()
    if not model.is_binary():
      raise TypeError('Optimization can be applied only to binary integer '
                      'linear programming problems.')
    self._model = model

  def solve(self, params=None):
    if not self._model:
      raise Error()
    if params and not isinstance(params, mp_model.OptimizationParameters):
      raise TypeError()
    if not params:
      params = mp_model.OptimizationParameters()
    self._finalize_optimization_parameters(params)
    self._optimization_params = params
    logging.info('Optimize model %s with %d rows and %d columns.',
                 self._model.get_name(), self._model.get_num_rows(),
                 self._model.get_num_columns())
    start_time = timeit.default_timer()
    try:
      decomposer = decomposer_manager.get_instance_of(params.decomposer,
                                                      self._model)
      logging.info("Decomposer: %s" % decomposer.__class__.__name__)
      decomposer.decompose()
    except Exception, e:
      logging.exception('Decomposition has been failed.')
      return
    logging.info("Model was decomposed in %f second(s)."
                 % (timeit.default_timer() - start_time,))
    tree = decomposer.get_decomposition_tree()
    if not self._process_decomposition_tree(tree):
      return
    self._pipeline = Pipeline()
    self._executor = executor_manager.get_instance_of(params.executor, self._pipeline)
    logging.debug('Executor: %s', self._executor.__class__.__name__)
    if tree.get_num_nodes() == 1:
      return self._solve_single_model()
    solution_table = solution_table_manager.get_instance_of(params.solution_table)
    if not solution_table:
      logging.info('Cannot create solution table: %d', params.solution_table)
      return
    solution_table.set_decomposition_tree(tree)
    self._driver = Driver(params, self._pipeline, decomposition_tree=tree,
                          solution_table=solution_table)
    logging.info('Default backend solver: %d', params.default_backend_solver)
    logging.info('Relaxation backend solvers: %s',
                 params.relaxation_backend_solvers)
    try:
      self._executor.start()
      start_time = timeit.default_timer()
      self._driver.run()
      self._executor.stop()
    except KeyboardInterrupt:
      self._executor.stop()
      return
    except Exception, e:
      logging.exception("Driver failed.")
      self._executor.stop()
      return
    logging.info("Model was solved in %f second(s)"
                 % (timeit.default_timer() - start_time,))
    self._model.set_solution(solution_table.get_solution())
