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

import collections
import timeit
import numpy
import networkx as nx

from les import decomposers
from les import solution_tables
from les.mp_model import mp_solution
from les.drivers import driver_base
from les.drivers.compl_diff_driver import search_tree
from les.graphs.decomposition_tree import Node
from les.graphs.decomposition_tree import Edge
from les.graphs.decomposition_tree import DecompositionTree
from networkx.algorithms.traversal.depth_first_search import dfs_postorder_nodes
from les.ext.google.operations_research.linear_solver import pywraplp
from les import executors as executor_manager
from les.backend_solvers.scip import SCIP
from les.utils import logging


class _SolveContext(object):

  def __init__(self, submodel, candidate_model, solver_id_stack,
               partial_solution):
    self.submodel = submodel
    self.candidate_model = candidate_model
    self.best_objective_value = 0
    self.solver_id_stack = solver_id_stack
    self.partial_solution = partial_solution

  def set_best_objective_value(self, value):
    if (self.submodel.maximization() and value < self.best_objective_value):
      return False
    elif (not self.submodel.maximization() and
          value > self.best_objective_value):
      return False
    self.best_objective_value = value
    return True


class ComplDiffDriver(driver_base.DriverBase):

  def __init__(self, model, optimization_parameters, pipeline):
    super(ComplDiffDriver, self).__init__()
    if not model.is_binary():
      raise TypeError("Optimization can be applied only to binary integer "
                      "linear programming problems.")
    self._model = model
    self._pipeline = pipeline
    self._optimization_params = optimization_parameters
    self._driver_params = optimization_parameters.driver.compl_diff_driver_parameters
    self._executor = executor_manager.get_instance_of(optimization_parameters.executor.executor, self._pipeline)
    self._decomposer = decomposers.get_instance_of(self._driver_params.decomposer, model)
    logging.info("Decomposer: %s" % self._decomposer.__class__.__name__)
    self._solution_table = solution_tables.get_instance_of(self._driver_params.solution_table)
    if not self._solution_table:
      logging.info("Cannot create solution table: %d", self._driver_params.solution_table)
      return
    logging.info("Relaxation backend solvers are %s", self._driver_params.relaxation_backend_solvers)
    self._solver_id_stack = list(self._driver_params.relaxation_backend_solvers)
    self._solver_id_stack.append(optimization_parameters.driver.default_backend_solver)
    self._active_contexts = collections.OrderedDict()
    self._frozen_contexts = {}

  def _trivial_case(self):
    request = self._executor.build_request()
    request.set_model(self._model)
    request.set_solver_id(self._optimization_params.driver.default_backend_solver)
    response = self._executor.execute(request)
    return response.get_solution()

  def _process_decomposition_tree(self, tree):
    # TODO(d2rk): merge nodes if necessary.
    for node in tree.get_nodes():
      if node.get_num_shared_variables() > self._driver_params.max_num_shared_variables:
        logging.debug('Node %s has too many shared variables: %d > %d',
                      node.get_name(), node.get_num_shared_variables(),
                      self._driver_params.max_num_shared_variables)
        return False
    return True

  def start(self):
    start_time = timeit.default_timer()
    try:
      self._decomposer.decompose()
    except Exception, e:
      logging.exception("Decomposition failed.")
      return
    logging.info("Model was decomposed in %f second(s)."
                 % (timeit.default_timer() - start_time,))
    #print "\nDEC_TIME:", timeit.default_timer() - start_time
    #start_time = timeit.default_timer()
    tree = self._decomposer.get_decomposition_tree()
    #if not self._process_decomposition_tree(tree): ###uncomment!
    #  return
    if tree.get_num_nodes() == 1:
      return self._trivial_case()
    
    #self._search_tree = search_tree.SearchTree(tree)
    #self._solution_table.set_decomposition_tree(tree)
    #self.run()
    dvars = {"$1":0, "$2":0}
    vars_list = self._model.get_columns_names()
    for i in range(len(vars_list)):
      dvars[vars_list[i]] = i 
    
    G = nx.DiGraph(tree)
    models = []
    num = 0
    models.append([None, [], [], 0])
    for node_name in dfs_postorder_nodes(G, tree.get_root()): # pay attention to order (quasiblock)
      node = tree.node[node_name]
      m = node.get_model()
      num += m.get_num_rows()
      models.append([m, node.get_local_variables(), node.get_shared_variables(), num])
    
    new_list = []
    new_list.append([None, 0])
    k = 0
    for i in range(1, len(models) - 1):
      locvars = []
      shvars = []
      
      for j in models[i][1]:
        locvars.append(dvars[j])
      m1 = self._model.slice(range(models[i-1][3], models[i][3]), locvars)
      sm1 = sum(m1.get_objective_coefficients())
      
      for j in models[i][2]:
        if j in models[i+1][2]:
          shvars.append(dvars[j])
      m2 = self._model.slice(range(models[i-1][3], models[i+1][3]), shvars)
      sm2 = sum(m2.get_objective_coefficients())
      
      total = new_list[k-1][1] + sm1 + sm2
      if new_list[k][0] != None:
        #print models[i-1][0]
        new_list[k][0].update_rhs(new_list[k][1]*1.0/total, models[i-1][0].get_num_rows(), new_list[k][0].get_num_rows())
        #print models[i-1][0].get_num_rows(), new_list[k][0].get_num_rows(), "=)))"
      m1.update_rhs(sm1*1.0/total, 0, m1.get_num_rows())
      m2.update_rhs(sm2*1.0/total, 0, models[i][3] - models[i-1][3])
      
      new_list.append([m1, sm1])
      new_list.append([m2, sm2])
      k += 2
    
    locvars = []
    i = len(models) - 1
    for j in models[i][1]:
      locvars.append(dvars[j])
    m1 = self._model.slice(range(models[i-1][3], models[i][3]), locvars) 
    sm1 = sum(m1.get_objective_coefficients())
    total = new_list[k][1] + sm1
    new_list[k][0].update_rhs(new_list[k][1]*1.0/total, new_list[k-1][0].get_num_rows(), new_list[k][0].get_num_rows()) 
    m1.update_rhs(sm1*1.0/total, 0, models[len(models)-1][3] - models[len(models)-2][3])
    new_list.append([m1, sm1])
    
    res_vars = []
    num = 0
    for m in new_list:
      if m[0] != None:
        #m[0].pprint()
        #print m[1], "\n"
        request = self._executor.build_request()
        request.set_model(m[0])
        request.set_solver_id(self._optimization_params.driver.default_backend_solver)
        response = self._executor.execute(request)
        sol = response.get_solution()
        for i in range(sol.get_num_variables()):
          if sol.get_variables_values()[i] == 1.0:
            res_vars.append(sol.get_variables_names()[i])
            num += 1
    
    #print num, "\n", res_vars 
    prev_model = self._model
    mtrx = self._model.rows_coefficients.toarray()#
    varlist = self._model.get_columns_names()
    prevrhs = self._model.get_rows_rhs()
    #print "SOL_TIME:", timeit.default_timer() - start_time
    #start_time = timeit.default_timer()
    res = 0
    for i in res_vars:
      res += self._model.get_objective_coefficient_by_name(i)
    #print res_vars
    self._model = self._model.make_simple_model(res_vars, res_vars)     
    #self._model.pprint()
    scip = SCIP()
    scip.load_model(self._model)
    params = pywraplp.MPSolverParameters()
    params.SetIntegerParam(params.PRESOLVE, params.PRESOLVE_OFF)
    scip.solve(params)
    sol = scip.get_solution()
    for i in range(len(sol.get_variables_names())):
      if sol.get_variables_values()[i] == 1.0:
        res_vars.append(sol.get_variables_names()[i])
    for i in range(len(mtrx)):
      sm = 0
      for j in range(len(mtrx[i])):
        if varlist[j] in res_vars:
          sm += mtrx[i][j]
      if sm > prevrhs[i]:
        print "ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!=)"
    '''self._decomposer = decomposers.get_instance_of(self._driver_params.decomposer, self._model)
    self._decomposer.decompose()
    tree = self._decomposer.get_decomposition_tree()
    self._search_tree = search_tree.SearchTree(tree)
    self._solution_table.set_decomposition_tree(tree)
    print "DEC2_TIME:", timeit.default_timer() - start_time
    start_time = timeit.default_timer()
    self.run()'''
    #for v in var_list:
    #  if not v in sol.get_variables_names():
    #    if v in res_vars:
          
    res += sol.get_objective_value()
    
    list_res = []
    solution = mp_solution.MPSolution()
    for v in prev_model.get_variables_names():
      if v in res_vars:
        list_res.append(1.0)
      else:
        list_res.append(0.0)
    solution.set_variables_values(prev_model.get_variables_names(), list_res)
    solution.set_objective_value(res)
    #print "RES:",res
    #print "SOL_RUN_TIME:", timeit.default_timer() - start_time
    return solution
    #return sol
    #return self._solution_table.get_solution()

  def run(self):
    while True:
      if self._pipeline.has_responses():
        self.process_response(self._pipeline.get_response())
      if self._search_tree.is_empty():
        break
      if (self._search_tree.is_blocked() and len(self._active_contexts) == 0):
        continue
      if len(self._active_contexts) == 0:
        submodel, candidate_model, partial_solution = self._search_tree.next_unsolved_model()
        self._active_contexts[candidate_model.get_name()] = _SolveContext(
          submodel, candidate_model, list(self._solver_id_stack), partial_solution)
      name, cxt = self._active_contexts.popitem()
      request = self._pipeline.build_request()
      request.set_model(cxt.candidate_model)
      request.set_solver_id(cxt.solver_id_stack[0])
      self._frozen_contexts[name] = cxt
      self._pipeline.put_request(request)

  def process_response(self, response):
    cxt = self._frozen_contexts.pop(response.get_id())
    solution = response.get_solution()
    logging.debug("Process %s solution produced by %d with status %d",
                  response.get_id(), cxt.solver_id_stack[0], solution.get_status())
    cxt.solver_id_stack.pop(0)
    # Check F3: whether optimal solution has been found.
    if (not solution.get_status() is mp_solution.MPSolution.NOT_SOLVED
        and not solution.get_variables_values() is None):
      if ((sum(solution.get_variables_values().tolist()) % 1.0) == 0 and
          solution.is_optimal()):
        objective_value = solution.get_objective_value()
        # Check F2: do we need to continue?
        # NOTE: the best objective value will be checked inside of
        # set_best_objective_value().
        if cxt.set_best_objective_value(objective_value):
          logging.debug('Model %s has a new best objective value: %f',
                        response.get_id(), objective_value)
          cxt.partial_solution.update_variables_values(solution)
          cxt.partial_solution.set_objective_value(objective_value)
          logging.debug('Write %s solution to the table.', response.get_id())
          self._solution_table.write_solution(cxt.submodel, cxt.partial_solution)
        else:
          logging.debug("Solution is rejected: not the best objective value"
                        ": %f <= %f", objective_value, cxt.best_objective_value)
      else:
        logging.debug("Solution is rejected: "
                      "not optimal or fractional solution.")
    else:
      logging.debug("Solution is rejected: model wasn't solved.")
    if len(cxt.solver_id_stack) == 0:
      self._search_tree.mark_model_as_solved(cxt.candidate_model)
    else:
      self._active_contexts[cxt.candidate_model.get_name()] = cxt

  def get_solution(self):
    return self._solution_table.get_solution()
