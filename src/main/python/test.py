import os
import timeit 

from les.mp_model import MPModel
from les.backend_solvers.scip import SCIP
from les.frontend_solver import FrontendSolver
from les.mp_model.optimization_parameters import OptimizationParameters
from les.ext.google.operations_research.linear_solver import pywraplp
from les.mp_model.mp_model_builder.formats import mps
from les.mp_model.mp_model_builder import MPModelBuilder

directory = '/home/ira/Desktop/imp/diploma/'
tests = os.listdir(directory)
for t in tests:
  if t.endswith('.mps'):
    model = MPModelBuilder.build_from(directory + t)
    
    print "\n\n", t, "\n"
    
    '''print "Les"
    params = OptimizationParameters()
    params.driver.driver = 0
    solver = FrontendSolver()
    solver.load_model(model)
    start_time = timeit.default_timer()
    sol1 = solver.solve(params)
    print sol1.get_objective_value(), timeit.default_timer() - start_time'''
    
    print "Scip"
    params = pywraplp.MPSolverParameters()
    params.SetIntegerParam(params.PRESOLVE, params.PRESOLVE_OFF)
    start_time = timeit.default_timer()
    scip = SCIP()
    scip.load_model(model)
    scip.solve(params)
    sol0 = scip.get_solution()
    print sol0.get_objective_value(), timeit.default_timer() - start_time
    
    names = sol0.get_variables_names()
    vals  = sol0.get_variables_values()
    
    '''print "Oracle"
    params = OptimizationParameters()
    params.driver.driver = 1
    solver = FrontendSolver()
    solver.load_model(model)
    start_time = timeit.default_timer()
    sol1 = solver.solve(params)
    print sol1.get_objective_value(), timeit.default_timer() - start_time
    tms = 0
    for i in names:
      if sol1.get_variable_value_by_name(i) != sol0.get_variable_value_by_name(i):
        tms += 1
    print "diff:", tms
  
    print "Simple"
    params = OptimizationParameters()
    params.driver.driver = 3
    solver = FrontendSolver()
    solver.load_model(model)
    start_time = timeit.default_timer()
    sol3 = solver.solve(params)
    print sol3.get_objective_value(), timeit.default_timer() - start_time
    tms = 0
    for i in names:
      if sol3.get_variable_value_by_name(i) != sol0.get_variable_value_by_name(i):
        tms += 1
    print "diff:", tms'''
  
    print "Compl_diff"
    params = OptimizationParameters()
    params.driver.driver = 5
    solver = FrontendSolver()
    solver.load_model(model)
    start_time = timeit.default_timer()
    sol5 = solver.solve(params)
    print sol5.get_objective_value(), timeit.default_timer() - start_time
    tms = 0
    for i in names:
      if sol5.get_variable_value_by_name(i) != sol0.get_variable_value_by_name(i):
        tms += 1
    print "diff:", tms
