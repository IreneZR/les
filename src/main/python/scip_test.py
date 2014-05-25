import timeit

import os 

from les.ext.google.operations_research.linear_solver import pywraplp
from les.mp_model import MPModelBuilder
from les.backend_solvers.scip import SCIP

directory = '/home/ira/Desktop/imp/diploma/'
#tests = os.listdir(directory)
tests = ['p200x500_10_4_192890124.mps']
for t in tests:
  if t.endswith('.mps'):
    scip = SCIP()
    model = MPModelBuilder.build_from(directory + t)
    print t
    #'/home/ira/Documents/problems/i6_200x500.mps')
    scip.load_model(model)
    params = pywraplp.MPSolverParameters()
    params.SetIntegerParam(params.PRESOLVE, params.PRESOLVE_OFF)
    #params.set_quiet(True)
    start_time = timeit.default_timer()
    scip.solve(params)
    sol = scip.get_solution()
    print "Time:", timeit.default_timer() - start_time
    print sol.get_objective_value()
    for i in sol.get_variables_names():
      print i, sol.get_variable_value_by_name(i)
