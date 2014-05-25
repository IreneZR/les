import numpy
import networkx as nx

from les.mp_model import MPModel
from les.mp_model import knapsack_model
from les.mp_model.mp_model_builder import mp_model_builder
from les.decomposers import decomposer_base
from les.mp_model.mp_solution import MPSolution
from les.graphs.decomposition_tree import Node
from les.graphs.decomposition_tree import DecompositionTree
from les.drivers.simple_driver.to_knapsack import AddOnsMPM
from networkx.algorithms.traversal.depth_first_search import dfs_preorder_nodes
from networkx.algorithms.traversal.depth_first_search import dfs_edges
from operator import itemgetter as it
from les.utils import logging

class VarWeightsTree(DecompositionTree):
  
  def __init__(self, tree):
    if not isinstance(tree, DecompositionTree):
      raise TypeError('model must be derived from MPModel: %s' % model)
    self._tree= tree
  
  def get_relaxed_solution(self, model): 
    vars_names = model.get_variables_names()
    list_of_vars = []  
    obj = model.get_objective_coefficients()
    dvars = {"$1":0, "$2":0}
    vars_rhs_cf = [] # the i-th element is the indexes of the blocks, where i-th var is, and coeffs
    
    for ii in range(len(vars_names)):
      list_of_vars.append([vars_names[ii], 0, 0]) # name, [obj, coeffs, rhs], times
      dvars[vars_names[ii]] = ii
      vars_rhs_cf.append([])
      
    num_cols = model.get_num_columns()
    rhs = [] # rhs of knapsack problems in nodes
    G = nx.DiGraph(self._tree)
    ii = 0
    for node in dfs_preorder_nodes(G, self._tree.get_root()):   
      m = self._tree.get_model_by_name(node)
      kvars, coeffs, krhs, w = AddOnsMPM().to_knapsack(m) # function, 4 - Sobj/Scon
      rhs.append(krhs)
      for i in range(len(kvars)):
        k = dvars[kvars[i]] # dvars - dictionary: key - index
        vars_rhs_cf[k].append(ii)
        vars_rhs_cf[k].append(coeffs[i])
        list_of_vars[k][1] += obj[k]/coeffs[i] #w*krhs*obj[k]/coeffs[i]
        list_of_vars[k][2] += 1
      ii += 1
    for ii in range(len(list_of_vars)):
      print list_of_vars[ii][0], list_of_vars[ii][1], 
      for jj in range(0, len(vars_rhs_cf[ii]), 2):
        print rhs[vars_rhs_cf[ii][jj]],
      print "\n"
    print list_of_vars, "\n",
    res_list = []
    used = [False]*num_cols
    num_rows = model.get_num_rows()
    lhs = [0]*num_rows
    mtrx = model.get_rows_coefficients().toarray()
    fbool = False
    print rhs, "\n"
    for jj in range(num_cols):
      cur_var = self.next_var(list_of_vars, vars_rhs_cf, rhs, used, num_cols) #
      print "\ncur_var", cur_var
      for ii in range(num_rows):
        lhs[ii] += mtrx[ii][dvars[cur_var]]
        if lhs[ii] > model.get_rows_rhs()[ii]:
          fbool = True
          break
      if fbool:
        break
      res_list.append(cur_var)
      ii = 0
      while ii < len(vars_rhs_cf[k]):
        rhs[vars_rhs_cf[k][ii]] -= vars_rhs_cf[k][ii+1]
        ii += 2
      used[dvars[cur_var]] = True
      print "rhs", rhs, "\n"
    #print res_list
    return res_list
    
  def next_var(self, list_of_vars, vars_rhs_cf, rhs, used, num_cols):
    #print "=)))"
    mx = 0.0
    var_name = ''
    #print rhs
    #print vars_rhs_cf
    for jj in range(num_cols):
      if not used[jj]:
        tmp, ii = 0, 0
        while ii < len(vars_rhs_cf[jj]):
          #print len(vars_rhs_cf[jj]), jj, ii, 
          #print vars_rhs_cf[jj][ii]
          #print "=)"
          tmp += list_of_vars[jj][1]*rhs[vars_rhs_cf[jj][ii]]
          ii += 2
        tmp /= list_of_vars[jj][2]
        print "[", list_of_vars[jj][0], tmp, "] ",
        if mx < tmp:
          mx = tmp
          var_name = list_of_vars[jj][0]
    return var_name
