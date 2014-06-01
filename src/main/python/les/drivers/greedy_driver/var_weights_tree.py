import numpy
import networkx as nx

from les.mp_model import MPModel
from les.mp_model import knapsack_model
from les.mp_model.mp_model_builder import mp_model_builder
from les.decomposers import decomposer_base
from les.mp_model.mp_solution import MPSolution
from les.graphs.decomposition_tree import Node
from les.graphs.decomposition_tree import DecompositionTree
from les.drivers.greedy_driver.to_knapsack import AddOnsMPM
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
    
    for ii in range(len(vars_names)):
      list_of_vars.append([vars_names[ii], 0, 0]) # name, weight, times
      dvars[vars_names[ii]] = ii
      
    G = nx.DiGraph(self._tree)
    for node in dfs_preorder_nodes(G, self._tree.get_root()):   
      m = self._tree.get_model_by_name(node)
      kvars, coeffs, krhs, w = AddOnsMPM().to_knapsack(m) # function, 4 - Sobj/Scon
      con_sum, obj_sum = 0, 0
      for i in range(len(kvars)):
        k = dvars[kvars[i]] # dvars - dictionary: key - index
        obj_sum += obj[i]
        con_sum += coeffs[i]
        list_of_vars[k][1] += krhs*obj[k]/coeffs[i] #w*krhs*obj[k]/coeffs[i]
        list_of_vars[k][2] += 1
      
    for v in list_of_vars:
      v[1] /= v[2] # making weights of shared_vars comparable to other vars
    list_of_vars.sort(key = it(1), reverse = True)
    
    res_list = []
    mm = len(list_of_vars)
    num_rows = model.get_num_rows()
    lhs = [0]*num_rows
    mtrx = model.get_rows_coefficients().toarray()
    fbool = False
    for jj in range(mm):
      for ii in range(num_rows):
        lhs[ii] += mtrx[ii][dvars[list_of_vars[jj][0]]]
        if lhs[ii] > model.get_rows_rhs()[ii]:
          fbool = True
          break
      if fbool:
        break
      res_list.append(list_of_vars[jj][0])
    return res_list
