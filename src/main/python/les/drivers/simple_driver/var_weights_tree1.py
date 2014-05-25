import networkx as nx

from les.mp_model import MPModel
from les.mp_model import knapsack_model
from les.mp_model.mp_model_builder import mp_model_builder
from les.decomposers import decomposer_base
from les.mp_model.mp_solution import MPSolution
from les.graphs.decomposition_tree import Node
from les.graphs.decomposition_tree import DecompositionTree
from networkx.algorithms.traversal.depth_first_search import dfs_preorder_nodes
from networkx.algorithms.traversal.depth_first_search import dfs_edges
from operator import itemgetter as it
from les.utils import logging

class VarWeightsTree(DecompositionTree): #not necessarily
  
  def __init__(self, tree):
    if not isinstance(tree, DecompositionTree):
      raise TypeError('tree must be derived from DecompositionTree: %s' % tree)
    self._tree= tree
  
  def get_relaxed_solution(self, model): 
    vars_names = model.get_variables_names()
    list_of_vars = []  
    '''list_of_vars[i] = [var_name, var_obj_val, sum_of_coeff_var, sum_of_coeffs_in_all_constr_with_var]'''
    kmodel = knapsack_model.KnapsackModel(model)
    kmodel.mp_model_to_knapsack(model)
    constr = kmodel.get_weights()
    obj = model.get_objective_coefficients()
    
    for vn in vars_names:
      list_of_vars.append([vn])
              
    G = nx.DiGraph(self._tree)
    for node in dfs_preorder_nodes(G, self._tree.get_root()):   
      m = self._tree.get_model_by_name(node)
      msum = 0 # sum of coefficients of variables in corresponding model
      mvars = m.get_variables_names()
      for v in mvars:
        for i in range(len(list_of_vars)):
          if list_of_vars[i][0] == v:
            if len(list_of_vars[i]) == 1:
              list_of_vars[i] = list_of_vars[i] + [obj[i], constr[i], 0]
              msum += constr[i]
            break
      for v in mvars:
        for i in range(len(list_of_vars)):
          if list_of_vars[i][0] == v:
            list_of_vars[i][3] += msum
            break
    
    #print
    #model.pprint()
    #print

    #for v in list_of_vars:
    #  print v 
    #print 
            
    for edge_name in dfs_edges(G, self._tree.get_root()):
      edge = self._tree.get_edge_between(edge_name[0], edge_name[1])
      svars = edge.get_shared_variables()
      for v in svars:
        #print v, "SVARS"
        for i in range(len(list_of_vars)):
          if list_of_vars[i][0] == v:
            list_of_vars[i][3] -= constr[i]
            break
            
    new_list = []
    for v in list_of_vars:
      #print v[0], v[1], v[2], v[3]
      new_list.append([v[0], 1.0*v[1]*v[3]/v[2], 0.0])
    
    new_list.sort(key = it(1), reverse = True)
    #for v in new_list:
    #  print v[0], v[1]      

    res_list = []
    n = len(new_list)
    for i in range(n):
      if i < (n+1)/2:
        new_list[i][2] = 1.0
        res_list.append(new_list[i][0])
        #print new_list[i][0],
    #print   
    new_list.sort(key = it(0))
    
    solution = MPSolution()  
    solution.set_variables_values([_[0] for _ in new_list], [_[2] for _ in new_list])
    return res_list #solution
