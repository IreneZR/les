import networkx as nx

from les.mp_model import MPModel
from les.mp_model import knapsack_model
from les.mp_model.mp_model_builder import mp_model_builder
from les.decomposers import decomposer_base
from les.graphs.decomposition_tree import Node
from les.graphs.decomposition_tree import DecompositionTree
from networkx.algorithms.traversal.depth_first_search import dfs_preorder_nodes
from les.utils import logging

class VarWeightsTree(DecompositionTree):
  
  def __init__(self, tree):
    if not isinstance(tree, DecompositionTree):
      raise TypeError('model must be derived from MPModel: %s' % model)
    self._tree= tree
  
  def get_relaxed_solution(self, model): 
    vars_names = model.get_variables_names()
    list_of_vars = []    
    constr = knapsack_model.KnapsackModel(model).get_weights()
    
    for vn in vars_names:
      list_of_vars.append((vn,))
              
    G = nx.DiGraph(self._tree)
    for node in dfs_preorder_nodes(G, self._tree.get_root()):   
      m = node.get_model()
      msum = 0 # sum of coefficients of variables in corresponding model
      mvars = m.get_variables_names()
      for v in mvars:
        for i in range(list_of_vars):
          if list_of_vars[i][0] == v:
            if len(list_of_vars[i]) == 1:
              list_of_vars[i] = list_of_vars[i] + (m.get_variable_value_by_name(v), constr[i], 0,)
              msum += constr[i]
            break
      for v in mvars:
        for i in range(list_of_vars):
          if list_of_vars[i][0] == v:
            list_of_vars[i][3] += msum
            break
            
    for edge in dfs_edges(G, self._tree.get_root()):
      svars = edge.get_share_variables()
      for v in svars:
        for i in range(list_of_vars):
          if list_of_vars[i][0] == v:
            list_of_vars[i][3] -= constr[i]
            break
    for v in list_of_vars:
      print v      
    return 
