import networkx as nx

from les.mp_model import MPModel
from les.mp_model import knapsack_model
from les.mp_model.mp_model_builder import mp_model_builder
from les.decomposers import decomposer_base
from les.graphs.decomposition_tree import Node
from les.graphs.decomposition_tree import DecompositionTree
from networkx.algorithms.traversal.depth_first_search import dfs_edges
from les.utils import logging

class MyTree(DecompositionTree):
  
  def __init__(self, tree):
    if not isinstance(tree, DecompositionTree):
      raise TypeError('model must be derived from MPModel: %s' % model)
    self._tree= tree
  
  def modify(self): 
    #list_of_models = self._tree.get_models()
    list_of_models = []
    names = []
    tmpbool = True   
    for node in self._tree.get_nodes():
      m = node.get_model()
      names.append(node.get_name())
      kmodel = knapsack_model.KnapsackModel(m)
      kmodel.mp_model_to_knapsack(m)
      new_model = mp_model_builder.MPModelBuilder().build_from_scratch(kmodel.get_profits(), [kmodel.get_weights()], ['L'], [kmodel.get_max_weight()], ["new_model"], [0 for j in range(m.get_num_columns())], [1 for j in range(m.get_num_columns())], m.columns_names)
      new_model.set_objective(kmodel.get_profits())
      new_model.set_name(m.get_name())
      list_of_models.append(new_model)
              
    G = nx.DiGraph(self._tree)
    for edge in dfs_edges(G, self._tree.get_root()):   
      fnode = edge[0]
      snode = edge[1]
      for i in range(len(names)):
        if names[i] == fnode:
          fmodel = list_of_models[i]
        elif names[i] == snode:
          smodel = list_of_models[i]
      if tmpbool:
        new_tree = DecompositionTree(self._tree._model)
        root = Node(fmodel)
        new_tree.set_root(root)
        tmpbool = False
      fbool = False
      if not fmodel in new_tree.get_models():
        new_tree.add_node(fmodel)
        fbool = True
      if not smodel in new_tree.get_models():
        new_tree.add_node(smodel)
        fbool = True
      if fbool: 
        shared_vars_names = []
        fvars = fmodel.columns_names
        svars = smodel.columns_names
        for p in fvars:
          if p in svars:
            shared_vars_names.append(p)
        new_tree.add_edge(fmodel, smodel, shared_vars_names)
    return new_tree
