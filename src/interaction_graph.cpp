/*
 * Copyright (c) 2012 Alexander Sviridenko
 */
#include <les/interaction_graph.hpp>
#include <les/packed_vector.hpp>

/* Required boost API */
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

using namespace boost;

/**
 * Compute and return the connected components of an undirected graph
 * using a DFS-based approach.
 *
 * Learn more:
 * http://www.boost.org/doc/libs/1_49_0/libs/graph/doc/connected_components.html
 */
vector<int>
InteractionGraph::get_connected_components()
{
  vector<int> components(get_num_vertices());
  int num = boost::connected_components(*this, &components[0]);

#if 0
  printf("Total number of components: %d\n", num);
  for (vector<int>::size_type i = 0; i != components.size(); ++i)
    printf("Vertex %d is in component %d\n", i, components[i]);
  printf("\n");
#endif

  return components;
}

InteractionGraph::InteractionGraph(MILPP* problem)
  : graph_t(problem->get_num_cols())
{
  int ri; /* row index */
  int ci; /* col index */
  int ni; /* neighbor index */
  PackedVector* row;

  problem_ = problem;

  /* Start forming variable neighbourhoods by using constraints */
  for (ri = 0; ri < problem->get_num_rows(); ri++)
    {
      row = problem->get_row(ri);
      for (ci = 0; ci < row->get_num_elements(); ci++)
        for (ni = 0; ni < row->get_num_elements(); ni++)
          {
            if ((ci == ni) || /* Skip variables with the same index */
                edge(row->get_index_by_pos(ci),
                     row->get_index_by_pos(ni), *this).second)
              continue;
            /* Add variable with index n as a neighbor for a
               variable with index c*/
            add_edge(row->get_index_by_pos(ci),
                     row->get_index_by_pos(ni), *this);
          }
    }
}
