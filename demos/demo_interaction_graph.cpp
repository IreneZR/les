/*
 * Copyright (c) 2012 Alexander Sviridenko
 */

#include <les/quasiblock_milp_problem.hpp>

/**
 * maximize 2x0 + 3x1 + x2 + 5x3 + 4x4 + 6x5 + x6
 * subject to
 * 3x0 + 4x1 +  x2                         <= 6
 *       2x1 + 3x2 + 3x3                   <= 5
 *       2x1             + 3x4             <= 4
 *             2x2             + 3x5 + 2x6 <= 5
 */

int
main()
{
  /* Vector of objective function coefficients */
  double c[] = {2.0, 3.0, 1.0, 5.0, 4.0, 6.0, 1.0};

  /* Matrix of constraints */
  double A[4][7] = {
    {3., 4., 1., 0., 0., 0., 0.},
    {0., 2., 3., 3., 0., 0., 0.},
    {0., 2., 0., 0., 3., 0., 0.},
    {0., 0., 2., 0., 0., 3., 2.},
  };

  /* Vector of rows sense */
  char s[] = {
    MILPP::ROW_SENSE_LOWER,
    MILPP::ROW_SENSE_LOWER,
    MILPP::ROW_SENSE_LOWER,
    MILPP::ROW_SENSE_LOWER,
  };

  /* Vector of right-hand side coefficients */
  double b[] = {
    6.0,
    5.0,
    4.0,
    5.0
  };

  /* Create quasi-block MILP problem by using predefined description.*/
  QBMILPP problem(c, 7, &A[0][0], 4, s, b);
  problem.print();

  /* Build interaction graph to check number of connected components. */
  InteractionGraph g(&problem);
  cout << "Interaction graph for a given problem:" << endl;
  g.dump();
  cout << "Interaction graph once vertex 2 was eliminated:" << endl;
  g.eliminate_vertex(2);
  g.dump();

  return 0;
}
