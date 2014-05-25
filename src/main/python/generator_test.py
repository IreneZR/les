#!/usr/bin/env python
#
# Copyright (c) 2013 Oleksandr Sviridenko

from __future__ import print_function

import glob
import os
import time
import timeit
import sys
import argparse

from les.mp_model.mp_model_generators.qbbilp_model_generator import QBBILPModelGenerator
from les.mp_model.mp_model_builder.formats.mps.encoder import Encoder
from les.decomposers.finkelstein_qb_decomposer import FinkelsteinQBDecomposer
from les.utils import unittest

def do_benchmarking(num_variables, num_constraints, num_blocks, separator_size,
                    num_iterations, problems_dirname, report_filename):
  name_format = 'p%dx%d_%d_%d_%d' # n, m, k, s
  if not os.path.exists(problems_dirname):
    raise IOError("%s doesn't exist" % problems_dirname)
  try:
    filename = None
    with open(report_filename, 'a+') as rh:
      for i in range(num_iterations):
        start_time = time.clock()
        print("Generate problem instance with %d variables, %d constraints, %d blocks, %d separator" % (num_variables, num_constraints, num_blocks, separator_size))
        problem = QBBILPModelGenerator().gen(
          num_variables=num_variables, num_constraints=num_constraints,
          num_blocks=num_blocks, separator_size=separator_size,
          fixed_separator_size=True, fix_block_size=True
        )
        name = name_format % (num_constraints, num_variables, num_blocks, separator_size, id(problem))
        problem.set_name(name)
        end_time = time.clock()
        print("%s was generated in %0.3f sec(s)" % (name, end_time - start_time))
        filename = os.path.join(problems_dirname, '%s.mps' % name)
        print("Write problem to %s" % filename)
        Encoder(filename).encode(problem)
        rh.write("Done =)")
  except KeyboardInterrupt:
    print("Stopping...", file=sys.stderr)
  #finally:
  #  os.unlink(filename)

def main(argv):
  '''model = QBBILPModelGenerator().gen(num_constraints=6, num_variables=9,
                        num_blocks=3, fix_block_size=True,
                        separator_size=1)
  model.pprint()'''
  parser = argparse.ArgumentParser()
  parser.add_argument('n', metavar='N', type=int, help='Number of constraints.')
  parser.add_argument('m', metavar='M', type=int, help='Number of variables.')
  parser.add_argument('k', metavar='K', type=int, default=2, help='Number of blocks.')
  parser.add_argument('-i', type=int, default=1, help='Specifies number of '
                      'iterations or how many test instances do we need to generate.')
  parser.add_argument('s', type=int, default=11, help='Separator size.')
  parser.add_argument('--problems-dir', type=str,
                      default=os.path.join(os.path.dirname(__file__), 'problems'),
                      help='Specifies directory where problem instances will be stored.')
  parser.add_argument('--report-file', type=str, default='report.csv',
                      help='Report file')
  ns = parser.parse_args()
  do_benchmarking(num_variables=ns.m, num_constraints=ns.n, num_blocks=ns.k,separator_size=ns.s,
                  num_iterations=ns.i, problems_dirname=ns.problems_dir,
                  report_filename=ns.report_file)

if __name__ == '__main__':
  exit(main(sys.argv))
