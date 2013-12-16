# Copyright (c) 2012-2013 Oleksandr Sviridenko
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy
from scipy import sparse
import string
import sys

from les.utils import logging
from les.utils import uuid
from les import object_base


class Error(Exception):
  pass


class MPModel(object_base.ObjectBase):
  """The model of mathmatical programming problem consists of a set of variables
  (where a variable is represented by
  :class:`~les.model.mp_variable.MPVariable`) and a set of constraints (where a
  constraint is represented by :class:`~les.model.mp_constraint.MPConstraint`).

  By default maximization is set.

  :param name: Optional string that represents model name.
  :param variable_name_format: A string that represents a format that will be
    used to automitically generate variable names if they were not provided,
    e.g. `x0`, `x1`, etc.
  :param constraint_name_format: A string that represents a format that will be
    used to automitically generate constraint names if they were not provided,
    e.g. `c0`, `c1`, etc.
  """

  def __init__(self, name=None):
    self._name = name or uuid.ShortUUID().uuid()
    self._maximization = True
    self.objective_coefficients = []
    self.objective_name = None
    self.objective_value = None
    self.columns_lower_bounds = []
    self.columns_upper_bounds = []
    self.columns_names = []
    self.columns_values = []
    self.rows_names = []
    self.rows_coefficients = None
    self.rows_rhs = []
    self.rows_senses = []

  def __str__(self):
<<<<<<< HEAD
    return '%s[name=%s, num_constraints=%d, num_variables=%d]' \
        % (self.__class__.__name__, self.get_name(), self.get_num_constraints(),
           self.get_num_variables())

  def add_constraint(self, *args, **kwargs):
    '''Adds and returns a model constraint.

    Examples::

      c1 = model.add_constraint(2.0 * x + 3.0 * y <= 10.0, 'c1')
      c2 = model.add_constraint(3.0 * x, 'L', 8.0)

    :param expression: Left-hand side for new constraint.
    :param sense: Sense for new constraint.
    :param rhs: Right-hand side for new constraint.
    :param name: Constraint name.
    :return: A :class:`les.model.mp_constraint.MPConstraint` instance.
    '''
    name = None
    cons = None
    if (len(args) in (1, 2) and
        not isinstance(args[0], mp_constraint.MPConstraint)):
      cons = mp_constraint.MPConstraint(args[0])
      name = args[1] if len(args) == 2 else kwargs.get('name', None)
    elif len(args) in (3, 4):
      sense = self.convert_sense_to_operator(args[1])
      cons = mp_constraint.MPConstraint(sense(args[0], args[2]))
      name = args[3] if len(args) == 4 else kwargs.get('name', None)
    else:
      raise Error()
    if not cons.get_name():
      if not name:
        name = self._cons_name_frmt.format(index=self.get_num_constraints() + 1)
      cons.set_name(name)
    # TODO(d2rk): fix the case when we remove the constraint.
    cons.set_index(self.get_num_constraints())
    self._cons[cons.get_name()] = cons
    return cons
    
  def make_simple_model(self, shared_variables=None, solution=None, 
  											NumVariables=0):
    new_model = MPModel()
    coefs, names = [], []
    for v in self.get_variables():
      if v.get_name() in shared_variables:
        continue
      coefs.append(self.get_objective().get_coefficient(v))
      names.append(v.get_name())
    new_model.set_objective_from_scratch(coefs, names)
    #new_model.set_objective(self.get_objective()._expr, True)
    constr = self.get_constraints()  
    m_vars = self.get_variables()  
    new_rows_rhs = []#constr[0].get_rhs()
    variables_names = solution.get_variables_names()    
    ii = 0
    for c in constr:
      constr_var_names = []
      for i in c.get_variables():
        constr_var_names.append(i.get_name())
      new_rows_rhs.append(c.get_rhs())
      for i in shared_variables:
        if i in variables_names and i in constr_var_names: # check
          new_rows_rhs[ii] -= c.get_coefficient(self.get_variable_by_name(i))
      ii = ii + 1
    '''ii = 0
    while ii < solution.get_num_variables():
    	print variables_names[ii], variables_values[ii]
    	ii = ii + 1'''
 
    new_coeffs = []
    for ii in range(len(constr)):
      tmp_coeffs = []
      constr_var_names = []
      for v in constr[ii].get_variables():
        constr_var_names.append(v.get_name())
      for v in m_vars:
        name = v.get_name()
        if not name in shared_variables: 
          if name in constr_var_names:
            tmp_coeffs.append(constr[ii].get_coefficient(v))
          else:
            tmp_coeffs.append(0)
      new_coeffs.append(tmp_coeffs)
    new_model.set_constraints_from_scratch(new_coeffs, [constr[i].get_sense() 
    for i in range(len(constr))], new_rows_rhs, [constr[i].get_name() 
    for i in range(len(constr))])
    #new_model.pprint()
    return new_model # self 

  def add_variables(self, variables):
    '''Adds variables from iterable `variables`.

    .. seealso:: :func:`add_variable`
    '''
    if not isinstance(variables, collections.Iterable):
      raise TypeError()
    for var in variables:
      self.add_variable(var)

  def add_variable(self, *args, **kwargs):
    '''Adds new variable.

    :returns: :class:`~les.model.mp_variable.MPVariable` instance.
    :raises: :exc:`TypeError`
    '''
    var = None
    if len(args) == 1 and isinstance(args[0], mp_variable.MPVariable):
      var = args[0]
    else:
      var = mp_variable.MPVariable(*args, **kwargs)
    if not var.get_name():
      var.set_name(self.gen_variable_name(index=self.get_num_variables() + 1))
    if var.get_name() in self._vars:
      return
    # TODO(d2rk): how about to remove the variable?
    i = self.get_num_variables()
    var.set_index(i)
    self._vars[var.get_name()] = var
    return var

  def add_binary_variable(self, *args, **kwargs):
    var = None
    if len(args) == 1 and isinstance(args[0], mp_variable.MPVariable):
      var = args[0]
    else:
      var = binary_mp_variable.BinaryMPVariable(*args, **kwargs)
    return self.add_variable(var)

  @classmethod
  def convert_sense_to_operator(cls, sense):
    if isinstance(sense, unicode):
      sense = str(sense)
    if type(sense) is types.BuiltinFunctionType:
      return sense
    elif not isinstance(sense, str):
      raise TypeError('Unknown sense type: ' + str(type(sense)))
    return SENSE_STR_TO_OPERATOR[sense.upper()]

  @classmethod
  def convert_senses_to_operators(cls, senses):
    if not isinstance(senses, collections.Iterable):
      raise TypeError()
    operators = []
    for sense in senses:
      operators.append(cls.convert_sense_to_operator(sense))
    return operators

  def gen_variable_name(self, **kwargs):
    '''Generates variable name.'''
    return self._var_name_frmt.format(**kwargs)

  def get_constraints(self):
    '''Returns a list of constraints.

    :returns: A list of :class:`~les.model.mp_constraint.MPConstraint`
      instances.
    '''
    return self._cons.values()
    
  #new
  def get_constraints_names(self):
    constraints_names = []
    for i in self._cons:
      constraints_names.append(i)
    return constraints_names

  def get_name(self):
    '''Returns the model name. Returns :attr:`default_model_name` if name wasn't
    defined.
=======
    return ("%s[num_rows=%d, num_columns=%d, maximization=%s]" %
            (self.__class__.__name__, self.get_num_rows(),
             self.get_num_columns(), self.maximization()))
>>>>>>> master

  def get_objective_value(self):
    return self.objective_value

  def set_solution(self, solution):
    self.objective_value = solution.get_objective_value()
    # TODO(d2rk): set only triggered variables.
    for i in range(self.get_num_columns()):
      name = self.columns_names[i]
      if name in solution.get_variables_names():
        self.columns_values[i] = solution.get_variable_value_by_name(name)

  def is_binary(self):
    """Returns whether the model is binary integer linear programming
    model.

    :returns: `True` or `False`.
    """
    for i in range(len(self.objective_coefficients)):
      if (self.columns_lower_bounds[i] != 0.0
          or self.columns_upper_bounds[i] != 1.0):
        return False
    return True

  def optimize(self, params=None):
    """Optimize the model by using given optimization parameters.

    :param params: A OptimizationParameters instance.
    """
    from les.frontend_solver import FrontendSolver
    solver = FrontendSolver()
    solver.load_model(self)
    solver.solve(params)

  def set_columns(self, columns_lower_bounds=[], columns_upper_bounds=[],
                  columns_names=[]):
    if (len(columns_lower_bounds) != len(columns_upper_bounds)
        or len(columns_upper_bounds) != len(columns_names)):
      raise ValueError("upper bounds == lower bounds == names: %d == %d == %d"
                       % (len(columns_lower_bounds), len(columns_upper_bounds),
                          len(columns_names)))
    self.columns_lower_bounds = columns_lower_bounds
    self.columns_upper_bounds = columns_upper_bounds
    self.columns_names = columns_names
    self.columns_values = [0.0] * len(columns_names)
    return self

  def set_objective(self, coefficients, name=None):
    if isinstance(coefficients, tuple):
      self.objective_coefficients = list(coefficients)
    if not isinstance(coefficients, list):
      raise TypeError("coefficients: %s" % coefficients)
    self.objective_coefficients = coefficients
    self.objective_name = name
    return self

  def set_objective_name(self, name):
    self.objective_name = name

  def set_rows(self, coefficients, senses, rhs, names=[]):
    # Normalize matrix of coefficients
    if isinstance(coefficients, list):
      coefficients = numpy.matrix(coefficients, dtype=float)
    if isinstance(coefficients, numpy.matrix):
      coefficients = sparse.csr_matrix(coefficients, dtype=float)
    else:
      coefficients = coefficients.tocsr()
    # Normalize RHS
    if isinstance(rhs, (tuple, list)):
      rhs = list(rhs)
    else:
      raise TypeError()
    # TODO(d2rk): Normalize senses.
    if (coefficients.shape[0] != len(senses) or
        len(senses) != len(rhs)):
      raise Exception()
    if len(names) and len(names) != len(rhs):
      raise Exception()
    elif not len(names):
      names = [None] * len(rhs)
    self.rows_coefficients = coefficients
    self.rows_senses = senses
    self.rows_rhs = rhs
    self.rows_names = names
    return self

  def maximization(self):
    return self._maximization

  def minimization(self):
    return self._minimization

  def check_columns_values(self, columns_values):
    """Checks whether the given columns values are correct.

    :param columns_values: A list of column values.

    :returns: ``True`` or ``False``.
    """
    raise NotImplementedError()
    # TODO: solution can be a tuple, list or sparse.base.spmatrix
    if type(columns_values) in (tuple, list):
      columns_values = numpy.array(columns_values)
    if columns_values.shape[0] != self.get_num_columns():
      logging.warning("Number of columns values doesn't match number of "
                      "columns: %d != %d", columns_values.shape[0],
                      self.get_num_columns())
      return False
    v = self.rows_coefficients.dot(columns_values)
    for i in xrange(len(v)):
      # TODO: add sense.
      if not v[i] <= self.rows_rhs[i]:
        return False
    return True

  def get_status(self):
    return None

  @classmethod
  def status_to_string(self):
    return None

  def is_feasible(self):
    return False

  def is_optimal(self):
    return False

<<<<<<< HEAD
    .. note:: The variables come in order they are stored in the constraint
              matrix.
    '''
    return self._vars.values()
  
  #new
  def get_variables_names(self):
    variables_names = []
    for i in self._vars: 
      variables_names.append(i)
    return variables_names
=======
  def is_optimal_or_feasible(self):
    return self.is_optimal() or self.is_feasible()
>>>>>>> master

  def get_num_columns(self):
    return self.rows_coefficients.shape[1]

  def get_columns_indices(self):
    return range(len(self.objective_coefficients))

  def get_num_rows(self):
    return self.rows_coefficients.shape[0]

  def get_objective_coefficients(self):
    return self.objective_coefficients

  def get_objective_coefficient(self, i):
    return self.objective_coefficients[i]

  def get_objective_name(self):
    return self.objective_name

  def get_rows_coefficients(self):
    """Returns matrix of constraints coefficients.

    :returns: A :class:`scipy.sparse.csr_matrix` instance. By default returns
      ``None``.
    """
    return self.rows_coefficients

  def get_rows_names(self):
    """Returns list of rows names."""
    return self.rows_names

  def get_row_name(self, i):
    return len(self.rows_names) and self.rows_names[i] or None

  def get_rows_rhs(self):
    """Returns vector that represents right-hand side."""
    return self.rows_rhs

  def get_rows_senses(self):
    return self.rows_senses

  def get_column_lower_bound(self, i):
    if not isinstance(i, int):
      raise TypeError()
    return self.columns_lower_bounds[i]

  def get_columns_lower_bounds(self):
    return self.columns_lower_bounds

  def get_columns_names(self):
    return self.columns_names

  def get_column_upper_bound(self, i):
    if not isinstance(i, int):
      raise TypeError()
    return self.columns_upper_bounds[i]

  def get_columns_upper_bounds(self):
    return self.columns_upper_bounds

  def get_column_name(self, i):
    if not isinstance(i, int):
      raise TypeError()
    return self.columns_names[i]

  def get_name(self):
    return self._name

  def set_name(self, name):
    self._name = name
    return self

  def slice(self, rows_scope, columns_scope):
    """Builds a submodel/slice based on this model."""
    columns_scope = list(columns_scope)
    rows_scope = list(rows_scope)
    return (self.__class__()
             .set_columns([self.columns_lower_bounds[_] for _ in columns_scope],
                          [self.columns_upper_bounds[_] for _ in columns_scope],
                          [self.columns_names[_] for _ in columns_scope])
             .set_objective([self.objective_coefficients[_] for _ in columns_scope],
                            self.objective_name)
             .set_rows(self.rows_coefficients[rows_scope, :][:, columns_scope],
                       [self.rows_senses[_] for _ in rows_scope],
                       [self.rows_rhs[_] for _ in rows_scope],
                       [self.rows_names[_] for _ in rows_scope]))

  def pprint(self, file=sys.stdout):
    if bool(len(self.objective_coefficients)):
      file.write("%s: %s" % (self.objective_name, self._maximization
                             and "maximize" or "minimize"))
      for i in range(len(self.objective_coefficients)):
        file.write(" %+.20g %s" % (self.objective_coefficients[i],
                                   self.columns_names[i]))
      file.write("\n")
    if bool(self.rows_coefficients.shape[0]):
      file.write("s.t.\n")
      for i in range(self.rows_coefficients.shape[0]):
        file.write("%s:" % self.rows_names[i])
        row = self.rows_coefficients.getrow(i)
        for ii, ij in zip(*row.nonzero()):
          file.write(" %+.20g %s" % (row[ii, ij], self.columns_names[ij]))
        file.write(" %s %+.20g" % (self.rows_senses[i], self.rows_rhs[i]))
        file.write("\n")
