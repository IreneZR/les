# Generated by the protocol buffer compiler.  DO NOT EDIT!

from google.protobuf import descriptor
from google.protobuf import message
from google.protobuf import reflection
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)


import les.mp_model.mp_model_pb2

DESCRIPTOR = descriptor.FileDescriptor(
  name='les/backend_solvers/backend_solvers.proto',
  package='les.backend_solvers',
  serialized_pb='\n)les/backend_solvers/backend_solvers.proto\x12\x13les.backend_solvers\x1a\x1bles/mp_model/mp_model.proto*\x92\x01\n\rBackendSolver\x12\x07\n\x03\x43LP\x10\x00\x12\x10\n\x0c\x44UMMY_SOLVER\x10\x01\x12\x1e\n\x1a\x46RAKTIONAL_KNAPSACK_SOLVER\x10\x02\x12\x08\n\x04GLPK\x10\x03\x12\x0c\n\x08LP_SOLVE\x10\x04\x12\x08\n\x04SCIP\x10\x05\x12\x0c\n\x08SYMPHONY\x10\x06\x12\x16\n\x12KNAPSACK_01_SOLVER\x10\x07')

_BACKENDSOLVER = descriptor.EnumDescriptor(
  name='BackendSolver',
  full_name='les.backend_solvers.BackendSolver',
  filename=None,
  file=DESCRIPTOR,
  values=[
    descriptor.EnumValueDescriptor(
      name='CLP', index=0, number=0,
      options=None,
      type=None),
    descriptor.EnumValueDescriptor(
      name='DUMMY_SOLVER', index=1, number=1,
      options=None,
      type=None),
    descriptor.EnumValueDescriptor(
      name='FRAKTIONAL_KNAPSACK_SOLVER', index=2, number=2,
      options=None,
      type=None),
    descriptor.EnumValueDescriptor(
      name='GLPK', index=3, number=3,
      options=None,
      type=None),
    descriptor.EnumValueDescriptor(
      name='LP_SOLVE', index=4, number=4,
      options=None,
      type=None),
    descriptor.EnumValueDescriptor(
      name='SCIP', index=5, number=5,
      options=None,
      type=None),
    descriptor.EnumValueDescriptor(
      name='SYMPHONY', index=6, number=6,
      options=None,
      type=None),
    descriptor.EnumValueDescriptor(
      name='KNAPSACK_01_SOLVER', index=7, number=7,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=96,
  serialized_end=242,
)


CLP = 0
DUMMY_SOLVER = 1
FRAKTIONAL_KNAPSACK_SOLVER = 2
GLPK = 3
LP_SOLVE = 4
SCIP = 5
SYMPHONY = 6
KNAPSACK_01_SOLVER = 7



# @@protoc_insertion_point(module_scope)
