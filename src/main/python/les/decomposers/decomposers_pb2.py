# Generated by the protocol buffer compiler.  DO NOT EDIT!

from google.protobuf import descriptor
from google.protobuf import message
from google.protobuf import reflection
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)


import les.mp_model.mp_model_pb2

DESCRIPTOR = descriptor.FileDescriptor(
  name='les/decomposers/decomposers.proto',
  package='les.decomposers',
  serialized_pb='\n!les/decomposers/decomposers.proto\x12\x0fles.decomposers\x1a\x1bles/mp_model/mp_model.proto\"j\n\x14\x44\x65\x63omposerParameters\x12R\n\ndecomposer\x18\x01 \x02(\x0e\x32\x1b.les.decomposers.Decomposer:!QUASIBLOCK_FINKELSTEIN_DECOMPOSER*N\n\nDecomposer\x12%\n!QUASIBLOCK_FINKELSTEIN_DECOMPOSER\x10\x00\x12\x19\n\x15MAX_CLIQUE_DECOMPOSER\x10\x01:j\n\x15\x64\x65\x63omposer_parameters\x12$.les.mp_model.OptimizationParameters\x18\x65 \x01(\x0b\x32%.les.decomposers.DecomposerParameters')

_DECOMPOSER = descriptor.EnumDescriptor(
  name='Decomposer',
  full_name='les.decomposers.Decomposer',
  filename=None,
  file=DESCRIPTOR,
  values=[
    descriptor.EnumValueDescriptor(
      name='QUASIBLOCK_FINKELSTEIN_DECOMPOSER', index=0, number=0,
      options=None,
      type=None),
    descriptor.EnumValueDescriptor(
      name='MAX_CLIQUE_DECOMPOSER', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=191,
  serialized_end=269,
)


QUASIBLOCK_FINKELSTEIN_DECOMPOSER = 0
MAX_CLIQUE_DECOMPOSER = 1

DECOMPOSER_PARAMETERS_FIELD_NUMBER = 101
decomposer_parameters = descriptor.FieldDescriptor(
  name='decomposer_parameters', full_name='les.decomposers.decomposer_parameters', index=0,
  number=101, type=11, cpp_type=10, label=1,
  has_default_value=False, default_value=None,
  message_type=None, enum_type=None, containing_type=None,
  is_extension=True, extension_scope=None,
  options=None)


_DECOMPOSERPARAMETERS = descriptor.Descriptor(
  name='DecomposerParameters',
  full_name='les.decomposers.DecomposerParameters',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    descriptor.FieldDescriptor(
      name='decomposer', full_name='les.decomposers.DecomposerParameters.decomposer', index=0,
      number=1, type=14, cpp_type=8, label=2,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=83,
  serialized_end=189,
)

_DECOMPOSERPARAMETERS.fields_by_name['decomposer'].enum_type = _DECOMPOSER
DESCRIPTOR.message_types_by_name['DecomposerParameters'] = _DECOMPOSERPARAMETERS

class DecomposerParameters(message.Message):
  __metaclass__ = reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _DECOMPOSERPARAMETERS
  
  # @@protoc_insertion_point(class_scope:les.decomposers.DecomposerParameters)

decomposer_parameters.message_type = _DECOMPOSERPARAMETERS
les.mp_model.mp_model_pb2.OptimizationParameters.RegisterExtension(decomposer_parameters)
# @@protoc_insertion_point(module_scope)
