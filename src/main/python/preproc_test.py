from les.mp_model import MPModel
from les.mp_model.mp_model_builder.formats import mps
from les.mp_model.mp_model_builder import MPModelBuilder

decoder = mps.Decoder()
filename = "/home/ira/les/data/demos/demo6.mps"#"/home/ira/Downloads/demo4.mps"#"/home/ira/Downloads/Tasks/demo_3f.mps"
with open(filename, "r") as stream:
  decoder.decode(stream)
model = MPModelBuilder.build_from(decoder)
model.pprint()
print
model.optimize()
print model.get_objective_value()
nm, res = model.preproc()
nm.pprint()
nm.optimize()
print nm.get_objective_value()
for i in res:
  print i
for i in range(len(nm.get_variables_names())):
  print nm.get_variables_names()[i], nm.columns_values[i]
