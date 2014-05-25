from les.mp_model.mp_model import MPModel

class AddOnsMPM:
  
  def __init__(self):
    self.addons = 1
  
  def to_knapsack(self, model):
    Sobj = sum(model.get_objective_coefficients())
    Scon = model.get_rows_coefficients().sum(0).tolist()[0]
    Srhs = sum(model.get_rows_rhs())
    vrs = model.get_variables_names()
    return vrs, Scon, Srhs, Sobj/sum(Scon)
