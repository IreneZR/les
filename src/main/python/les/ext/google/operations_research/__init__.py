import os
import sys

_ROOT_DIR = os.path.join(__path__[0], '..', '..', '..', '..', '..', '..', '..')
OR_TOOLS_HOME_DIR = os.path.join(_ROOT_DIR, 'third_party', 'or-tools')
#OR_TOOLS_HOME_DIR = "/home/ira/Documents/third_party/or-tools"

if not (os.path.exists(OR_TOOLS_HOME_DIR) or os.path.islink(OR_TOOLS_HOME_DIR)): #cite
  raise IOError('%s does not exist' % os.path.realpath(OR_TOOLS_HOME_DIR))

if not os.environ.get('OR_TOOLS_HOME_DIR', None):
  os.environ['OR_TOOLS_HOME_DIR'] = os.path.realpath(OR_TOOLS_HOME_DIR)
