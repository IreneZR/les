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

import sys

from les.utils import logging
from les.drivers import drivers_pb2
from les.drivers.local_elimination_driver import LocalEliminationDriver
from les.drivers.oracle_driver.oracle_driver import OracleDriver
from les.drivers.comparison_driver.comparison_driver import ComparisonDriver
from les.drivers.simple_driver.simple_driver import SimpleDriver
from les.drivers.greedy_driver.greedy_driver import GreedyDriver
from les.drivers.compl_diff_driver.compl_diff_driver import ComplDiffDriver
from les.drivers.compl_diff2_driver.compl_diff2_driver import ComplDiff2Driver


LOCAL_ELIMINATION_DRIVER = drivers_pb2.LOCAL_ELIMINATION_DRIVER
ORACLE_DRIVER            = drivers_pb2.ORACLE_DRIVER
COMPARISON_DRIVER        = drivers_pb2.COMPARISON_DRIVER
SIMPLE_DRIVER            = drivers_pb2.SIMPLE_DRIVER
GREEDY_DRIVER            = drivers_pb2.GREEDY_DRIVER
COMPL_DIFF_DRIVER        = drivers_pb2.COMPL_DIFF_DRIVER
COMPL_DIFF2_DRIVER        = drivers_pb2.COMPL_DIFF2_DRIVER

_DRIVERS_TABLE = {
  LOCAL_ELIMINATION_DRIVER: LocalEliminationDriver,
  ORACLE_DRIVER: OracleDriver,
  COMPARISON_DRIVER: ComparisonDriver,
  SIMPLE_DRIVER: SimpleDriver,
  GREEDY_DRIVER: GreedyDriver,
  COMPL_DIFF_DRIVER: ComplDiffDriver,
  COMPL_DIFF2_DRIVER: ComplDiff2Driver
}

def get_instance_of(driver_id, *args, **kwargs):
  if not isinstance(driver_id, int):
    raise TypeError()
  if not driver_id in _DRIVERS_TABLE:
    return None
  driver_class = _DRIVERS_TABLE[driver_id]
  return driver_class(*args, **kwargs)
