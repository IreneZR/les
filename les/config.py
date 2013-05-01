# -*- coding: utf-8;-*-
#
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

import logging

from les.config_autogen import *

log_handler = logging.StreamHandler()
log_handler.setFormatter(logging.Formatter("[%(levelname).1s] %(message)s"))
logging.getLogger().addHandler(log_handler)

# TODO: remove this
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
