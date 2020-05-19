# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : conf.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import os
import logging
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

_ELEMENT_DB_PATH_ = os.path.join(PROJECT_ROOT, r"data/elements.db")

# _DEVICE_ = "cuda"

# LOGconfig
# ALL detail
_integration_output = 0  # No integration matrix elements and values
_hamilton_output = 0  # No integration values
_iter_values_output = 1  # iter detail
_iter_info_output = 1  # iter info

# For HF method
_erfE = 1e-16
_erfP = 1e-9
_MAXiternum = 500
