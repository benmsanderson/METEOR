"""
METEOR
"""
import logging
import os

import numpy as np
import pandas as pd

from .prpatt import imodel_filter_scl,rmodel

LOGGER = logging.getLogger(__name__)

class METEOR:
    def __init__(self, cfg):
        pscl=self.cfg["pscl"]
        pepc=self.cfg["pepc"]
        fco2=self.cfg["fco2"]
        
    def _run(self, cfg):
        self.epc=imodel_filter_scl(pscl,pepc,fco2)
        self.psim=rmodel(peof,epc)
