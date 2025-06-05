#!/usr/bin/env python3

import csv
import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import curve_fit
import argparse
from global_fit import fit_global_model
import matplotlib.pyplot as plt


class PhotonicsDataset:
    """Represents a single dataset with N_s, N_i, N_c columns."""

    def __init__(self, name: str = ""):
        self.name = name
        self.dark_counts = {}  # N_s, N_i, N_c dark counts
        self.piezo_pos = np.array...
        self.N_s = np.array..
        self.N_i = np.array..
        self.N_c = np.array...
        self.metadata = {}     # metadata key -> value

    def __repr__(self):
        return f"PhotonicsDataset(name='{self.name}', {len(self.piezo_data)} piezo positions)"
