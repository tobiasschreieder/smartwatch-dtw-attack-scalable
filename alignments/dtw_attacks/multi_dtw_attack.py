from preprocessing.datasets.dataset import Dataset
from alignments.dtw_attacks.dtw_attack import DtwAttack
from config import Config

from joblib import Parallel, delayed
from typing import Dict, Tuple, List
import pandas as pd
from dtaidistance import dtw
import os
import json
import time


cfg = Config.get()


class MultiDtwAttack(DtwAttack):
    """
    Try to init SingleDtwAttack
    """
    def __init__(self):
        super().__init__()

        self.name = "Single-DTW-Attack"
        self.windows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 100, 500, 1000, 10000]

    def get_windows(self) -> List[int]:
        """
        Get test-window-sizes
        :return: List with all test-window-sizes
        """
        return self.windows
