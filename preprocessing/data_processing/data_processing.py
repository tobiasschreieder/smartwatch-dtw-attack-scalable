import pandas as pd
from typing import Dict


class DataProcessing:
    """
    Base class for different types of data-processing
    """
    def __init__(self):
        """
        Init data-processing
        """
        self.name = "Data-Processing"
        pass

    @classmethod
    def process_data(cls, data_dict: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """
        Process dictionary with dataset
        :param data_dict: Dictionary with dataset per subject
        :return: Processed dictionary
        """
        return data_dict
