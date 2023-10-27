from preprocessing.data_processing.data_processing import DataProcessing

from typing import Dict
import pandas as pd


class StandardProcessing(DataProcessing):
    """
    Class to process dataset with method standard-processing
    """
    def __init__(self):
        """
        Init standard-processing
        """
        super().__init__()

        self.name = "Standard-Processing"

    @classmethod
    def process_data(cls, data_dict: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """
        Run standard-processing for given dataset (no processing)
        :param data_dict: Dictionary with dataset
        :return: Dictionary with processed data
        """
        return data_dict
