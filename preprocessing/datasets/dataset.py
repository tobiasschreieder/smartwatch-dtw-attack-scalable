from typing import Dict, List
import pandas as pd


class Dataset:
    def __init__(self):
        """
        Generate, preprocess and load dataset
        """
        self.name = "DATASET"
        self.data = Dict[int, pd.DataFrame]
        pass

    def load_dataset(self, resample_factor: int = None) -> Dict[int, pd.DataFrame]:
        """
        Load dataset with specified resample-factor (down-sampling)
        :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
        :return: Dictionary with preprocessed data
        """
        return {0: pd.DataFrame()}

    def get_dataset_name(self) -> str:
        """
        Get name of dataset
        :return: String with name
        """
        return self.name

    def get_subject_list(self) -> List[int]:
        """
        Get List with all available subjects
        :return: List with subject-ids
        """
        return []

    def get_sensor_combinations(self) -> List[List[str]]:
        """
        Get sensor-combinations
        :return: sensor-combinations
        """
        return [[]]

    def get_classes(self) -> List[str]:
        """
        Get classes
        :return: List with all classes
        """
        return []
