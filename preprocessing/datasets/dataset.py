from preprocessing.data_processing.data_processing import DataProcessing

from typing import Dict, List
import pandas as pd
import itertools


class Dataset:
    def __init__(self):
        """
        Generate, preprocess and load dataset
        """
        self.name = "DATASET"
        self.data = Dict[int, pd.DataFrame]
        pass

    def load_dataset(self, data_processing: DataProcessing, resample_factor: int = None) -> Dict[int, pd.DataFrame]:
        """
        Load dataset with specified resample-factor (down-sampling)
        :param data_processing: Specify type of data-processing
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

    def get_classes(self) -> List[str]:
        """
        Get classes
        :return: List with all classes
        """
        return []


def get_sensor_combinations(dataset: Dataset, resample_factor: int, data_processing: DataProcessing) -> List[List[str]]:
    """
    Get all combinations of sensors from processed dataset
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :return: List with sensor-combinations
    """
    data_dict = dataset.load_dataset(resample_factor=resample_factor, data_processing=data_processing)
    sensors = list(data_dict[list(data_dict.keys())[0]].keys().drop("label"))
    if "acc_x" in sensors:
        sensors.remove("acc_x")
        sensors.remove("acc_y")
        sensors.remove("acc_z")
        sensors.append("acc")

    sensor_combinations = list()
    for i in range(1, len(sensors) + 1):
        combinations = (list(itertools.combinations(sensors, i)))
        for comb in combinations:
            sensor_combinations.append(list(comb))

    return sensor_combinations


