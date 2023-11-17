from preprocessing.data_processing.data_processing import DataProcessing
from preprocessing.datasets.dataset import Dataset

from typing import Tuple, Dict, List
import pandas as pd


class DtwAttack:
    """
    Base Class for DTW-attacks
    """
    def __init__(self):
        """
        Init DTW-Attack
        """
        self.name = "DTW-Attack"
        self.windows = []
        pass

    def create_subject_data(self, data_dict: Dict[int, pd.DataFrame], method: str, test_window_size: int,
                            subject_id: int, additional_windows: int, resample_factor: int) \
            -> Tuple[Dict[str, Dict[int, Dict[str, pd.DataFrame]]], Dict[int, Dict[str, int]]]:
        """
        Create dictionary with all subjects and their sensor data as Dataframe split into train and test data
        Create dictionary with label information for test subject
        :param data_dict: Dictionary with preprocessed dataset
        :param method: String to specify which method should be used (non-stress / stress)
        :param test_window_size: Specify amount of windows (datapoints) in test set (int)
        :param subject_id: Specify which subject should be used as test subject
        :param additional_windows: Specify amount of additional windows to be removed around test-window
        :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
        :return: Tuple with create subject_data and labels (containing label information)
        """
        subject_data = {}
        labels = {}
        return subject_data, labels

    def test_max_window_size(self, data_dict: Dict[int, pd.DataFrame], test_window_sizes: List[int],
                             additional_windows: int, resample_factor: int) -> bool:
        """
        Test all given test window-sizes if they are valid
        :param data_dict: Dictionary with preprocessed dataset
        :param test_window_sizes: List with all test_window-sizes to be tested
        :param additional_windows: Specify amount of additional windows to be removed around test-window
        :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
        :return: Boolean -> False if there is at least one wrong test window-size
        """
        return False

    def calculate_alignment(self, data_dict: Dict[int, pd.DataFrame], subject_id: int, method: str,
                            test_window_size: int, resample_factor: int, additional_windows: int) \
            -> Dict[int, Dict[str, float]]:
        """
        Calculate DTW-Alignments for sensor data using Dynamic Time Warping
        :param data_dict: Dictionary with preprocessed dataset
        :param subject_id: Specify which subject should be used as test subject
        :param method: String to specify which method should be used (non-stress / stress)
        :param test_window_size: Specify amount of windows (datapoints) in test set (int)
        :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
        :param additional_windows: Specify amount of additional windows to be removed around test-window
        :return: Tuple with Dictionaries of standard and normalized results
        """
        results_standard = {}
        return results_standard

    def run_calculations(self, dataset: Dataset, data_processing: DataProcessing, test_window_sizes: List[int],
                         resample_factor: int, additional_windows: int, n_jobs: int, methods: List[str] = None,
                         subject_ids: List[int] = None, runtime_simulation: bool = False):
        """
        Run DTW-calculations with all given parameters and save results as json
        :param dataset: Specify dataset, which should be used
        :param data_processing: Specify type of data-processing
        :param test_window_sizes: List with all test windows that should be used (int)
        :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
        :param additional_windows: Specify amount of additional windows to be removed around test-window
        :param n_jobs: Number of processes to use (parallelization)
        :param methods:  List with all method that should be used -> "non-stress" / "stress" (str)
        :param subject_ids: List with all subjects that should be used as test subjects (int) -> None = all subjects
        :param runtime_simulation: If True -> only simulate isolated attack and save runtime
        """
        pass
