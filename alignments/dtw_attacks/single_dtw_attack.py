from preprocessing.data_processing.data_processing import DataProcessing
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


class SingleDtwAttack(DtwAttack):
    """
    Class for Single-DTW-ATTack
    """

    def __init__(self):
        """
        Try to init SingleDtwAttack
        """
        super().__init__()

        self.name = "Single-DTW-Attack"
        self.windows = [i for i in range(1, 37)]

    @classmethod
    def create_subject_data(cls, data_dict: Dict[int, pd.DataFrame], method: str, test_window_size: int,
                            subject_id: int, additional_windows: int = 1000, resample_factor: int = 1) \
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
        subject_data = {"train": dict(), "test": dict(), "method": method}
        labels = dict()

        for subject in data_dict:
            subject_data["train"].setdefault(subject, dict())

            if subject != subject_id:
                sensor_data = dict()
                for sensor in data_dict[subject]:
                    if sensor != "label":
                        sensor_data.setdefault(sensor, data_dict[subject][sensor])
                subject_data["train"].setdefault(subject, dict())
                subject_data["train"][subject] = sensor_data

            else:
                label_data = data_dict[subject]

                # Method "non-stress"
                if method == "non-stress":
                    label_data = label_data[label_data.label == 0]  # Data for subject with label == 0 -> non-stress

                # Method "stress"
                else:
                    label_data = label_data[label_data.label == 1]  # Data for subject with label == 1 -> stress

                data_start = label_data.iloc[:round(len(label_data) * 0.5), :]
                data_end = label_data.iloc[round(len(label_data) * 0.5):, :]

                # Create test and train data
                amount_remove_windows = (max(1, int(round(test_window_size * 0.5))) +
                                         int(additional_windows / resample_factor))
                amount_test_windows = max(1, int(test_window_size * 0.5))

                # Test-window-size == 1
                if test_window_size == 1:
                    test_1 = data_start.iloc[len(data_end):, :]
                    test_2 = data_end.iloc[:1, :]
                    remove_1 = data_start.iloc[(len(data_end) - amount_remove_windows):, :]
                    remove_2 = data_end.iloc[:amount_remove_windows, :]

                # Test-window-size == 2, 4, 6, ...
                elif test_window_size % 2 == 0:
                    test_1 = data_start.iloc[(-1 * amount_test_windows):]
                    test_2 = data_end.iloc[:amount_test_windows, :]
                    remove_1 = data_start.iloc[(-1 * amount_remove_windows):]
                    remove_2 = data_end.iloc[:amount_remove_windows, :]

                # Test-window-size = 3, 5, 7, ...
                else:
                    test_1 = data_start.iloc[(-1 * amount_test_windows):]
                    test_2 = data_end.iloc[:(amount_test_windows + 1), :]
                    remove_1 = data_start.iloc[(-1 * (amount_remove_windows - 1)):]
                    remove_2 = data_end.iloc[:amount_remove_windows, :]

                # Combine start and end DataFrame
                test = pd.concat([test_1, test_2])
                remove = pd.concat([remove_1, remove_2])
                train = data_dict[subject].drop(remove.index)

                # Create labels dictionary
                for sensor in label_data:
                    test_subject = test[sensor].to_frame()
                    train_subject = train[sensor].to_frame()

                    if sensor != "label":
                        subject_data["train"].setdefault(subject, dict())
                        subject_data["train"][subject].setdefault(sensor, train_subject)
                        subject_data["test"].setdefault(subject, dict())
                        subject_data["test"][subject].setdefault(sensor, test_subject)
                    else:
                        test_stress = (test["label"] == 1).sum()
                        test_non_stress = (test["label"] == 0).sum()
                        train_stress = (train["label"] == 1).sum()
                        train_non_stress = (train["label"] == 0).sum()
                        labels.setdefault(subject, {"test_stress": test_stress, "test_non_stress": test_non_stress,
                                                    "train_stress": train_stress, "train_non_stress": train_non_stress})

        return subject_data, labels

    @classmethod
    def test_max_window_size(cls, data_dict: Dict[int, pd.DataFrame], test_window_sizes: List[int],
                             additional_windows: int, resample_factor: int) -> bool:
        """
        Test all given test window-sizes if they are valid
        :param data_dict: Dictionary with preprocessed dataset
        :param test_window_sizes: List with all test_window-sizes to be tested
        :param additional_windows: Specify amount of additional windows to be removed around test-window
        :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
        :return: Boolean -> False if there is at least one wrong test window-size
        """
        additional_windows = max(1, int(round(additional_windows / resample_factor)))

        # Calculate max_window
        min_method_length = additional_windows * 2

        for subject, data in data_dict.items():
            non_stress_length = len(data[data.label == 0])
            stress_length = len(data[data.label == 1])
            min_method_subject_length = min(non_stress_length, stress_length)

            if min_method_length == (additional_windows * 2) or min_method_subject_length < min_method_length:
                min_method_length = min_method_subject_length

        # Test all test-window-sizes
        valid_windows = True
        for test_window_size in test_window_sizes:
            if test_window_size <= 0 or test_window_size > min_method_length:
                valid_windows = False
                print("Wrong test-window-size: " + str(test_window_size))

        return valid_windows

    @classmethod
    def calculate_alignment(cls, data_dict: Dict[int, pd.DataFrame], subject_id: int, method: str,
                            test_window_size: int, resample_factor: int = 1, additional_windows: int = 1000) \
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
        results_standard = dict()
        subject_data, labels = cls.create_subject_data(data_dict=data_dict, method=method,
                                                       test_window_size=test_window_size, subject_id=subject_id,
                                                       additional_windows=additional_windows,
                                                       resample_factor=resample_factor)

        for subject in subject_data["train"]:
            results_standard.setdefault(subject, dict())

            for sensor in subject_data["train"][subject]:
                test = subject_data["test"][subject_id][sensor]
                train = subject_data["train"][subject][sensor]
                test = test.values.flatten()
                train = train.values.flatten()

                distance_standard = dtw.distance_fast(train, test)
                results_standard[subject].setdefault(sensor, round(distance_standard, 4))

        return results_standard

    def run_calculations(self, dataset: Dataset, test_window_sizes: List[int], data_processing: DataProcessing,
                         resample_factor: int = 1, additional_windows: int = 1000, n_jobs: int = -1,
                         methods: List[str] = None, subject_ids: List[int] = None, runtime_simulation: bool = False):
        """
        Run DTW-calculations with all given parameters and save results as json
        :param dataset: Specify dataset, which should be used
        :param test_window_sizes: List with all test windows that should be used (int)
        :param data_processing: Specify type of data-processing
        :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
        :param additional_windows: Specify amount of additional windows to be removed around test-window
        :param n_jobs: Number of processes to use (parallelization)
        :param methods:  List with all method that should be used -> "non-stress" / "stress" (str)
        :param subject_ids: List with all subjects that should be used as test subjects (int) -> None = all subjects
        :param runtime_simulation: If True -> only simulate isolated attack and save runtime
        """
        def parallel_calculation(current_subject_id: int) -> Dict[int, Dict[int, Dict[str, float]]]:
            """
            Run parallel calculations
            :param current_subject_id: Specify subject_id
            :return: Dictionary with results
            """
            result = self.calculate_alignment(data_dict=data_dict, subject_id=current_subject_id, method=method,
                                              test_window_size=test_window_size, additional_windows=additional_windows,
                                              resample_factor=resample_factor)
            results_subject = {current_subject_id: result}

            return results_subject

        if subject_ids is None:
            subject_ids = dataset.subject_list
        if methods is None:
            methods = dataset.get_classes()

        data_dict = dataset.load_dataset(resample_factor=resample_factor, data_processing=data_processing)

        if self.test_max_window_size(data_dict=data_dict, test_window_sizes=test_window_sizes,
                                     resample_factor=resample_factor, additional_windows=additional_windows):
            print("Test-window-size test successful: All test-window-sizes are valid")

            for test_window_size in test_window_sizes:
                start = time.perf_counter()
                print("-Current window-size: " + str(test_window_size))
                for method in methods:
                    print("--Current method: " + str(method))

                    # Parallelization
                    with Parallel(n_jobs=n_jobs) as parallel:
                        results = parallel(
                            delayed(parallel_calculation)(current_subject_id=subject_id) for subject_id in subject_ids)

                    results_standard = dict()
                    for res in results:
                        results_standard.setdefault(list(res.keys())[0], list(res.values())[0])

                    end = time.perf_counter()

                    # Save runtime as txt file
                    dataset_path = os.path.join(cfg.out_dir, dataset.name + "_" + str(len(dataset.subject_list)))
                    resample_path = os.path.join(dataset_path, "resample-factor=" + str(resample_factor))
                    attack_path = os.path.join(resample_path, self.name)
                    processing_path = os.path.join(attack_path, data_processing.name)
                    alignments_path = os.path.join(processing_path, "alignments")
                    if runtime_simulation:
                        alignments_path = os.path.join(alignments_path, "isolated_simulation")

                    method_path = os.path.join(alignments_path, str(method))
                    runtime_path = os.path.join(alignments_path, "runtime")
                    os.makedirs(runtime_path, exist_ok=True)
                    os.makedirs(method_path, exist_ok=True)

                    try:
                        if not runtime_simulation:
                            # Save Runtime as TXT
                            runtime_file_name = "runtime_window_size=" + str(test_window_size) + ".txt"
                            runtime_save_path = os.path.join(runtime_path, runtime_file_name)

                            text_file = open(runtime_save_path, "w")
                            text = "Runtime: " + str(end - start)
                            text_file.write(text)
                            text_file.close()

                        # Save results as JSON
                        path_string_standard = "SW-DTW_results_standard_" + str(method) + "_" + str(
                            test_window_size) + ".json"

                        with open(os.path.join(method_path, path_string_standard), "w", encoding="utf-8") as outfile:
                            json.dump(results_standard, outfile)

                        print("SW-DTW results saved at: " + str(os.path.join(method_path, path_string_standard)))

                    except FileNotFoundError:
                        print("FileNotFoundError: results could not be saved!")

        else:
            print("Please specify valid window-sizes!")
