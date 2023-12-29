from preprocessing.data_processing.data_processing import DataProcessing
from preprocessing.datasets.dataset import Dataset
from alignments.dtw_attacks.dtw_attack import DtwAttack
from config import Config

from joblib import Parallel, delayed
from typing import Dict, Tuple, List, Any
from dtaidistance import dtw
import pandas as pd
import math
import statistics
import os
import json
import time


cfg = Config.get()


class MultiSlicingDtwAttack(DtwAttack):
    """
    Class for Multi-Slicing-DTW-Attack
    """
    def __init__(self):
        """
        Try to init MultiSlicingDtwAttack
        """
        super().__init__()

        self.name = "Multi-Slicing-DTW-Attack"
        self.windows = [i for i in range(1, 13)]

    @classmethod
    def create_subject_data(cls, data_dict: Dict[int, pd.DataFrame], method: str, test_window_size: int,
                            subject_id: int, multi: int = 3, additional_windows: int = 1000, resample_factor: int = 1) \
            -> Tuple[Dict[str, Any], Dict[int, Dict[str, int]]]:
        """
        Create dictionary with all subjects and their sensor data as Dataframe split into train and test data
        Create dictionary with label information for test subject
        :param data_dict: Dictionary with preprocessed dataset
        :param method: String to specify which method should be used (non-stress / stress)
        :param test_window_size: Specify amount of windows (datapoints) in test set (int)
        :param subject_id: Specify which subject should be used as test subject
        :param multi: Specify number of combined single attacks
        :param additional_windows: Specify amount of additional windows to be removed around test-window
        :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
        :return: Tuple with create subject_data and labels (containing label information)
        """
        def split_train_data(train_data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
            """
            Split train-data into maximum number of slicing windows with same size as test-window
            If test-window-size = 10: train-windows 0:9, 5-14, 10-19, ... max
            :param train_data: DataFrame with train-data
            :return: Dictionary with sliced train DataFrames
            """
            amount_train_windows = math.floor(len(train_data) / test_window_size)
            train_data = train_data.iloc[:(amount_train_windows * test_window_size)].reset_index(drop=True)
            train_dict = dict()

            t_id = 0
            for i in range(0, len(train_data) - test_window_size + 1, max(1, int(round(test_window_size / 2)))):
                train_slice = train_data.iloc[i: i + test_window_size]
                train_dict.setdefault(t_id, train_slice)
                t_id += 1

            return train_dict

        if multi <= 2:
            multi = 2
            print("Invalid number of combined attacks (multi)! Multi needs to be >= 2.")
            print("Set multi = 2!")

        subject_data = {"train": dict(), "test": dict(), "method": method}
        labels = dict()

        for subject in data_dict:
            subject_data["train"].setdefault(subject, dict())

            if subject != subject_id:
                train_dict = split_train_data(train_data=data_dict[subject])
                subject_data["train"].setdefault(subject, dict())
                for train_slice in train_dict:
                    subject_data["train"][subject].setdefault(train_slice, dict())
                    sensor_data = dict()
                    for sensor in train_dict[train_slice]:
                        if sensor != "label":
                            sensor_data.setdefault(sensor, train_dict[train_slice][sensor].to_frame())
                    subject_data["train"][subject][train_slice] = sensor_data

            else:
                label_data = data_dict[subject]

                # Method "non-stress"
                if method == "non-stress":
                    label_data = label_data[label_data.label == 0]  # Data for subject with label == 0 -> non-stress

                # Method "stress"
                else:
                    label_data = label_data[label_data.label == 1]  # Data for subject with label == 1 -> stress

                # Create test and train data
                test_windows = test_window_size * multi
                half_data_length = round(len(label_data) / 2)
                resampled_additional_windows = max(1, int(round(additional_windows / resample_factor)))

                start = int(half_data_length - (test_windows / 2))
                end = int(half_data_length + (test_windows / 2))
                test_combined = label_data.iloc[start: end]
                remove = label_data.iloc[start - resampled_additional_windows: end + resampled_additional_windows]

                test_combined_2 = test_combined.reset_index(drop=True)
                test_dict = dict()
                for m in range(0, multi):
                    test = test_combined_2.iloc[(m * test_window_size): (m * test_window_size + test_window_size)]
                    test_dict.setdefault(m, test)

                train = data_dict[subject].drop(remove.index)
                train_dict = split_train_data(train_data=train)

                # Create labels dictionary
                for sensor in label_data:
                    if sensor != "label":
                        subject_data["train"].setdefault(subject, dict())
                        subject_data["test"].setdefault(subject, dict())
                        for test_id, test in test_dict.items():
                            subject_data["test"][subject].setdefault(test_id, dict())
                            test_subject = test[sensor].to_frame()
                            subject_data["test"][subject][test_id].setdefault(sensor, test_subject)
                        for train_id, test in train_dict.items():
                            subject_data["train"][subject].setdefault(train_id, dict())
                            train_subject = train_dict[train_id][sensor].to_frame()
                            subject_data["train"][subject][train_id].setdefault(sensor, train_subject)

                    else:
                        train_stress = (train["label"] == 1).sum()
                        train_non_stress = (train["label"] == 0).sum()
                        test_stress = (test_combined["label"] == 1).sum()
                        test_non_stress = (test_combined["label"] == 0).sum()
                        labels.setdefault(subject, {"test_stress": test_stress, "test_non_stress": test_non_stress,
                                                    "train_stress": train_stress, "train_non_stress": train_non_stress})

        return subject_data, labels

    @classmethod
    def test_max_window_size(cls, data_dict: Dict[int, pd.DataFrame], test_window_sizes: List[int],
                             additional_windows: int, resample_factor: int, multi: int = 3) -> bool:
        """
        Test all given test window-sizes if they are valid
        :param data_dict: Dictionary with preprocessed dataset
        :param test_window_sizes: List with all test_window-sizes to be tested
        :param additional_windows: Specify amount of additional windows to be removed around test-window
        :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
        :param multi: Specify number of combined single attacks
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
            if test_window_size * multi <= 0 or test_window_size * multi > min_method_length:
                valid_windows = False
                print("Wrong test-window-size: " + str(test_window_size))

        return valid_windows

    @classmethod
    def calculate_alignment(cls, data_dict: Dict[int, pd.DataFrame], subject_id: int, method: str,
                            test_window_size: int, multi: int = 3, resample_factor: int = 1,
                            additional_windows: int = 1000) -> Dict[int, Dict[str, float]]:
        """
        Calculate DTW-Alignments for sensor data using Dynamic Time Warping
        :param data_dict: Dictionary with preprocessed dataset
        :param subject_id: Specify which subject should be used as test subject
        :param method: String to specify which method should be used (non-stress / stress)
        :param test_window_size: Specify amount of windows (datapoints) in test set (int)
        :param multi: Specify number of combined single attacks
        :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
        :param additional_windows: Specify amount of additional windows to be removed around test-window
        :return: Tuple with Dictionaries of standard and normalized results
        """
        results_standard = dict()
        subject_data, labels = cls.create_subject_data(data_dict=data_dict, method=method,
                                                       test_window_size=test_window_size, multi=multi,
                                                       subject_id=subject_id, additional_windows=additional_windows,
                                                       resample_factor=resample_factor)

        for subject in subject_data["train"]:
            results_standard.setdefault(subject, dict())

            for sensor in subject_data["test"][subject_id][0]:
                for test_multi in subject_data["test"][subject_id]:
                    test = subject_data["test"][subject_id][test_multi][sensor]
                    test = test.values.flatten()

                    results_standard[subject].setdefault(test_multi, dict())
                    for train_slice in subject_data["train"][subject]:
                        train = subject_data["train"][subject][train_slice][sensor]
                        train = train.values.flatten()

                        distance_standard = dtw.distance_fast(train, test)
                        results_standard[subject][test_multi].setdefault(train_slice, dict())
                        results_standard[subject][test_multi][train_slice].setdefault(sensor,
                                                                                      round(distance_standard, 4))

        # Calculate average distances
        for subject in results_standard:
            for sensor in results_standard[subject][0][0]:
                sensor_results = list()
                for test_multi in results_standard[subject]:
                    results_standard[subject][test_multi].setdefault("min", dict())
                    for train_slice in results_standard[subject][test_multi]:
                        if train_slice != "min":
                            sensor_results.append(results_standard[subject][test_multi][train_slice][sensor])
                        results_standard[subject][test_multi]["min"].setdefault(sensor, round(min(sensor_results), 4))

        for subject in results_standard:
            for sensor in results_standard[subject][0][0]:
                sensor_results = list()
                results_standard[subject].setdefault("mean", dict())
                for test_multi in results_standard[subject]:
                    if test_multi != "mean":
                        sensor_results.append(results_standard[subject][test_multi]["min"][sensor])
                results_standard[subject]["mean"].setdefault(sensor, round(min(sensor_results), 4))

        return results_standard

    def run_calculations(self, dataset: Dataset, test_window_sizes: List[int], data_processing: DataProcessing,
                         multi: int = 3, resample_factor: int = 1, additional_windows: int = 1000, n_jobs: int = -1,
                         methods: List[str] = None, subject_ids: List[int] = None, runtime_simulation: bool = False):
        """
        Run DTW-calculations with all given parameters and save results as json
        :param dataset: Specify dataset, which should be used
        :param test_window_sizes: List with all test windows that should be used (int)
        :param data_processing: Specify type of data-processing
        :param multi: Specify number of combined single attacks
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
                                              test_window_size=test_window_size, multi=multi,
                                              additional_windows=additional_windows, resample_factor=resample_factor)
            results_subject = {current_subject_id: result}

            return results_subject

        if subject_ids is None:
            subject_ids = dataset.subject_list
        if methods is None:
            methods = dataset.get_classes()

        data_dict = dataset.load_dataset(resample_factor=resample_factor, data_processing=data_processing)

        if self.test_max_window_size(data_dict=data_dict, test_window_sizes=test_window_sizes, multi=multi,
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
