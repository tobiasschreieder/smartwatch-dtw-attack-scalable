from preprocessing.datasets.dataset import Dataset
from alignments.dtw_attacks.dtw_attack import DtwAttack
from config import Config

from joblib import Parallel, delayed
from typing import Dict, Tuple, List, Any
import pandas as pd
from dtaidistance import dtw
import statistics
import os
import json
import time


cfg = Config.get()


class MultiDtwAttack(DtwAttack):
    """
    Try to init MultiDtwAttack
    """
    def __init__(self):
        super().__init__()

        self.name = "Multi-DTW-Attack"
        self.windows = [2, 5, 7]

    def get_windows(self) -> List[int]:
        """
        Get test-window-sizes
        :return: List with all test-window-sizes
        """
        return self.windows

    @classmethod
    def create_subject_data(cls, dataset: Dataset, method: str, test_window_size: int, subject_id: int, multi: int = 3,
                            additional_windows: int = 1000, resample_factor: int = 1) \
            -> Tuple[Dict[str, Any], Dict[int, Dict[str, int]]]:
        """
        Create dictionary with all subjects and their sensor data as Dataframe split into train and test data
        Create dictionary with label information for test subject
        :param dataset: Specify dataset
        :param method: String to specify which method should be used (non-stress / stress)
        :param test_window_size: Specify amount of windows (datapoints) in test set (int)
        :param subject_id: Specify which subject should be used as test subject
        :param multi: Specify number of combined single attacks
        :param additional_windows: Specify amount of additional windows to be removed around test-window
        :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
        :return: Tuple with create subject_data and labels (containing label information)
        """
        if multi <= 2:
            multi = 2
            print("Invalid number of combined attacks (multi)! Multi needs to be >= 2.")
            print("Set multi = 2!")

        data_dict = dataset.load_dataset(resample_factor=resample_factor)
        subject_data = {"train": dict(), "test": dict(), "method": method}
        labels = dict()

        for subject in data_dict:
            subject_data["train"].setdefault(subject, dict())

            if subject != subject_id:
                sensor_data = {"bvp": data_dict[subject][["bvp"]],
                               "eda": data_dict[subject][["eda"]],
                               "acc_x": data_dict[subject][["acc_x"]],
                               "acc_y": data_dict[subject][["acc_y"]],
                               "acc_z": data_dict[subject][["acc_z"]],
                               "temp": data_dict[subject][["temp"]]}
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

                # Create labels dictionary
                for sensor in label_data:
                    train_subject = train[sensor].to_frame()
                    if sensor != "label":
                        subject_data["train"].setdefault(subject, dict())
                        subject_data["train"][subject].setdefault(sensor, train_subject)
                        subject_data["test"].setdefault(subject, dict())
                        for test_id, test in test_dict.items():
                            subject_data["test"][subject].setdefault(test_id, dict())
                            test_subject = test[sensor].to_frame()
                            subject_data["test"][subject][test_id].setdefault(sensor, test_subject)

                    else:
                        train_stress = (train["label"] == 1).sum()
                        train_non_stress = (train["label"] == 0).sum()
                        test_stress = (test_combined["label"] == 1).sum()
                        test_non_stress = (test_combined["label"] == 0).sum()
                        labels.setdefault(subject, {"test_stress": test_stress, "test_non_stress": test_non_stress,
                                                    "train_stress": train_stress, "train_non_stress": train_non_stress})

        return subject_data, labels

    @classmethod
    def test_max_window_size(cls, dataset: Dataset, test_window_sizes: List[int], additional_windows: int,
                             resample_factor: int, multi: int = 3) -> bool:
        """
        Test all given test window-sizes if they are valid
        :param dataset: Specify dataset
        :param test_window_sizes: List with all test_window-sizes to be tested
        :param additional_windows: Specify amount of additional windows to be removed around test-window
        :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
        :param multi: Specify number of combined single attacks
        :return: Boolean -> False if there is at least one wrong test window-size
        """
        data_dict = dataset.load_dataset(resample_factor=resample_factor)  # read data_dict
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
    def calculate_alignment(cls, dataset: Dataset, subject_id: int, method: str, test_window_size: int, multi: int,
                            resample_factor: int = 1, additional_windows: int = 1000) -> Dict[int, Dict[str, float]]:
        """
        Calculate DTW-Alignments for sensor data using Dynamic Time Warping
        :param dataset: Specify dataset
        :param subject_id: Specify which subject should be used as test subject
        :param method: String to specify which method should be used (non-stress / stress)
        :param test_window_size: Specify amount of windows (datapoints) in test set (int)
        :param multi: Specify number of combined single attacks
        :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
        :param additional_windows: Specify amount of additional windows to be removed around test-window
        :return: Tuple with Dictionaries of standard and normalized results
        """
        results_standard = dict()
        subject_data, labels = cls.create_subject_data(dataset=dataset, method=method,
                                                       test_window_size=test_window_size, subject_id=subject_id,
                                                       multi=multi, additional_windows=additional_windows,
                                                       resample_factor=resample_factor)

        for subject in subject_data["train"]:
            results_standard.setdefault(subject, dict())

            for sensor in subject_data["train"][subject]:
                train = subject_data["train"][subject][sensor]
                train = train.values.flatten()
                for m in subject_data["test"][subject_id]:
                    test = subject_data["test"][subject_id][m][sensor]
                    test = test.values.flatten()

                    distance_standard = dtw.distance_fast(train, test)
                    results_standard[subject].setdefault(m, dict())
                    results_standard[subject][m].setdefault(sensor, round(distance_standard, 4))

        # Calculate average distances
        for subject in results_standard:
            results_standard[subject].setdefault("mean", dict())
            for sensor in results_standard[subject][0]:
                sensor_results = list()
                for m in results_standard[subject]:
                    if m != "mean":
                        sensor_results.append(results_standard[subject][m][sensor])
                results_standard[subject]["mean"].setdefault(sensor, round(statistics.mean(sensor_results), 4))

        return results_standard

    def run_calculations(self, dataset: Dataset, test_window_sizes: List[int], multi: int = 3, resample_factor: int = 1,
                         additional_windows: int = 1000, n_jobs: int = -1, methods: List[str] = None,
                         subject_ids: List[int] = None):
        """
        Run DTW-calculations with all given parameters and save results as json
        :param dataset: Specify dataset, which should be used
        :param test_window_sizes: List with all test windows that should be used (int)
        :param multi: Specify number of combined single attacks
        :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
        :param additional_windows: Specify amount of additional windows to be removed around test-window
        :param n_jobs: Number of processes to use (parallelization)
        :param methods:  List with all method that should be used -> "non-stress" / "stress" (str)
        :param subject_ids: List with all subjects that should be used as test subjects (int) -> None = all subjects
        """
        def parallel_calculation(current_subject_id: int):
            """
            Run parallel calculations
            :param current_subject_id: Specify subject_id
            """
            results_standard = self.calculate_alignment(dataset=dataset, subject_id=current_subject_id, method=method,
                                                        test_window_size=test_window_size, multi=multi,
                                                        additional_windows=additional_windows,
                                                        resample_factor=resample_factor)

            # Save results as json
            try:
                path = os.path.join(cfg.out_dir, dataset.name)  # add /dataset-name to path
                path = os.path.join(path, "resample-factor=" + str(resample_factor))  # add /rs-factor= to path
                path = os.path.join(path, self.name)
                path = os.path.join(path, "alignments")  # add /alignments to path
                path = os.path.join(path, str(method))  # add /method to path
                path = os.path.join(path, "window-size=" + str(test_window_size))  # add /test=X to path
                os.makedirs(path, exist_ok=True)

                path_string_standard = "SW-DTW_results_standard_" + str(method) + "_" + str(
                    test_window_size) + "_S" + str(current_subject_id) + ".json"

                with open(os.path.join(path, path_string_standard), "w", encoding="utf-8") as outfile:
                    json.dump(results_standard, outfile)

                print("SW-DTW results saved at: " + str(os.path.join(path, path_string_standard)))

            except FileNotFoundError:
                with open("SW-DTW_results_standard_" + str(method) + "_" + str(test_window_size) + "_S" +
                          str(current_subject_id) + ".json", "w", encoding="utf-8") as outfile:
                    json.dump(results_standard, outfile)

                print("FileNotFoundError: results saved at working dir")

        if subject_ids is None:
            subject_ids = dataset.get_subject_list()
        if methods is None:
            methods = dataset.get_classes()

        if self.test_max_window_size(dataset=dataset, test_window_sizes=test_window_sizes,
                                     additional_windows=additional_windows, resample_factor=resample_factor,
                                     multi=multi):
            print("Test-window-size test successful: All test-window-sizes are valid")

            for test_window_size in test_window_sizes:
                start = time.perf_counter()
                print("-Current window-size: " + str(test_window_size))
                for method in methods:
                    print("--Current method: " + str(method))

                    # Parallelization
                    with Parallel(n_jobs=n_jobs) as parallel:
                        parallel(
                            delayed(parallel_calculation)(current_subject_id=subject_id) for subject_id in subject_ids)

                    end = time.perf_counter()

                    # Save runtime as txt file
                    dataset_path = os.path.join(cfg.out_dir, dataset.name)
                    resample_path = os.path.join(dataset_path, "resample-factor=" + str(resample_factor))
                    attack_path = os.path.join(resample_path, self.name)
                    alignments_path = os.path.join(attack_path, "alignments")
                    runtime_path = os.path.join(alignments_path, "runtime")
                    os.makedirs(runtime_path, exist_ok=True)

                    file_name = "runtime_window_size=" + str(test_window_size) + ".txt"
                    save_path = os.path.join(runtime_path, file_name)

                    text_file = open(save_path, "w")
                    text = "Runtime: " + str(end - start)
                    text_file.write(text)
                    text_file.close()

        else:
            print("Please specify valid window-sizes!")
