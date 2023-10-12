from preprocessing.datasets.dataset import Dataset
from config import Config

from typing import Dict, Tuple, List
import pandas as pd
from dtaidistance import dtw
import os
import json
import time


cfg = Config.get()


WINDOWS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 100, 500, 1000]  # All calculated window-sizes
WINDOWS = [1, 2, 3]


def get_windows() -> List[int]:
    """
    Get test-window-sizes
    :return: List with all test-window-sizes
    """
    return WINDOWS


def create_subject_data(dataset: Dataset, method: str, test_window_size: int, subject_id: int,
                        additional_windows: int = 1000, resample_factor: int = 1) \
        -> Tuple[Dict[str, Dict[int, Dict[str, pd.DataFrame]]], Dict[int, Dict[str, int]]]:
    """
    Create dictionary with all subjects and their sensor data as Dataframe split into train and test data
    Create dictionary with label information for test subject
    :param dataset: Specify dataset
    :param method: String to specify which method should be used (baseline / amusement / stress)
    :param test_window_size: Specify amount of windows (datapoints) in test set (int)
    :param subject_id: Specify which subject should be used as test subject
    :param additional_windows: Specify amount of additional windows to be removed around test-window
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :return: Tuple with create subject_data and labels (containing label information)
    """
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

            # Method "baseline"
            if method == "baseline":
                label_data = label_data[label_data.label == 0]  # Data for subject with label == 0 -> baseline

            # Method "amusement"
            elif method == "amusement":
                label_data = label_data[label_data.label == 0.5]  # Data for subject with label == 0.5 -> amusement

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
                    test_baseline = (test["label"] == 0).sum()
                    test_amusement = (test["label"] == 0.5).sum()
                    train_stress = (train["label"] == 1).sum()
                    train_baseline = (train["label"] == 0).sum()
                    train_amusement = (train["label"] == 0.5).sum()
                    labels.setdefault(subject, {"test_stress": test_stress, "test_baseline": test_baseline,
                                                "test_amusement": test_amusement, "train_stress": train_stress,
                                                "train_baseline": train_baseline, "train_amusement": train_amusement})

    return subject_data, labels


def test_max_window_size(dataset: Dataset, test_window_sizes: List[int], additional_windows: int, resample_factor: int)\
        -> bool:
    """
    Test all given test window-sizes if they are valid
    :param dataset: Specify dataset
    :param test_window_sizes: List with all test_window-sizes to be tested
    :param additional_windows: Specify amount of additional windows to be removed around test-window
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :return: Boolean -> False if there is at least one wrong test window-size
    """
    data_dict = dataset.load_dataset(resample_factor=resample_factor)  # read data_dict
    additional_windows = max(1, int(round(additional_windows / resample_factor)))

    # Calculate max_window
    min_method_length = additional_windows * 2

    for subject, data in data_dict.items():
        baseline_length = len(data[data.label == 0])
        amusement_length = len(data[data.label == 0.5])
        stress_length = len(data[data.label == 1])
        min_method_subject_length = min(baseline_length, amusement_length, stress_length)

        if min_method_length == (additional_windows * 2) or min_method_subject_length < min_method_length:
            min_method_length = min_method_subject_length

    # Test all test-window-sizes
    valid_windows = True
    for test_window_size in test_window_sizes:
        if test_window_size <= 0 or test_window_size > min_method_length:
            valid_windows = False
            print("Wrong test-window-size: " + str(test_window_size))

    return valid_windows


def calculate_alignment(dataset: Dataset, subject_id: int, method: str, test_window_size: int, resample_factor: int = 1,
                        additional_windows: int = 10) -> Dict[int, Dict[str, float]]:
    """
    Calculate DTW-Alignments for sensor data using Dynamic Time Warping
    :param dataset: Specify dataset
    :param subject_id: Specify which subject should be used as test subject
    :param method: String to specify which method should be used (baseline / amusement / stress)
    :param test_window_size: Specify amount of windows (datapoints) in test set (int)
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param additional_windows: Specify amount of additional windows to be removed around test-window
    :return: Tuple with Dictionaries of standard and normalized results
    """
    results_standard = dict()
    subject_data, labels = create_subject_data(dataset=dataset, method=method, test_window_size=test_window_size,
                                               subject_id=subject_id, additional_windows=additional_windows,
                                               resample_factor=resample_factor)

    for subject in subject_data["train"]:
        print("----Current subject: " + str(subject))
        results_standard.setdefault(subject, dict())

        for sensor in subject_data["train"][subject]:
            test = subject_data["test"][subject_id][sensor]
            train = subject_data["train"][subject][sensor]
            test = test.values.flatten()
            train = train.values.flatten()

            distance_standard = dtw.distance_fast(train, test)
            results_standard[subject].setdefault(sensor, round(distance_standard, 4))

    return results_standard


def run_calculations(dataset: Dataset, test_window_sizes: List[int], resample_factor: int = 1,
                     additional_windows: int = 1000, methods: List[str] = None, subject_ids: List[int] = None):
    """
    Run DTW-calculations with all given parameters and save results as json
    :param dataset: Specify dataset, which should be used
    :param test_window_sizes: List with all test windows that should be used (int)
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param additional_windows: Specify amount of additional windows to be removed around test-window
    :param methods:  List with all method that should be used -> "baseline" / "amusement" / "stress" (str)
    :param subject_ids: List with all subjects that should be used as test subjects (int) -> None = all subjects
    """
    if subject_ids is None:
        subject_ids = dataset.get_subject_list()
    if methods is None:
        methods = dataset.get_classes()

    if test_max_window_size(dataset=dataset, test_window_sizes=test_window_sizes, resample_factor=resample_factor,
                            additional_windows=additional_windows):
        print("Test-window-size test successful: All test-window-sizes are valid")

        for test_window_size in test_window_sizes:
            start = time.perf_counter()
            print("-Current window-size: " + str(test_window_size))
            for method in methods:
                print("--Current method: " + str(method))

                # Run DTW Calculations
                for subject_id in subject_ids:
                    print("---Current id: " + str(subject_id))

                    results_standard = calculate_alignment(dataset=dataset, subject_id=subject_id, method=method,
                                                           test_window_size=test_window_size,
                                                           additional_windows=additional_windows,
                                                           resample_factor=resample_factor)

                    # Save results as json
                    try:
                        path = os.path.join(cfg.out_dir, dataset.name)  # add /dataset-name to path
                        path = os.path.join(path, "resample-factor=" + str(resample_factor))  # add /rs-factor= to path
                        path = os.path.join(path, "alignments")  # add /alignments to path
                        path = os.path.join(path, str(method))  # add /method to path
                        path = os.path.join(path, "window-size=" + str(test_window_size))  # add /test=X to path
                        os.makedirs(path, exist_ok=True)

                        path_string_standard = "SW-DTW_results_standard_" + str(method) + "_" + str(
                            test_window_size) + "_S" + str(subject_id) + ".json"

                        with open(os.path.join(path, path_string_standard), "w", encoding="utf-8") as outfile:
                            json.dump(results_standard, outfile)

                        print("SW-DTW results saved at: " + str(path))

                    except FileNotFoundError:
                        with open("SW-DTW_results_standard_" + str(method) + "_" + str(test_window_size) + "_S" +
                                  str(subject_id) + ".json", "w", encoding="utf-8") as outfile:
                            json.dump(results_standard, outfile)

                        print("FileNotFoundError: results saved at working dir")

            end = time.perf_counter()

            # Save runtime as txt file
            dataset_path = os.path.join(cfg.out_dir, dataset.name)
            resample_path = os.path.join(dataset_path, "resample-factor=" + str(resample_factor))
            alignments_path = os.path.join(resample_path, "alignments")
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
