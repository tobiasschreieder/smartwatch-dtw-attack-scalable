from preprocessing.data_preparation import load_dataset, get_subject_list

from typing import Dict, Tuple, List
import pandas as pd
from dtw import *
import os
import json


MAIN_PATH = os.path.abspath(os.getcwd())
DATA_PATH = os.path.join(MAIN_PATH, "dataset")  # add /dataset to path

CLASSES = ["baseline", "amusement", "stress"]  # All available classes
# All calculated window-sizes (test-proportions)
PROPORTIONS_TEST = [0.0001, 0.001, 0.01, 0.05, 0.1]


def get_classes() -> List[str]:
    """
    Get classes ("baseline", "amusement", "stress")
    :return: List with all classes
    """
    return CLASSES


def get_proportions() -> List[float]:
    """
    Get test-proportions
    :return: List with all test-proportions
    """
    return PROPORTIONS_TEST


def create_subject_data(method: str, proportion_test: float, subject_id: int) \
        -> Tuple[Dict[str, Dict[int, Dict[str, pd.DataFrame]]], Dict[int, Dict[str, int]]]:
    """
    Create dictionary with all subjects and their sensor data as Dataframe split into train and test data
    Create dictionary with label information for test subject
    :param method: String to specify which method should be used (baseline / amusement / stress)
    :param proportion_test: Specify the test proportion 0.XX (float)
    :param subject_id: Specify which subject should be used as test subject
    :return: Tuple with create subject_data and labels (containing label information)
    """
    data_dict = load_dataset()
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

            test_data_amount = round(len(data_dict[subject]) * proportion_test)
            method_proportion_test = test_data_amount / len(label_data)

            data_start = label_data.iloc[round(len(label_data) * 0.5):, :]
            data_end = label_data.iloc[:round(len(label_data) * 0.5), :]

            test_1 = data_end.iloc[round(len(data_end) * (1 - method_proportion_test)):, :]
            test_2 = data_start.iloc[:round(len(data_start) * method_proportion_test), :]
            test = pd.concat([test_1, test_2])

            train = data_dict[subject].drop(test.index)

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


def test_max_proportions(proportions: List[float], safety_proportion: float = 0.05) -> bool:
    """
    Test all given test proportions if they are valid
    :param proportions: List with all test proportions
    :param safety_proportion:
    :return: Boolean -> False if there is at least one wrong proportion
    """
    data_dict = load_dataset()  # read data_dict

    # Calculate max_proportion
    min_method_length = 0
    min_overall_length = 0

    for subject, data in data_dict.items():
        if min_overall_length == 0 or len(data) < min_overall_length:
            min_overall_length = len(data)

        baseline_length = len(data[data.label == 0])
        amusement_length = len(data[data.label == 0.5])
        stress_length = len(data[data.label == 1])
        min_method_subject_length = min(baseline_length, amusement_length, stress_length)

        if min_method_length == 0 or min_method_subject_length < min_method_length:
            min_method_length = min_method_subject_length

    max_proportion = round(min_method_length * (1 - safety_proportion) / min_overall_length, 4)

    # Test all proportions
    valid_proportion = True
    for proportion in proportions:
        if proportion <= 0.0 or proportion > max_proportion:
            valid_proportion = False
            print("Wrong proportion: " + str(proportion))

    return valid_proportion


def calculate_alignment(subject_id: int, method: str, proportion_test: float) \
        -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    """
    Calculate DTW-Alignments for sensor data using Dynamic Time Warping
    :param subject_id: Specify which subject should be used as test subject
    :param method: String to specify which method should be used (baseline / amusement / stress)
    :param proportion_test: Specify the test proportion 0.XX (float)
    :return: Tuple with Dictionaries of standard and normalized results
    """
    results_normalized = dict()
    results_standard = dict()
    subject_data, labels = create_subject_data(method=method, proportion_test=proportion_test, subject_id=subject_id)

    for subject in subject_data["train"]:
        print("----Current subject: " + str(subject))
        results_normalized.setdefault(subject, dict())
        results_standard.setdefault(subject, dict())

        for sensor in subject_data["train"][subject]:
            test = subject_data["test"][subject_id][sensor]
            train = subject_data["train"][subject][sensor]

            alignment = dtw(train, test, keep_internals=False)
            distance_normalized = alignment.normalizedDistance
            distance_standard = alignment.distance

            results_normalized[subject].setdefault(sensor, round(distance_normalized, 4))
            results_standard[subject].setdefault(sensor, round(distance_standard, 4))

    return results_normalized, results_standard


def run_calculations(proportions: List[float], methods: List[str] = None, subject_ids: List[int] = None):
    """
    Run DTW-Calculations with all given Parameters and save results as json
    :param proportions: List with all test proportions that should be used (float)
    :param methods:  List with all method that should be used -> "baseline" / "amusement" / "stress" (str)
    :param subject_ids: List with all subjects that should be used as test subjects (int) -> None = all subjects
    """
    if subject_ids is None:
        subject_ids = get_subject_list()
    if methods is None:
        methods = get_classes()

    if test_max_proportions(proportions=proportions):
        print("Test proportion test successful: All proportions are valid")

        for method in methods:
            print("-Current method: " + str(method))
            for proportion_test in proportions:
                print("--Current test proportion: " + str(proportion_test))

                # Run DTW Calculations
                for subject_id in subject_ids:
                    print("---Current id: " + str(subject_id))

                    results_normalized, results_standard = calculate_alignment(subject_id=subject_id, method=method,
                                                                               proportion_test=proportion_test)

                    # Save results as json
                    try:
                        path = os.path.join(MAIN_PATH, "/out")  # add /out to path
                        path = os.path.join(path, "/alignments")  # add /alignments to path
                        path = os.path.join(path, str(method))  # add /method to path
                        path = os.path.join(path, "test=" + str(proportion_test))  # add /test=0.XX to path
                        os.makedirs(path, exist_ok=True)

                        path_string_normalized = "/SW-DTW_results_normalized_" + str(method) + "_" + str(
                            proportion_test) + "_S" + str(subject_id) + ".json"
                        path_string_standard = "/SW-DTW_results_standard_" + str(method) + "_" + str(
                            proportion_test) + "_S" + str(subject_id) + ".json"

                        with open(path + path_string_normalized, "w", encoding="utf-8") as outfile:
                            json.dump(results_normalized, outfile)

                        with open(path + path_string_standard, "w", encoding="utf-8") as outfile:
                            json.dump(results_standard, outfile)

                        print("SW-DTW results saved at: " + str(path))

                    except FileNotFoundError:
                        with open("/SW-DTW_results_normalized_" + str(method) + "_" + str(proportion_test) + "_S" +
                                  str(subject_id) + ".json", "w", encoding="utf-8") as outfile:
                            json.dump(results_normalized, outfile)

                        with open("/SW-DTW_results_standard_" + str(method) + "_" + str(proportion_test) + "_S" +
                                  str(subject_id) + ".json", "w", encoding="utf-8") as outfile:
                            json.dump(results_standard, outfile)

                        print("FileNotFoundError: results saved at working dir")

    else:
        print("Please specify valid proportions!")
