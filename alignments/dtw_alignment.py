from preprocessing.data_preparation import load_dataset, get_subject_list

from dtw import *
import pandas as pd
import scipy.signal
from typing import Dict, List, Tuple
import json
import os


MAIN_PATH = os.path.abspath(os.getcwd())
DATA_PATH = os.path.join(MAIN_PATH, "dataset")  # add /dataset to path


def create_full_subject_data(subject_id: int, resample_factor: float) -> Dict[str, pd.DataFrame]:
    """
    Create dictionary with all subjects and their sensor data as Dataframe
    :param subject_id: Specify subject_id
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :return: Dictionary with subject_data
    """
    data_dict = load_dataset()

    sensor_data = {"bvp": scipy.signal.resample(data_dict[subject_id][["bvp"]],
                                                round(len(data_dict[subject_id][["bvp"]]) / resample_factor)),
                   "eda": scipy.signal.resample(data_dict[subject_id][["eda"]],
                                                round(len(data_dict[subject_id][["eda"]]) / resample_factor)),
                   "acc_x": scipy.signal.resample(data_dict[subject_id][["acc_x"]],
                                                  round(len(data_dict[subject_id][["acc_x"]]) / resample_factor)),
                   "acc_y": scipy.signal.resample(data_dict[subject_id][["acc_y"]],
                                                  round(len(data_dict[subject_id][["acc_y"]]) / resample_factor)),
                   "acc_z": scipy.signal.resample(data_dict[subject_id][["acc_z"]],
                                                  round(len(data_dict[subject_id][["acc_z"]]) / resample_factor)),
                   "temp": scipy.signal.resample(data_dict[subject_id][["temp"]],
                                                 round(len(data_dict[subject_id][["temp"]]) / resample_factor))}

    return sensor_data


def calculate_complete_subject_alignment(subject_id: int, resample_factor: float) \
        -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    """
    Calculate dtw-alignments for all sensors and subjects (no train-test split)
    :param subject_id: Specify subject-id
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :return: Tuple with Dictionaries of normalized and standard results
    """
    results_normalized = dict()
    results_standard = dict()
    subject_data_1 = create_full_subject_data(resample_factor=resample_factor, subject_id=subject_id)
    subject_list = get_subject_list()

    for subject in subject_list:
        print("--Current subject: " + str(subject))
        subject_data_2 = create_full_subject_data(resample_factor=resample_factor, subject_id=subject)
        results_normalized.setdefault(subject, dict())
        results_standard.setdefault(subject, dict())

        for sensor in subject_data_2:
            test = subject_data_1[sensor]
            train = subject_data_2[sensor]

            alignment = dtw(train, test, keep_internals=False)
            distance_normalized = alignment.normalizedDistance
            distance_standard = alignment.distance

            results_normalized[subject].setdefault(sensor, round(distance_normalized, 4))
            results_standard[subject].setdefault(sensor, round(distance_standard, 4))

    return results_normalized, results_standard


def run_dtw_alignments(resample_factor: float = 2, subject_ids: List[int] = None):
    """
    Run DTW-Calculations with all given parameters and save results as json (no train-test split)
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param subject_ids: List with all subjects that should be used as test subjects (int) -> None = all subjects
    """
    if subject_ids is None:
        subject_ids = get_subject_list()

    # Run DTW Calculations
    for subject_id in subject_ids:
        print("-Current id: " + str(subject_id))

        results_normalized, results_standard = calculate_complete_subject_alignment(subject_id=subject_id,
                                                                                    resample_factor=resample_factor)

        # Save results as json
        try:
            path = os.path.join(MAIN_PATH, "out")  # add /out to path
            path = os.path.join(path, "alignments")  # add /alignments to path
            path = os.path.join(path, "complete")  # add /complete to path
            os.makedirs(path, exist_ok=True)

            path_string_normalized = "/SW-DTW_results_normalized_" + "complete_S" + str(subject_id) + ".json"
            path_string_standard = "/SW-DTW_results_standard_" + "complete_S" + str(subject_id) + ".json"

            with open(path + path_string_normalized, "w", encoding="utf-8") as outfile:
                json.dump(results_normalized, outfile)

            with open(path + path_string_standard, "w", encoding="utf-8") as outfile:
                json.dump(results_standard, outfile)

            print("SW-DTW results saved at: " + str(path))

        except FileNotFoundError:
            with open("/SW-DTW_results_normalized_" + "complete_S" + str(subject_id) + ".json", "w", encoding="utf-8") \
                    as outfile:
                json.dump(results_normalized, outfile)

            with open("/SW-DTW_results_standard_" + "complete_S" + str(subject_id) + ".json", "w", encoding="utf-8") \
                    as outfile:
                json.dump(results_standard, outfile)

            print("FileNotFoundError: results saved at working dir")
