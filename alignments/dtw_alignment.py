from preprocessing.datasets.load_wesad import Dataset
from config import Config

from dtaidistance import dtw
from joblib import Parallel, delayed
import pandas as pd
from typing import Dict, List
import json
import os


cfg = Config.get()


def create_full_subject_data(dataset: Dataset, subject_id: int, resample_factor: int) -> Dict[str, pd.DataFrame]:
    """
    Create dictionary with all subjects and their sensor data as Dataframe
    :param dataset: Specify dataset
    :param subject_id: Specify subject_id
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :return: Dictionary with subject_data
    """
    data_dict = dataset.load_dataset(resample_factor=resample_factor)

    sensor_data = {"bvp": data_dict[subject_id][["bvp"]],
                   "eda": data_dict[subject_id][["eda"]],
                   "acc_x": data_dict[subject_id][["acc_x"]],
                   "acc_y": data_dict[subject_id][["acc_y"]],
                   "acc_z": data_dict[subject_id][["acc_z"]],
                   "temp": data_dict[subject_id][["temp"]]
                   }

    return sensor_data


def calculate_complete_subject_alignment(dataset: Dataset, subject_id: int, resample_factor: int) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate dtw-alignments for all sensors and subjects (no train-test split)
    :param dataset: Specify dataset
    :param subject_id: Specify subject-id
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :return: Dictionary of standard results (not normalized)
    """
    results_standard = dict()
    subject_data_1 = create_full_subject_data(dataset=dataset, resample_factor=resample_factor, subject_id=subject_id)
    subject_list = dataset.get_subject_list()

    for subject in subject_list:
        subject_data_2 = create_full_subject_data(dataset=dataset, resample_factor=resample_factor, subject_id=subject)
        results_standard.setdefault(subject, dict())

        for sensor in subject_data_2:
            test = subject_data_1[sensor]
            train = subject_data_2[sensor]
            test = test.values.flatten()
            train = train.values.flatten()

            distance_standard = dtw.distance_fast(train, test)
            results_standard[subject].setdefault(sensor, round(distance_standard, 4))

    return results_standard


def run_dtw_alignments(dataset: Dataset, resample_factor: int, n_jobs: int = -1, subject_ids: List[int] = None):
    """
    Run DTW-Calculations with all given parameters and save results as json (no train-test split)
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param n_jobs: Number of processes to use (parallelization)
    :param subject_ids: List with all subjects that should be used as test subjects (int) -> None = all subjects
    """
    def parallel_calculation(current_subject_id: int):
        """
        Run parallel alignment calculations
        :param current_subject_id: Specify subject-id
        """
        results_standard = calculate_complete_subject_alignment(dataset=dataset, subject_id=current_subject_id,
                                                                resample_factor=resample_factor)

        # Save results as json
        try:
            path_string_standard = "SW-DTW_results_standard_" + "complete_S" + str(current_subject_id) + ".json"

            with open(os.path.join(complete_path, path_string_standard), "w", encoding="utf-8") as outfile:
                json.dump(results_standard, outfile)

            print("SW-DTW results saved at: " + str(os.path.join(complete_path, path_string_standard)))

        except FileNotFoundError:
            with open("/SW-DTW_results_standard_" + "complete_S" + str(current_subject_id) + ".json", "w",
                      encoding="utf-8") as outfile:
                json.dump(results_standard, outfile)

            print("FileNotFoundError: results saved at working dir")

    if subject_ids is None:
        subject_ids = dataset.get_subject_list()

    data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
    resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
    complete_path = os.path.join(resample_path, "complete-alignments")  # add /complete to path
    os.makedirs(complete_path, exist_ok=True)

    # Run DTW Calculations
    # Parallelization
    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(delayed(parallel_calculation)(current_subject_id=subject_id) for subject_id in subject_ids)


