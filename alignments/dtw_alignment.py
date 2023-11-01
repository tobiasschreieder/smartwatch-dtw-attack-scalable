from preprocessing.data_processing.data_processing import DataProcessing
from preprocessing.datasets.load_wesad import Dataset
from config import Config

from dtaidistance import dtw
from joblib import Parallel, delayed
import pandas as pd
from typing import Dict, List
import json
import os


cfg = Config.get()


def create_full_subject_data(data_dict: Dict[int, pd.DataFrame], subject_id: int) -> Dict[str, pd.DataFrame]:
    """
    Create dictionary with all subjects and their sensor data as Dataframe
    :param data_dict: Dictionary with preprocessed dataset
    :param subject_id: Specify subject_id
    :return: Dictionary with subject_data
    """
    sensor_data = dict()
    for sensor in data_dict[subject_id]:
        if sensor != "label":
            sensor_data.setdefault(sensor, data_dict[subject_id][sensor])

    return sensor_data


def calculate_complete_subject_alignment(data_dict: Dict[int, pd.DataFrame], dataset: Dataset, subject_id: int) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate dtw-alignments for all sensors and subjects (no train-test split)
    :param data_dict: Dictionary with preprocessed dataset
    :param dataset: Specify dataset
    :param subject_id: Specify subject-id
    :return: Dictionary of standard results (not normalized)
    """
    results_standard = dict()
    subject_data_1 = create_full_subject_data(data_dict=data_dict, subject_id=subject_id)
    subject_list = dataset.get_subject_list()

    for subject in subject_list:
        subject_data_2 = create_full_subject_data(data_dict=data_dict, subject_id=subject)
        results_standard.setdefault(subject, dict())

        for sensor in subject_data_2:
            test = subject_data_1[sensor]
            train = subject_data_2[sensor]
            test = test.values.flatten()
            train = train.values.flatten()

            distance_standard = dtw.distance_fast(train, test)
            results_standard[subject].setdefault(sensor, round(distance_standard, 4))

    return results_standard


def run_dtw_alignments(dataset: Dataset, data_processing: DataProcessing, resample_factor: int, n_jobs: int = -1,
                       subject_ids: List[int] = None):
    """
    Run DTW-Calculations with all given parameters and save results as json (no train-test split)
    :param dataset: Specify dataset
    :param data_processing: Specify type of data-processing
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param n_jobs: Number of processes to use (parallelization)
    :param subject_ids: List with all subjects that should be used as test subjects (int) -> None = all subjects
    """
    def parallel_calculation(current_subject_id: int):
        """
        Run parallel alignment calculations
        :param current_subject_id: Specify subject-id
        :return: Dictionary with results
        """
        result = calculate_complete_subject_alignment(data_dict=data_dict, dataset=dataset,
                                                      subject_id=current_subject_id)
        results_subject = {current_subject_id: result}

        return results_subject

    if subject_ids is None:
        subject_ids = dataset.get_subject_list()

    data_dict = dataset.load_dataset(resample_factor=resample_factor, data_processing=data_processing)

    # Run DTW Calculations
    # Parallelization
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(delayed(parallel_calculation)(current_subject_id=subject_id) for subject_id in subject_ids)

    results_standard = dict()
    for res in results:
        results_standard.setdefault(list(res.keys())[0], list(res.values())[0])

    data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
    resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
    complete_path = os.path.join(resample_path, "Complete-Alignments")  # add /complete to path
    processing_path = os.path.join(complete_path, data_processing.name)  # add /data-processing to path
    os.makedirs(processing_path, exist_ok=True)

    try:
        path_string_standard = "SW-DTW_results_standard_complete.json"

        with open(os.path.join(processing_path, path_string_standard), "w", encoding="utf-8") as outfile:
            json.dump(results_standard, outfile)

        print("SW-DTW results saved at: " + str(os.path.join(processing_path, path_string_standard)))

    except FileNotFoundError:
        print("FileNotFoundError: results could not be saved!")
