from preprocessing.data_processing.data_processing import DataProcessing
from preprocessing.datasets.dataset import Dataset
from alignments.dtw_attacks.dtw_attack import DtwAttack
from alignments.dtw_attacks.multi_dtw_attack import MultiDtwAttack
from config import Config

import json
import os
import statistics
from typing import Dict, Union, List


cfg = Config.get()


def load_results(dataset: Dataset, resample_factor: int, data_processing: DataProcessing, dtw_attack: DtwAttack,
                 subject_id: int, method: str, test_window_size: int) -> Dict[str, Dict[str, float]]:
    """
    Load DTW-attack results from ../out/alignments/
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-attack
    :param subject_id: Specify subject
    :param method: Specify method ("non-stress", "stress")
    :param test_window_size: Specify test-window-size
    :return: Dictionary with results
    """
    subject_ids = dataset.get_subject_list()
    subject_ids_string = list()
    for subject in subject_ids:
        subject_ids_string.append(str(subject))

    reduced_results = dict()
    try:
        data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
        resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
        attack_path = os.path.join(resample_path, dtw_attack.get_attack_name())  # add /attack-name to path
        processing_path = os.path.join(attack_path, data_processing.name)  # add /data-processing to path
        alignment_path = os.path.join(processing_path, "alignments")  # add /alignments to path
        method_path = os.path.join(alignment_path, str(method))  # add /method to path
        window_path = os.path.join(method_path, "window-size=" + str(test_window_size))  # add /test=X to path

        path_string = "SW-DTW_results_standard_" + str(method) + "_" + str(test_window_size) + "_S" + str(
            subject_id) + ".json"
        path = os.path.join(window_path, path_string)

        f = open(path, "r")
        results = json.loads(f.read())

        # If Multi-DTW-Attack just use mean results
        if dtw_attack.get_attack_name() == MultiDtwAttack().get_attack_name():
            multi_dtw_attack_results = dict()
            for subject in results:
                multi_dtw_attack_results.setdefault(subject, results[subject]["mean"])
            results = multi_dtw_attack_results

        # Calculate mean of all 3 "ACC" Sensor distances
        for i in results:
            if "acc_x" in results[i]:
                results[i].setdefault("acc", round(statistics.mean([results[i]["acc_x"], results[i]["acc_y"],
                                                                    results[i]["acc_z"]]), 4))

        # Reduce results to specified subject-ids
        reduced_results = dict()
        for subject_id in results:
            if subject_id in subject_ids_string:
                reduced_results.setdefault(subject_id, results[subject_id])

    except FileNotFoundError:
        print("FileNotFoundError: no results with this configuration available")

    return reduced_results


def load_max_precision_results(dataset: Dataset, resample_factor: int, dtw_attack: DtwAttack, method: str,
                               test_window_size: int, k: int) -> Dict[str, Union[float, List[Dict[str, float]]]]:
    """
    Load max-precision results
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param dtw_attack: Specify DTW-attack
    :param method: Specify method
    :param test_window_size: Specify test-window-size
    :param k: Specify k
    :return: Dictionary with results
    """
    results = dict()
    try:
        data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
        resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
        attack_path = os.path.join(resample_path, dtw_attack.get_attack_name())  # add /attack-name to path
        precision_path = os.path.join(attack_path, "precision")  # add /precision to path
        method_path = os.path.join(precision_path, str(method))  # add /method to path
        window_path = os.path.join(method_path, "window-size=" + str(test_window_size))  # add /test=0.XX to path
        max_precision_path = os.path.join(window_path, "max-precision")  # add /max-precision to path

        file_name = "SW-DTW_max-precision_" + str(method) + "_" + str(test_window_size) + "_k=" + str(k) + ".json"
        save_path = os.path.join(max_precision_path, file_name)

        f = open(save_path, "r")
        results = json.loads(f.read())

    except FileNotFoundError:
        print("FileNotFoundError: no max-precision with this configuration available")

    return results


def load_complete_alignment_results(dataset: Dataset, resample_factor: int, data_processing: DataProcessing,
                                    subject_id: int) -> Dict[str, float]:
    """
    Load complete alignment results from ../out/alignments/complete
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param subject_id: Specify subject-id
    :return: Dictionary with results
    """
    average_results = dict()
    try:
        data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
        resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
        complete_path = os.path.join(resample_path, "complete-alignments")  # add /complete to path
        processing_path = os.path.join(complete_path, data_processing.name)  # add /data-processing to path

        path = os.path.join(processing_path, "SW-DTW_results_standard_complete_S" + str(subject_id) + ".json")

        f = open(path, "r")
        results = json.loads(f.read())

        # Calculate mean of all 3 "ACC" Sensor distances
        for i in results:
            if "acc_x" in results[i]:
                results[i].setdefault("acc", round(statistics.mean([results[i]["acc_x"], results[i]["acc_y"],
                                                                    results[i]["acc_z"]]), 4))

        for i in results:
            sensor_results_list = list()
            for sensor in results[i]:
                sensor_results_list.append(results[i][sensor])
            average_results.setdefault(i, round(statistics.mean(sensor_results_list), 2))

    except FileNotFoundError:
        print("FileNotFoundError: no results with this configuration available")

    return average_results
