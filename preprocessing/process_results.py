from preprocessing.datasets.dataset import Dataset
from config import Config

import json
import os
import statistics
from typing import Dict, Union, List


cfg = Config.get()


def load_results(dataset: Dataset, resample_factor: int, subject_id: int, method: str, test_window_size: int,
                 normalized_data: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Load DTW-attack results from ../out/alignments/
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param subject_id: Specify subject
    :param method: Specify method ("baseline", "amusement", "stress")
    :param test_window_size: Specify test-window-size
    :param normalized_data: True if normalized results should be used
    :return: Dictionary with results
    """
    results = dict()
    try:
        data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
        resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
        alignment_path = os.path.join(resample_path, "alignments")  # add /alignments to path
        method_path = os.path.join(alignment_path, str(method))  # add /method to path
        window_path = os.path.join(method_path, "window-size=" + str(test_window_size))  # add /test=X to path

        if normalized_data:
            path_string = "SW-DTW_results_normalized_" + str(method) + "_" + str(test_window_size) + "_S" + str(
                subject_id) + ".json"
            path = os.path.join(window_path, path_string)
        else:
            path_string = "SW-DTW_results_standard_" + str(method) + "_" + str(test_window_size) + "_S" + str(
                subject_id) + ".json"
            path = os.path.join(window_path, path_string)

        f = open(path, "r")
        results = json.loads(f.read())

        # Calculate mean of all 3 "ACC" Sensor distances
        for i in results:
            results[i].setdefault("acc", round(statistics.mean([results[i]["acc_x"], results[i]["acc_y"],
                                                                results[i]["acc_z"]]), 4))

    except FileNotFoundError:
        print("FileNotFoundError: no results with this configuration available")

    return results


def load_max_precision_results(dataset: Dataset, resample_factor: int, method: str, test_window_size: int, k: int) \
        -> Dict[str, Union[float, List[Dict[str, float]]]]:
    """
    Load max-precision results
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param method: Specify method
    :param test_window_size: Specify test-window-size
    :param k: Specify k
    :return: Dictionary with results
    """
    results = dict()
    try:
        data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
        resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
        precision_path = os.path.join(resample_path, "precision")  # add /precision to path
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


def load_complete_alignment_results(dataset: Dataset, resample_factor: int, subject_id: int,
                                    normalized_data: bool = False) -> Dict[str, float]:
    """
    Load complete alignment results from ../out/alignments/complete
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param subject_id: Specify subject-id
    :param normalized_data: If True: normalized data is loaded
    :return: Dictionary with results
    """
    average_results = dict()
    try:
        data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
        resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
        alignments_path = os.path.join(resample_path, "alignments")  # add /subject-plots to path
        complete_path = os.path.join(alignments_path, "complete")  # add /complete to path

        if normalized_data:
            path = os.path.join(complete_path, "SW-DTW_results_normalized_complete_S" + str(subject_id) + ".json")
        else:
            path = os.path.join(complete_path, "SW-DTW_results_standard_complete_S" + str(subject_id) + ".json")

        f = open(path, "r")
        results = json.loads(f.read())

        # Calculate mean of all 3 "ACC" Sensor distances
        for i in results:
            results[i].setdefault("acc", round(statistics.mean([results[i]["acc_x"], results[i]["acc_y"],
                                                                results[i]["acc_z"]]), 4))

        for i in results:
            average_results.setdefault(i, round(statistics.mean([results[i]["acc"], results[i]["bvp"],
                                                                 results[i]["eda"], results[i]["temp"]]), 2))

    except FileNotFoundError:
        print("FileNotFoundError: no results with this configuration available")

    return average_results
