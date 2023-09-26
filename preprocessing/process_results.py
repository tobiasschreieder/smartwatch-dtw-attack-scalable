import json
import os
import statistics
from typing import Dict, Union, List


# Specify path
MAIN_PATH = os.path.abspath(os.getcwd())
OUT_PATH = os.path.join(MAIN_PATH, "out")  # add /out to path
ALIGNMENT_PATH = os.path.join(OUT_PATH, "alignments")  # add /alignments to path
PRECISION_PATH = os.path.join(OUT_PATH, "precision")  # add /precision to path


def load_results(subject_id: int, method: str, proportion_test: float, normalized_data: bool = True) \
        -> Dict[str, Dict[str, float]]:
    """
    Load DTW-attack results from ../out/alignments/
    :param subject_id: Specify subject
    :param method: Specify method ("baseline", "amusement", "stress")
    :param proportion_test: Specify test-proportion
    :param normalized_data: True if normalized results should be used
    :return: Dictionary with results
    """
    results = dict()
    try:
        path = os.path.join(ALIGNMENT_PATH, str(method))  # add /method to path
        path = os.path.join(path, "test=" + str(proportion_test))  # add /test=0.XX to path

        if normalized_data:
            path = path + "/SW-DTW_results_normalized_" + str(method) + "_" + str(proportion_test) + "_S" + str(
                subject_id) + ".json"
        else:
            path = path + "/SW-DTW_results_standard_" + str(method) + "_" + str(proportion_test) + "_S" + str(
                subject_id) + ".json"

        f = open(path, "r")
        results = json.loads(f.read())

        # Calculate mean of all 3 "ACC" Sensor distances
        for i in results:
            results[i].setdefault("acc", round(statistics.mean([results[i]["acc_x"], results[i]["acc_y"],
                                                                results[i]["acc_z"]]), 4))

    except FileNotFoundError:
        print("FileNotFoundError: no results with this configuration available")

    return results


def load_max_precision_results(method: str, proportion_test: float, k: int) \
        -> Dict[str, Union[float, List[Dict[str, float]]]]:
    """
    Load max-precision results
    :param method: Specify method
    :param proportion_test: Specify test-proportion
    :param k: Specify k
    :return: Dictionary with results
    """
    results = dict()
    try:
        path = os.path.join(PRECISION_PATH, str(method))  # add /method to path
        path = os.path.join(path, "test=" + str(proportion_test))  # add /test=0.XX to path
        path = os.path.join(path, "max-precision")  # add /max-precision to path

        path = path + "/SW-DTW_max-precision_" + str(method) + "_" + str(proportion_test) + "_k=" + str(k) + ".json"

        f = open(path, "r")
        results = json.loads(f.read())

    except FileNotFoundError:
        print("FileNotFoundError: no max-precision with this configuration available")

    return results


def load_complete_alignment_results(subject_id: int, normalized_data: bool = True) -> Dict[str, float]:
    """
    Load complete alignment results from ../out/alignments/complete
    :param subject_id: Specify subject-id
    :param normalized_data: If True: normalized data is loaded
    :return: Dictionary with results
    """
    average_results = dict()
    try:
        path = os.path.join(ALIGNMENT_PATH, "complete")  # add /complete to path

        if normalized_data:
            path = path + "/SW-DTW_results_normalized_complete_S" + str(subject_id) + ".json"
        else:
            path = path + "/SW-DTW_results_standard_complete_S" + str(subject_id) + ".json"

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
