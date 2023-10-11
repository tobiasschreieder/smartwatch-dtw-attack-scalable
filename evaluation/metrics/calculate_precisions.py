from preprocessing.datasets.load_wesad import get_subject_list
from evaluation.metrics.calculate_ranks import run_calculate_ranks, realistic_rank, get_realistic_ranks_combinations
from preprocessing.process_results import load_results
from config import Config

from typing import List, Dict, Union
import json
import os


cfg = Config.get()

# Specify path
PRECISION_PATH = os.path.join(cfg.out_dir, "precision")  # add /precision to path

SUBJECT_LIST = get_subject_list()


def calculate_precision(subject_ids: List[int], k: int, rank_method: str, method: str, test_window_size: int) -> float:
    """
    Calculate precision@k scores
    :param subject_ids: List with subject-ids
    :param k: Specify k parameter
    :param rank_method: Specify ranking method ("rank" or "score")
    :param method: DTW-method ("baseline", "amusement", "stress")
    :param test_window_size: Specify test-window-size
    :return: precision@k value
    """
    true_positives = 0
    for subject_id in subject_ids:
        results = load_results(subject_id=subject_id, method=method, test_window_size=test_window_size)
        overall_ranks, individual_ranks = run_calculate_ranks(results, rank_method)
        real_rank = realistic_rank(overall_ranks=overall_ranks, subject_id=subject_id)

        if real_rank < k:
            true_positives += 1

    precision = round(true_positives / len(subject_ids), 3)

    return precision


def calculate_precision_combinations(realistic_ranks_comb, k: int) -> Dict[str, float]:
    """
    Calculate precision@k scores for sensor combinations
    :param realistic_ranks_comb: Dictionary with rank combinations
    :param k: Specify parameter k for precision@k
    :return: Dictionary with precision values for combinations
    """
    precision_comb = dict()
    for i in realistic_ranks_comb:
        true_positives = 0
        for j in realistic_ranks_comb[i]:
            if j < k:
                true_positives += 1

        precision = round(true_positives / len(SUBJECT_LIST), 3)
        precision_comb.setdefault(i, precision)

    return precision_comb


def calculate_max_precision(k: int, step_width: float, method: str, test_window_size: int) \
        -> Dict[str, Union[float, List[float]]]:
    """
    Calculate and save maximum possible precision value with all sensor weight characteristics
    :param k: Specify k for precision@k
    :param step_width: Specify step_with for weights
    :param method: Specify method of alignments
    :param test_window_size: Specify test-window-size of alignments
    :return: Maximum-precision
    """
    weight_precisions = list()
    steps = int(100/(step_width*100))

    print("Calculation of maximum precision@" + str(k) + " for method = '" + str(method) + "' with test-window-size = '"
          + str(test_window_size) + "'")
    for step_bvp in range(0, steps):
        weight_bvp = step_bvp / steps

        for step_eda in range(0, steps):
            weight_eda = step_eda / steps
            for step_acc in range(0, steps):
                weight_acc = step_acc / steps
                for step_temp in range(0, steps):
                    weight_temp = step_temp / steps

                    if weight_bvp + weight_eda + weight_acc + weight_temp == 1:
                        sensor_combinations = [["bvp", "eda", "acc", "temp"]]
                        weights = {"bvp": weight_bvp, "eda": weight_eda, "acc": weight_acc, "temp": weight_temp}

                        realistic_ranks_comb = get_realistic_ranks_combinations(rank_method="max",
                                                                                combinations=sensor_combinations,
                                                                                method=method,
                                                                                test_window_size=test_window_size,
                                                                                weights=weights)
                        precision_combinations = calculate_precision_combinations(realistic_ranks_comb=
                                                                                  realistic_ranks_comb, k=k)

                        for c, p in precision_combinations.items():
                            weight_precisions.append({"precision": p, "weights": weights})

    max_precision = 0
    for precision in weight_precisions:
        if precision["precision"] > max_precision:
            max_precision = precision["precision"]

    max_precisions = {"precision": max_precision, "weights": list()}
    for precision in weight_precisions:
        if precision["precision"] == max_precision:
            max_precisions["weights"].append(precision["weights"])

    # Save max_precisions as json
    try:
        path = os.path.join(PRECISION_PATH, str(method))  # add /method to path
        path = os.path.join(path, "test=" + str(test_window_size))  # add /test=X to path
        path = os.path.join(path, "max-precision")  # add /max-precision to path
        os.makedirs(path, exist_ok=True)

        path_string_normalized = "/SW-DTW_max-precision_" + str(method) + "_" + str(test_window_size) + "_k=" + str(k) \
                                 + ".json"

        with open(path + path_string_normalized, "w", encoding="utf-8") as outfile:
            json.dump(max_precisions, outfile)

        print("SW-DTW max-precision saved at: " + str(path))

    except FileNotFoundError:
        with open("/SW-DTW_max-precision_" + str(method) + "_" + str(test_window_size) + "_k=" + str(k) + ".json",
                  "w", encoding="utf-8") as outfile:
            json.dump(max_precisions, outfile)

        print("FileNotFoundError: results saved at working dir")

    return max_precisions
