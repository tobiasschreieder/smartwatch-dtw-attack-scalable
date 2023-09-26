from alignments.dtw_attack import get_classes
from evaluation.create_md_tables import create_md_precision_overall
from evaluation.optimization.class_evaluation import calculate_average_class_precisions, get_best_class_configuration, \
    get_class_distribution
from evaluation.optimization.rank_method_evaluation import calculate_rank_method_precisions, \
    get_best_rank_method_configuration
from evaluation.optimization.sensor_evaluation import calculate_sensor_precisions, get_best_sensor_configuration, \
    list_to_string
from evaluation.optimization.window_evaluation import calculate_window_precisions, get_best_window_configuration
from preprocessing.data_preparation import get_subject_list
from preprocessing.process_results import load_max_precision_results

from typing import Dict, List, Union
import os
import statistics
import json


MAIN_PATH = os.path.abspath(os.getcwd())
OUT_PATH = os.path.join(MAIN_PATH, "out")  # add /out to path
EVALUATIONS_PATH = os.path.join(OUT_PATH, "evaluations")  # add /evaluations to path


def calculate_best_configurations() -> Dict[str, Union[str, float, List[List[str]]]]:
    """
    Calculate the best configurations for rank-method, classes, sensors and windows
    :return: Dictionary with best configurations
    """
    # Best rank-method
    results = calculate_rank_method_precisions()
    best_rank_method = get_best_rank_method_configuration(res=results)

    # Best class
    average_results, weighted_average_results = calculate_average_class_precisions(rank_method=best_rank_method)
    best_class_method = get_best_class_configuration(average_res=average_results,
                                                     weighted_average_res=weighted_average_results)

    # Best sensors
    results = calculate_sensor_precisions(rank_method=best_rank_method, average_method=best_class_method)
    best_sensors = get_best_sensor_configuration(res=results)

    # Best window
    results = calculate_window_precisions(rank_method=best_rank_method, average_method=best_class_method,
                                          sensor_combination=best_sensors)
    best_window = get_best_window_configuration(res=results)

    best_configurations = {"rank_method": best_rank_method, "class": best_class_method, "sensor": best_sensors,
                           "window": best_window}

    return best_configurations


def get_average_max_precision(average_method: str, window: float, k: int) -> float:
    """
    Calculate average max-precision for specified averaging method
    :param average_method: Specify averaging method ("mean" or "weighted-mean")
    :param window: Specify window size (test-proportion)
    :param k: Specify k parameter
    :return: Average max-precision
    """
    baseline_results = load_max_precision_results(method="baseline", proportion_test=window, k=k)
    amusement_results = load_max_precision_results(method="amusement", proportion_test=window, k=k)
    stress_results = load_max_precision_results(method="stress", proportion_test=window, k=k)

    result = None
    try:
        # Averaging method = "mean"
        if average_method == "mean":
            result = round(statistics.mean([baseline_results["precision"], amusement_results["precision"],
                                            stress_results["precision"]]), 3)
        # Averaging method = "weighted-mean"
        else:
            class_distributions = get_class_distribution()
            result = round(baseline_results["precision"] * class_distributions["baseline"] +
                           amusement_results["precision"] * class_distributions["amusement"] +
                           stress_results["precision"] * class_distributions["stress"], 3)

    except KeyError:
        print("SW-DTW_max-precision for k=" + str(k) + " not available!")

    return result


def get_random_guess_precision(k: int) -> float:
    """
    Calculate precision for random guess
    :param k: Specify k parameter
    :return: random guess precision
    """
    amount_subjects = len(get_subject_list())
    result = round(k / amount_subjects, 3)
    return result


def calculate_optimized_precisions(k_list: List[int] = None) -> Dict[int, Dict[str, float]]:
    """
    Calculate overall evaluation precision scores (DTW-results, maximum results and random guess results)
    :return: Dictionary with results
    """
    if k_list is None:
        k_list = [1, 3, 5]

    best_configuration = calculate_best_configurations()
    results = calculate_window_precisions(rank_method=best_configuration["rank_method"],
                                          average_method=best_configuration["class"],
                                          sensor_combination=best_configuration["sensor"], k_list=k_list)

    overall_results = dict()
    for k in results:
        # Calculate DTW-Results
        overall_results.setdefault(k, {"results": results[k][best_configuration["window"]]})

        # Calculate maximum results
        overall_results[k].setdefault("max", get_average_max_precision(average_method=best_configuration["class"],
                                                                       window=best_configuration["window"], k=k))

        # Calculate random guess results
        overall_results[k].setdefault("random", get_random_guess_precision(k=k))

    return overall_results


def calculate_best_k_parameters() -> Dict[str, int]:
    """
    Calculate k-parameters where precision@k == 1
    :return: Dictionary with results
    """
    amount_subjects = len(get_subject_list())
    k_list = list(range(1, amount_subjects + 1))  # List with all possible k parameters
    results = calculate_optimized_precisions(k_list=k_list)
    best_k_parameters = dict()

    set_method = False
    for k in results:
        for method, value in results[k].items():
            if set_method is False:
                if value == 1.0:
                    best_k_parameters.setdefault(method, 1)
                else:
                    best_k_parameters.setdefault(method, amount_subjects)
            elif value == 1.0 and set_method is True:
                if best_k_parameters[method] > k:
                    best_k_parameters[method] = k
        set_method = True

    return best_k_parameters


def get_best_sensor_weightings(window_size: float, methods: List[str] = None, k_list: List[int] = None) \
        -> Dict[str, Dict[int, List[Dict[str, float]]]]:
    """
    Calculate best sensor-weightings for specified window-size
    :param window_size: Specify window-size (test-proportion)
    :param methods: List with methods (baseline, amusement, stress); if None: all methods are used
    :param k_list: Specify k parameters for precision@k; if None: 1, 3, 5 are used
    :return: Dictionary with weighting results
    """
    if methods is None:
        methods = get_classes()
    if k_list is None:
        k_list = [1, 3, 5]

    weightings = dict()
    for method in methods:
        weightings.setdefault(method, dict())
        for k in k_list:
            results = load_max_precision_results(k=k, method=method, proportion_test=window_size)
            weightings[method].setdefault(k, results["weights"])

    return weightings


def run_overall_evaluation(save_weightings: bool = False):
    """
    Run and save overall evaluation (DTW-results, maximum results, random guess results)
    """
    best_configuration = calculate_best_configurations()
    overall_results = calculate_optimized_precisions()
    weightings = get_best_sensor_weightings(window_size=best_configuration["window"])
    best_k_parameters = calculate_best_k_parameters()

    text = [create_md_precision_overall(results=overall_results, rank_method=best_configuration["rank_method"],
                                        average_method=best_configuration["class"],
                                        sensor_combination=list_to_string(input_list=best_configuration["sensor"][0]),
                                        window=best_configuration["window"], weightings=weightings,
                                        best_k_parameters=best_k_parameters)]

    # Save MD-File
    os.makedirs(EVALUATIONS_PATH, exist_ok=True)

    path_string = "/SW-DTW_evaluation_overall.md"
    with open(EVALUATIONS_PATH + path_string, 'w') as outfile:
        for item in text:
            outfile.write("%s\n" % item)

    print("SW-DTW evaluation overall saved at: " + str(EVALUATIONS_PATH))

    # Save weightings as JSON-File
    if save_weightings:
        path_string = "/SW-DTW_evaluation_weightings.json"
        with open(EVALUATIONS_PATH + path_string, "w", encoding="utf-8") as outfile:
            json.dump(weightings, outfile)

        print("SW-DTW evaluation weightings saved at: " + str(EVALUATIONS_PATH))
