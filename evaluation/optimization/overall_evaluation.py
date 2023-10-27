from evaluation.create_md_tables import create_md_precision_overall
from evaluation.optimization.class_evaluation import calculate_average_class_precisions, get_best_class_configuration, \
    get_class_distribution
from evaluation.optimization.rank_method_evaluation import calculate_rank_method_precisions, \
    get_best_rank_method_configuration
from evaluation.optimization.sensor_evaluation import calculate_sensor_precisions, get_best_sensor_configuration, \
    list_to_string
from evaluation.optimization.window_evaluation import calculate_window_precisions, get_best_window_configuration
from preprocessing.data_processing.data_processing import DataProcessing
from preprocessing.datasets.dataset import Dataset
from preprocessing.process_results import load_max_precision_results
from alignments.dtw_attacks.dtw_attack import DtwAttack
from config import Config

from typing import Dict, List, Union
import os
import statistics
import json


cfg = Config.get()


def calculate_best_configurations(dataset: Dataset, resample_factor: int, data_processing: DataProcessing,
                                  dtw_attack: DtwAttack, k_list: List[int] = None) \
        -> Dict[str, Union[str, int, List[List[str]]]]:
    """
    Calculate the best configurations for rank-method, classes, sensors and windows
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-attack
    :param k_list: Specify k-parameters
    :return: Dictionary with best configurations
    """
    # Specify k-parameters
    if k_list is None:
        k_list = [1, 3, 5]

    # Best rank-method
    results = calculate_rank_method_precisions(dataset=dataset, resample_factor=resample_factor,
                                               data_processing=data_processing, dtw_attack=dtw_attack,
                                               k_list=k_list)
    best_rank_method = get_best_rank_method_configuration(res=results)

    # Best class
    average_results, weighted_average_results = calculate_average_class_precisions(dataset=dataset,
                                                                                   resample_factor=resample_factor,
                                                                                   data_processing=data_processing,
                                                                                   dtw_attack=dtw_attack,
                                                                                   rank_method=best_rank_method,
                                                                                   k_list=k_list)
    best_class_method = get_best_class_configuration(average_res=average_results,
                                                     weighted_average_res=weighted_average_results)

    # Best sensors
    results = calculate_sensor_precisions(dataset=dataset, resample_factor=resample_factor,
                                          data_processing=data_processing, dtw_attack=dtw_attack,
                                          rank_method=best_rank_method, average_method=best_class_method, k_list=k_list)
    best_sensors = get_best_sensor_configuration(res=results)

    # Best window
    results = calculate_window_precisions(dataset=dataset, resample_factor=resample_factor,
                                          data_processing=data_processing, dtw_attack=dtw_attack,
                                          rank_method=best_rank_method, average_method=best_class_method,
                                          sensor_combination=best_sensors, k_list=k_list)
    best_window = get_best_window_configuration(res=results)

    best_configurations = {"rank_method": best_rank_method, "class": best_class_method, "sensor": best_sensors,
                           "window": best_window}

    return best_configurations


def get_average_max_precision(dataset: Dataset, resample_factor: int, dtw_attack: DtwAttack, average_method: str,
                              window: int, k: int) -> float:
    """
    Calculate average max-precision for specified averaging method
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param dtw_attack: Specify DTW-attack
    :param average_method: Specify averaging method ("mean" or "weighted-mean")
    :param window: Specify test-window-size
    :param k: Specify k parameter
    :return: Average max-precision
    """
    non_stress_results = load_max_precision_results(dataset=dataset, resample_factor=resample_factor,
                                                    dtw_attack=dtw_attack, method="non-stress", test_window_size=window,
                                                    k=k)
    stress_results = load_max_precision_results(dataset=dataset, resample_factor=resample_factor, dtw_attack=dtw_attack,
                                                method="stress", test_window_size=window, k=k)

    result = None
    try:
        # Averaging method = "mean"
        if average_method == "mean":
            result = round(statistics.mean([non_stress_results["precision"], stress_results["precision"]]), 3)
        # Averaging method = "weighted-mean"
        else:
            class_distributions = get_class_distribution(dataset=dataset)
            result = round(non_stress_results["precision"] * class_distributions["non-stress"] +
                           stress_results["precision"] * class_distributions["stress"], 3)

    except KeyError:
        print("SW-DTW_max-precision for k=" + str(k) + " not available!")

    return result


def get_random_guess_precision(dataset: Dataset, k: int) -> float:
    """
    Calculate precision for random guess
    :param dataset: Specify dataset
    :param k: Specify k parameter
    :return: Random guess precision
    """
    amount_subjects = len(dataset.get_subject_list())
    result = round(k / amount_subjects, 3)
    return result


def calculate_optimized_precisions(dataset: Dataset, resample_factor: int, dtw_attack: DtwAttack,
                                   k_list: List[int] = None) -> Dict[int, Dict[str, float]]:
    """
    Calculate overall evaluation precision scores (DTW-results, maximum results and random guess results)
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param dtw_attack: Specify DTW-attack
    :param k_list: List with all k's
    :return: Dictionary with results
    """
    if k_list is None:
        k_list = [1, 3, 5]

    best_configuration = calculate_best_configurations(dataset=dataset, resample_factor=resample_factor,
                                                       dtw_attack=dtw_attack)
    results = calculate_window_precisions(dataset=dataset, resample_factor=resample_factor, dtw_attack=dtw_attack,
                                          rank_method=best_configuration["rank_method"],
                                          average_method=best_configuration["class"],
                                          sensor_combination=best_configuration["sensor"], k_list=k_list)

    overall_results = dict()
    for k in results:
        # Calculate DTW-Results
        overall_results.setdefault(k, {"results": results[k][best_configuration["window"]]})

        # Calculate maximum results
        overall_results[k].setdefault("max", get_average_max_precision(dataset=dataset, resample_factor=resample_factor,
                                                                       dtw_attack=dtw_attack,
                                                                       average_method=best_configuration["class"],
                                                                       window=best_configuration["window"], k=k))

        # Calculate random guess results
        overall_results[k].setdefault("random", get_random_guess_precision(dataset=dataset, k=k))

    return overall_results


def calculate_best_k_parameters(dataset: Dataset, resample_factor: int, dtw_attack: DtwAttack) -> Dict[str, int]:
    """
    Calculate k-parameters where precision@k == 1
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param dtw_attack: Specify DTW-attack
    :return: Dictionary with results
    """
    amount_subjects = len(dataset.get_subject_list())
    k_list = list(range(1, amount_subjects + 1))  # List with all possible k parameters
    results = calculate_optimized_precisions(dataset=dataset, resample_factor=resample_factor, dtw_attack=dtw_attack,
                                             k_list=k_list)
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


def get_best_sensor_weightings(dataset: Dataset, resample_factor: int, dtw_attack: DtwAttack, test_window_size: int,
                               methods: List[str] = None, k_list: List[int] = None) \
        -> Dict[str, Dict[int, List[Dict[str, float]]]]:
    """
    Calculate best sensor-weightings for specified window-size
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param dtw_attack: Specify DTW-attack
    :param test_window_size: Specify test-window-size
    :param methods: List with methods (non-stress, stress); if None: all methods are used
    :param k_list: Specify k parameters for precision@k; if None: 1, 3, 5 are used
    :return: Dictionary with weighting results
    """
    if methods is None:
        methods = dataset.get_classes()
    if k_list is None:
        k_list = [1, 3, 5]

    weightings = dict()
    for method in methods:
        weightings.setdefault(method, dict())
        for k in k_list:
            results = load_max_precision_results(dataset=dataset, resample_factor=resample_factor,
                                                 dtw_attack=dtw_attack, k=k, method=method,
                                                 test_window_size=test_window_size)
            weightings[method].setdefault(k, results["weights"])

    return weightings


def run_overall_evaluation(dataset: Dataset, resample_factor: int, dtw_attack: DtwAttack,
                           save_weightings: bool = False):
    """
    Run and save overall evaluation (DTW-results, maximum results, random guess results)
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param dtw_attack: Specify DTW-attack
    :param save_weightings: If true -> Weighting will be saved as json-file
    """
    best_configuration = calculate_best_configurations(dataset=dataset, resample_factor=resample_factor,
                                                       dtw_attack=dtw_attack)
    overall_results = calculate_optimized_precisions(dataset=dataset, resample_factor=resample_factor,
                                                     dtw_attack=dtw_attack)
    weightings = get_best_sensor_weightings(dataset=dataset, resample_factor=resample_factor, dtw_attack=dtw_attack,
                                            test_window_size=best_configuration["window"])
    best_k_parameters = calculate_best_k_parameters(dataset=dataset, resample_factor=resample_factor,
                                                    dtw_attack=dtw_attack)

    text = [create_md_precision_overall(results=overall_results, rank_method=best_configuration["rank_method"],
                                        average_method=best_configuration["class"],
                                        sensor_combination=list_to_string(input_list=best_configuration["sensor"][0]),
                                        window=best_configuration["window"], weightings=weightings,
                                        best_k_parameters=best_k_parameters)]

    # Save MD-File
    data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
    resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
    attack_path = os.path.join(resample_path, dtw_attack.get_attack_name())  # add /attack-name to path
    evaluations_path = os.path.join(attack_path, "evaluations")  # add /evaluations to path
    os.makedirs(evaluations_path, exist_ok=True)

    path_string = "SW-DTW_evaluation_overall.md"
    with open(os.path.join(evaluations_path, path_string), 'w') as outfile:
        for item in text:
            outfile.write("%s\n" % item)

    print("SW-DTW evaluation overall saved at: " + str(evaluations_path))

    # Save weightings as JSON-File
    if save_weightings:
        path_string = "SW-DTW_evaluation_weightings.json"
        with open(os.path.join(evaluations_path, path_string), "w", encoding="utf-8") as outfile:
            json.dump(weightings, outfile)

        print("SW-DTW evaluation weightings saved at: " + str(evaluations_path))
