from alignments.dtw_attacks.dtw_attack import DtwAttack
from evaluation.metrics.calculate_precisions import calculate_precision_combinations
from evaluation.metrics.calculate_ranks import get_realistic_ranks_combinations
from evaluation.create_md_tables import create_md_precision_classes
from preprocessing.datasets.dataset import Dataset
from config import Config

from typing import Dict, List, Tuple
import pandas as pd
import statistics
import os
import random
import json


cfg = Config.get()


def get_class_distribution(dataset: Dataset) -> Dict[str, float]:
    """
    Get proportions of baseline, stress and amusement data (mean over all subjects)
    :param dataset: Specify dataset
    :return: Dictionary with proportions
    """
    data_dict = dataset.load_dataset()

    non_stress_proportions = list()
    stress_proportions = list()

    for subject in data_dict:
        data = pd.DataFrame(data_dict[subject]["label"])
        subject_length = data.shape[0]

        non_stress_length = data[data.label == 0].shape[0]
        stress_length = data[data.label == 1].shape[0]

        non_stress_proportions.append(round(non_stress_length / subject_length, 2))
        stress_proportions.append(round(stress_length / subject_length, 2))

    non_stress_proportion = round(statistics.mean(non_stress_proportions), 2)
    stress_proportion = round(statistics.mean(stress_proportions), 2)

    return {"non-stress": non_stress_proportion, "stress": stress_proportion}


def calculate_class_precisions(dataset: Dataset, resample_factor: int, dtw_attack: DtwAttack,
                               rank_method: str = "score", subject_ids: List = None, k_list: List[int] = None) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate precisions per class ("baseline", "amusement", "stress"), mean over sensors and test-proportions
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param dtw_attack: Specify DTW-attack
    :param rank_method: Specify rank-method "score" or "rank" (use beste rank-method)
    :param subject_ids: Specify subject-ids, if None: all subjects are used
    :param k_list: Specify k parameters; if None: 1, 3, 5 are used
    :return: Dictionary with results
    """
    sensor_combinations = dataset.get_sensor_combinations()  # Get all sensor-combinations
    classes = dataset.get_classes()  # Get all classes
    test_window_sizes = dtw_attack.get_windows()  # Get all test-windows

    # List with all k for precision@k that should be considered
    complete_k_list = [i for i in range(1, len(dataset.get_subject_list()) + 1)]

    if subject_ids is None:
        subject_ids = dataset.get_subject_list()

    # Specify paths
    data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
    resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
    attack_path = os.path.join(resample_path, dtw_attack.get_attack_name())  # add /attack-name to path
    evaluations_path = os.path.join(attack_path, "evaluations")  # add /evaluations to path
    results_path = os.path.join(evaluations_path, "results")  # add /results to path
    os.makedirs(results_path, exist_ok=True)
    path_string = ("SW-DTW_class-results_" + dataset.get_dataset_name() + "_" + str(resample_factor) + ".json")

    # Try to load existing results
    try:
        f = open(os.path.join(results_path, path_string), "r")
        results = json.loads(f.read())

    # Calculate results if not existing
    except FileNotFoundError:
        window_results_dict = dict()
        for test_window_size in test_window_sizes:
            results_sensor = dict()
            for k in complete_k_list:
                results_sensor.setdefault(k, dict())
                for method in classes:
                    # Calculate realistic ranks with specified rank-method
                    realistic_ranks_comb = get_realistic_ranks_combinations(dataset=dataset,
                                                                            resample_factor=resample_factor,
                                                                            dtw_attack=dtw_attack,
                                                                            rank_method=rank_method,
                                                                            combinations=sensor_combinations,
                                                                            method=method,
                                                                            test_window_size=test_window_size,
                                                                            subject_ids=subject_ids)
                    # Calculate precision values with specified rank-method
                    precision_comb = calculate_precision_combinations(dataset=dataset,
                                                                      realistic_ranks_comb=realistic_ranks_comb, k=k)

                    # Calculate mean over precision-values per sensor-combinations for specified rank-method
                    sensor_combined_precision = statistics.mean(precision_comb.values())

                    # Save results in dictionary
                    results_sensor[k].setdefault(method, sensor_combined_precision)

            window_results_dict.setdefault(test_window_size, results_sensor)

        # Calculate mean over test-proportions
        results = dict()
        for method in classes:
            for k in complete_k_list:
                results.setdefault(k, dict())
                precision_k_list = list()
                for test_window_size in window_results_dict:
                    precision_k_list.append(window_results_dict[test_window_size][k][method])
                results[k].setdefault(method, round(statistics.mean(precision_k_list), 3))

        # Save interim results as JSON-File
        with open(os.path.join(results_path, path_string), "w", encoding="utf-8") as outfile:
            json.dump(results, outfile)

        print("SW-DTW_class-results.json saved at: " + str(os.path.join(resample_path, path_string)))

    results = {int(k): v for k, v in results.items()}

    reduced_results = dict()
    if k_list is not None:
        for k in results:
            if k in k_list:
                reduced_results.setdefault(k, results[k])
        results = reduced_results

    return results


def calculate_average_class_precisions(dataset: Dataset, resample_factor: int, dtw_attack: DtwAttack,
                                       rank_method: str = "score", subject_ids: List = None, k_list: List[int] = None) \
        -> Tuple[Dict[int, float], Dict[int, int]]:
    """
    Calculate average class precision values (mean and weighted mean over classes)
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param dtw_attack: Specify DTW-attack
    :param rank_method: Specify rank-method "score" or "rank" (use beste rank-method)
    :param subject_ids: Specify subject-ids, if None: all subjects are used
    :param k_list: Specify k parameters; if None: 1, 3, 5 are used
    :return: Tuple with result dictionaries
    """
    if k_list is None:
        k_list = [1, 3, 5]  # List with all k for precision@k that should be considered

    if subject_ids is None:
        subject_ids = dataset.get_subject_list()  # List with all subject-ids

    results = calculate_class_precisions(dataset=dataset, resample_factor=resample_factor, dtw_attack=dtw_attack,
                                         rank_method=rank_method, subject_ids=subject_ids, k_list=k_list)
    class_distribution = get_class_distribution(dataset=dataset)

    average_results = dict()
    weighted_average_results = dict()
    for k in results:
        average_results.setdefault(k, 0)
        weighted_average_results.setdefault(k, 0)
        average_precision_list = list()
        weighted_average_precision = 0
        for method in results[k]:
            average_precision_list.append(results[k][method])
            weighted_average_precision += results[k][method] * class_distribution[method]
        average_results[k] = round(statistics.mean(average_precision_list), 3)
        weighted_average_results[k] = round(weighted_average_precision, 3)

    # Handle rounding errors for maximum k
    if weighted_average_results[len(results)] != 1.0:
        weighted_average_results[len(results)] = 1.0

    return average_results, weighted_average_results


def calculate_best_k_parameters(dataset: Dataset, resample_factor: int, dtw_attack: DtwAttack, rank_method: str) \
        -> Dict[str, int]:
    """
    Calculate k-parameters where precision@k == 1
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param dtw_attack: Specify DTW-attack
    :param rank_method: Specify ranking-method ("score" or "rank")
    :return: Dictionary with results
    """
    amount_subjects = len(dataset.get_subject_list())
    k_list = list(range(1, amount_subjects + 1))  # List with all possible k parameters
    results = calculate_class_precisions(dataset=dataset, resample_factor=resample_factor, dtw_attack=dtw_attack,
                                         k_list=k_list, rank_method=rank_method)
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


def calculate_best_average_k_parameters(dataset: Dataset, resample_factor: int, dtw_attack: DtwAttack,
                                        rank_method: str) -> Dict[str, int]:
    """
    Calculate k-parameters where precision@k == 1 for average-classes
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param dtw_attack: Specify DTW-attack
    :param rank_method: Specify ranking-method ("score" or "rank")
    :return: Dictionary with results
    """
    subject_ids = dataset.get_subject_list()
    k_list = list(range(1, len(subject_ids) + 1))  # List with all possible k parameters
    average_results, weighted_average_results = calculate_average_class_precisions(dataset=dataset,
                                                                                   resample_factor=resample_factor,
                                                                                   dtw_attack=dtw_attack,
                                                                                   rank_method=rank_method,
                                                                                   k_list=k_list,
                                                                                   subject_ids=subject_ids)
    best_average_k_parameters = dict()
    set_k = False
    for k, value in average_results.items():
        if set_k is False and value == 1.0:
            best_average_k_parameters.setdefault("mean", k)
            set_k = True

    set_k = False
    for k, value in weighted_average_results.items():
        if set_k is False and value == 1.0:
            best_average_k_parameters.setdefault("weighted-mean", k)
            set_k = True

    return best_average_k_parameters


def get_best_class_configuration(average_res: Dict[int, float], weighted_average_res: Dict[int, float]) -> str:
    """
    Calculate best class configuration "mean" or "weighted-mean" from given results
    :param average_res: Dictionary with averaged results
    :param weighted_average_res: Dictionary with weighted averaged results
    :return: String with best class-configuration
    """
    best_class_method = str()
    for dec_k in average_res:
        if average_res[dec_k] > weighted_average_res[dec_k]:
            best_class_method = "mean"
            break
        elif average_res[dec_k] < weighted_average_res[dec_k]:
            best_class_method = "weighted-mean"
            break
        else:
            best_class_method = random.choice(["mean", "weighted-mean"])

    return best_class_method


def run_class_evaluation(dataset: Dataset, resample_factor: int, dtw_attack: DtwAttack, rank_method: str = "score",
                         k_list: List[int] = None):
    """
    Run and save evaluation for classes
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param dtw_attack: Specify DTW-attack
    :param rank_method: Specify rank-method "score" or "rank" (use best performing method)
    :param k_list: Specify k-parameters
    """
    # Specify k-parameters
    if k_list is None:
        k_list = [1, 3, 5]

    results = calculate_class_precisions(dataset=dataset, resample_factor=resample_factor, dtw_attack=dtw_attack,
                                         rank_method=rank_method, k_list=k_list)
    average_results, weighted_average_results = calculate_average_class_precisions(dataset=dataset,
                                                                                   resample_factor=resample_factor,
                                                                                   dtw_attack=dtw_attack,
                                                                                   rank_method=rank_method)
    best_class_method = get_best_class_configuration(average_res=average_results,
                                                     weighted_average_res=weighted_average_results)

    best_k_parameters = calculate_best_k_parameters(dataset=dataset, resample_factor=resample_factor,
                                                    dtw_attack=dtw_attack, rank_method=rank_method)
    best_average_k_parameters = calculate_best_average_k_parameters(dataset=dataset, resample_factor=resample_factor,
                                                                    dtw_attack=dtw_attack, rank_method=rank_method)

    text = [create_md_precision_classes(rank_method=rank_method, results=results, average_results=average_results,
                                        weighted_average_results=weighted_average_results,
                                        best_class_method=best_class_method, best_k_parameters=best_k_parameters,
                                        best_average_k_parameters=best_average_k_parameters)]

    # Save MD-File
    data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
    resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
    attack_path = os.path.join(resample_path, dtw_attack.get_attack_name())  # add /attack-name to path
    evaluations_path = os.path.join(attack_path, "evaluations")  # add /evaluations to path
    os.makedirs(evaluations_path, exist_ok=True)

    path_string = "SW-DTW_evaluation_classes.md"
    with open(os.path.join(evaluations_path, path_string), 'w') as outfile:
        for item in text:
            outfile.write("%s\n" % item)

    print("SW-DTW evaluation for classes saved at: " + str(evaluations_path))
