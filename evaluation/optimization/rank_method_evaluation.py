from alignments.dtw_attack import get_windows
from evaluation.metrics.calculate_precisions import calculate_precision_combinations
from evaluation.metrics.calculate_ranks import get_realistic_ranks_combinations
from evaluation.create_md_tables import create_md_precision_rank_method
from preprocessing.datasets.dataset import Dataset
from config import Config

from typing import List, Dict
import statistics
import os
import random
import json


cfg = Config.get()


def calculate_rank_method_precisions(dataset: Dataset, resample_factor: int, subject_ids: List = None,
                                     k_list: List[int] = None) -> Dict[int, Dict[str, float]]:
    """
    Calculate precision@k values for rank-method evaluation -> Mean over sensor-combinations, methods
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param subject_ids: Specify subject-ids, if None: all subjects are used
    :param k_list: Specify k parameters; if None: 1, 3, 5 are used
    :return: Dictionary with precision values
    """
    sensor_combinations = dataset.get_sensor_combinations()  # Get all sensor-combinations
    classes = dataset.get_classes()  # Get all classes
    test_window_sizes = get_windows()  # Get all test-window-sizes
    if k_list is None:
        k_list = [1, 3, 5]  # List with all k for precision@k that should be considered

    if subject_ids is None:
        subject_ids = dataset.get_subject_list()

    class_results_dict = dict()
    for method in classes:
        window_results_dict = dict()
        for test_window_size in test_window_sizes:
            results_sensor = dict()
            for k in k_list:
                # Calculate realistic ranks with rank method "rank"
                realistic_ranks_comb_rank = get_realistic_ranks_combinations(dataset=dataset,
                                                                             resample_factor=resample_factor,
                                                                             rank_method="rank",
                                                                             combinations=sensor_combinations,
                                                                             method=method,
                                                                             test_window_size=test_window_size,
                                                                             subject_ids=subject_ids)
                # Calculate precision values with rank method "rank"
                precision_comb_rank = calculate_precision_combinations(dataset=dataset,
                                                                       realistic_ranks_comb=
                                                                       realistic_ranks_comb_rank,
                                                                       k=k)

                # Calculate realistic ranks with rank method "score"
                realistic_ranks_comb_score = get_realistic_ranks_combinations(dataset=dataset,
                                                                              resample_factor=resample_factor,
                                                                              rank_method="score",
                                                                              combinations=sensor_combinations,
                                                                              method=method,
                                                                              test_window_size=test_window_size,
                                                                              subject_ids=subject_ids)
                # Calculate precision values with rank method "score"
                precision_comb_score = calculate_precision_combinations(dataset=dataset,
                                                                        realistic_ranks_comb=
                                                                        realistic_ranks_comb_score,
                                                                        k=k)

                sensor_combined_precision_rank = statistics.mean(precision_comb_rank.values())
                sensor_combined_precision_score = statistics.mean(precision_comb_score.values())

                # Calculate mean over results from methods "rank" and "score"
                sensor_combined_precision_mean = statistics.mean([sensor_combined_precision_score,
                                                                 sensor_combined_precision_rank])
                # Save results in dictionary
                results_sensor.setdefault(k, {"rank": sensor_combined_precision_rank,
                                              "score": sensor_combined_precision_score,
                                              "mean": sensor_combined_precision_mean})

            window_results_dict.setdefault(test_window_size, results_sensor)

        # Calculate mean precisions over all test-window-sizes
        results_windows = dict()
        for k in k_list:
            rank_precisions = list()
            score_precisions = list()
            mean_precisions = list()
            for result in window_results_dict.values():
                rank_precisions.append(result[k]["rank"])
                score_precisions.append(result[k]["score"])
                mean_precisions.append(result[k]["mean"])

            results_windows.setdefault(k, {"rank": statistics.mean(rank_precisions),
                                           "score": statistics.mean(score_precisions),
                                           "mean": statistics.mean(mean_precisions)})

        class_results_dict.setdefault(method, results_windows)

    # Calculate mean precisions over all classes
    results = dict()
    for k in k_list:
        rank_precisions = list()
        score_precisions = list()
        mean_precisions = list()
        for result in class_results_dict.values():
            rank_precisions.append(result[k]["rank"])
            score_precisions.append(result[k]["score"])
            mean_precisions.append(result[k]["mean"])

        results.setdefault(k, {"rank": round(statistics.mean(rank_precisions), 3),
                               "score": round(statistics.mean(score_precisions), 3),
                               "mean": round(statistics.mean(mean_precisions), 3)})

    return results


def calculate_best_k_parameters(dataset: Dataset, resample_factor: int) -> Dict[str, int]:
    """
    Calculate k-parameters where precision@k == 1
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :return: Dictionary with results
    """
    amount_subjects = len(dataset.get_subject_list())
    k_list = list(range(1, amount_subjects + 1))  # List with all possible k parameters
    results = calculate_rank_method_precisions(dataset=dataset, resample_factor=resample_factor, k_list=k_list)
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


def get_best_rank_method_configuration(res: Dict[int, Dict[str, float]]) -> str:
    """
    Calculate best ranking-method configuration "score" or "rank" from given results
    :param res: Dictionary with results
    :return: String with best ranking-method
    """
    best_rank_method = str()
    for dec_k in res:
        if res[dec_k]["score"] > res[dec_k]["rank"]:
            best_rank_method = "score"
            break
        elif res[dec_k]["score"] < res[dec_k]["rank"]:
            best_rank_method = "rank"
            break
        else:
            best_rank_method = random.choice(["rank", "score"])

    return best_rank_method


def run_rank_method_evaluation(dataset: Dataset, resample_factor: int):
    """
    Run and save evaluation for rank-methods
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    """
    results = calculate_rank_method_precisions(dataset=dataset, resample_factor=resample_factor)
    best_rank_method = get_best_rank_method_configuration(res=results)
    best_k_parameters = calculate_best_k_parameters(dataset=dataset, resample_factor=resample_factor)
    text = [create_md_precision_rank_method(results=results, best_rank_method=best_rank_method,
                                            best_k_parameters=best_k_parameters)]

    # Save MD-File
    data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
    resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
    evaluations_path = os.path.join(resample_path, "evaluations")  # add /evaluations to path
    os.makedirs(evaluations_path, exist_ok=True)

    path_string = "SW-DTW_evaluation_rank_methods.md"
    with open(os.path.join(evaluations_path, path_string), 'w') as outfile:
        for item in text:
            outfile.write("%s\n" % item)

    print("SW-DTW evaluation for rank-methods saved at: " + str(evaluations_path))
