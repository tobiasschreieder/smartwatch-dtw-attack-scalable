from alignments.dtw_attack import get_classes, get_proportions
from evaluation.metrics.calculate_precisions import calculate_precision_combinations
from evaluation.metrics.calculate_ranks import get_realistic_ranks_combinations
from evaluation.create_md_tables import create_md_precision_rank_method
from preprocessing.data_preparation import get_sensor_combinations
from preprocessing.data_preparation import get_subject_list

from typing import List, Dict
import statistics
import os
import random


MAIN_PATH = os.path.abspath(os.getcwd())
OUT_PATH = os.path.join(MAIN_PATH, "out")  # add /out to path
EVALUATIONS_PATH = os.path.join(OUT_PATH, "evaluations")  # add /evaluations to path


def calculate_rank_method_precisions(subject_ids: List = None, k_list: List[int] = None) -> Dict[int, Dict[str, float]]:
    """
    Calculate precision@k values for rank-method evaluation -> Mean over sensor-combinations, methods
    :param subject_ids: Specify subject-ids, if None: all subjects are used
    :param k_list: Specify k parameters; if None: 1, 3, 5 are used
    :return: Dictionary with precision values
    """
    sensor_combinations = get_sensor_combinations()  # Get all sensor-combinations
    classes = get_classes()  # Get all classes
    proportions_test = get_proportions()  # Get all test-proportions
    if k_list is None:
        k_list = [1, 3, 5]  # List with all k for precision@k that should be considered

    if subject_ids is None:
        subject_ids = get_subject_list()

    class_results_dict = dict()
    for method in classes:
        proportion_results_dict = dict()
        for proportion_test in proportions_test:
            results_sensor = dict()
            for k in k_list:
                # Calculate realistic ranks with rank method "rank"
                realistic_ranks_comb_rank = get_realistic_ranks_combinations(rank_method="rank",
                                                                             combinations=sensor_combinations,
                                                                             method=method,
                                                                             proportion_test=proportion_test,
                                                                             subject_ids=subject_ids)
                # Calculate precision values with rank method "rank"
                precision_comb_rank = calculate_precision_combinations(realistic_ranks_comb=realistic_ranks_comb_rank,
                                                                       k=k)

                # Calculate realistic ranks with rank method "score"
                realistic_ranks_comb_score = get_realistic_ranks_combinations(rank_method="score",
                                                                              combinations=sensor_combinations,
                                                                              method=method,
                                                                              proportion_test=proportion_test,
                                                                              subject_ids=subject_ids)
                # Calculate precision values with rank method "score"
                precision_comb_score = calculate_precision_combinations(realistic_ranks_comb=realistic_ranks_comb_score,
                                                                        k=k)

                # Calculate mean over precision-values per sensor-combinations for methods "rank" and "score"
                sensor_combined_precision_rank = statistics.mean(precision_comb_rank.values())
                sensor_combined_precision_score = statistics.mean(precision_comb_score.values())

                # Calculate mean over results from methods "rank" and "score"
                sensor_combined_precision_mean = statistics.mean([sensor_combined_precision_score,
                                                                 sensor_combined_precision_rank])
                # Save results in dictionary
                results_sensor.setdefault(k, {"rank": sensor_combined_precision_rank,
                                              "score": sensor_combined_precision_score,
                                              "mean": sensor_combined_precision_mean})

            proportion_results_dict.setdefault(proportion_test, results_sensor)

        # Calculate mean precisions over all test-proportions
        results_proportions = dict()
        for k in k_list:
            rank_precisions = list()
            score_precisions = list()
            mean_precisions = list()
            for result in proportion_results_dict.values():
                rank_precisions.append(result[k]["rank"])
                score_precisions.append(result[k]["score"])
                mean_precisions.append(result[k]["mean"])

            results_proportions.setdefault(k, {"rank": statistics.mean(rank_precisions),
                                               "score": statistics.mean(score_precisions),
                                               "mean": statistics.mean(mean_precisions)})

        class_results_dict.setdefault(method, results_proportions)

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


def calculate_best_k_parameters() -> Dict[str, int]:
    """
    Calculate k-parameters where precision@k == 1
    :return: Dictionary with results
    """
    amount_subjects = len(get_subject_list())
    k_list = list(range(1, amount_subjects + 1))  # List with all possible k parameters
    results = calculate_rank_method_precisions(k_list=k_list)
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


def run_rank_method_evaluation():
    """
    Run and save evaluation for rank-methods
    """
    results = calculate_rank_method_precisions()
    best_rank_method = get_best_rank_method_configuration(res=results)
    best_k_parameters = calculate_best_k_parameters()
    text = [create_md_precision_rank_method(results=results, best_rank_method=best_rank_method,
                                            best_k_parameters=best_k_parameters)]

    # Save MD-File
    os.makedirs(EVALUATIONS_PATH, exist_ok=True)

    path_string = "/SW-DTW_evaluation_rank_methods.md"
    with open(EVALUATIONS_PATH + path_string, 'w') as outfile:
        for item in text:
            outfile.write("%s\n" % item)

    print("SW-DTW evaluation for rank-methods saved at: " + str(EVALUATIONS_PATH))
