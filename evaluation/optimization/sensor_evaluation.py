from alignments.dtw_attack import get_classes, get_proportions
from evaluation.metrics.calculate_precisions import calculate_precision_combinations
from evaluation.metrics.calculate_ranks import get_realistic_ranks_combinations
from evaluation.optimization.class_evaluation import get_class_distribution
from evaluation.create_md_tables import create_md_precision_sensors
from preprocessing.data_preparation import get_sensor_combinations, get_subject_list

from typing import List, Dict, Union
import os
import statistics
import random


MAIN_PATH = os.path.abspath(os.getcwd())
OUT_PATH = os.path.join(MAIN_PATH, "out")  # add /out to path
EVALUATIONS_PATH = os.path.join(OUT_PATH, "evaluations")  # add /evaluations to path


def list_to_string(input_list: List[str]) -> str:
    """
    Get string for possible sensor-name combinations e.g. ["acc", "temp"] -> "acc+temp"
    :param input_list: List to be transformed
    :return: String with text
    """
    text = str()
    for i in input_list:
        text += i
        text += "+"
    text = text[:-1]
    return text


def string_to_list(input_string: str) -> List[List[str]]:
    """
    Get list from string with sensor-name combinations e.g. "acc+temp" -> [["acc", "temp"]]
    :param input_string: String to be transformed
    :return: List with sensor-combinations
    """
    sensor_list = input_string.split(sep="+")
    return [sensor_list]


def calculate_sensor_precisions(rank_method: str = "score", average_method: str = "weighted-mean",
                                subject_ids: List[int] = None, k_list: List[int] = None) -> Dict[int, Dict[str, float]]:
    """
    Calculate precisions per sensor-combination, mean over classes and test-proportions
    :param rank_method: Specify rank-method "score" or "rank" (Choose best one)
    :param average_method: Specify averaging-method "mean" or "weighted-mean" (Choose best one)
    :param subject_ids: Specify subject-ids, if None: all subjects are used
    :param k_list: Specify k parameters; if None: 1, 3, 5 are used
    :return: Dictionary with results
    """
    sensor_combinations = get_sensor_combinations()  # Get all sensor-combinations
    classes = get_classes()  # Get all classes
    proportions_test = get_proportions()  # Get all test-proportions
    if k_list is None:
        k_list = [1, 3, 5]  # List with all k for precision@k that should be considered
    class_distributions = get_class_distribution()  # Get class distributions

    if subject_ids is None:
        subject_ids = get_subject_list()

    proportion_results_dict = dict()
    for proportion_test in proportions_test:
        results_class = dict()
        for k in k_list:
            results_class.setdefault(k, dict())
            for method in classes:
                # Calculate realistic ranks with specified rank-method
                realistic_ranks_comb = get_realistic_ranks_combinations(rank_method=rank_method,
                                                                        combinations=sensor_combinations,
                                                                        method=method,
                                                                        proportion_test=proportion_test,
                                                                        subject_ids=subject_ids)
                # Calculate precision values with rank-method
                precision_comb = calculate_precision_combinations(realistic_ranks_comb=realistic_ranks_comb, k=k)

                # Save results in dictionary
                results_class[k].setdefault(method, precision_comb)

        proportion_results_dict.setdefault(proportion_test, results_class)

    # Calculate mean over test-proportions and classes
    results = dict()
    for sensor in sensor_combinations:
        sensor = list_to_string(input_list=sensor)
        for k in k_list:
            results.setdefault(k, dict())
            precision_k_list = list()
            for proportion in proportion_results_dict:
                precision_class_list = list()

                # averaging method "mean" -> unweighted mean
                if average_method == "mean":
                    for method in classes:
                        precision_class_list.append(proportion_results_dict[proportion][k][method][sensor])
                    precision_k_list.append(statistics.mean(precision_class_list))

                # averaging method "weighted mean" -> weighted mean
                else:
                    for method in classes:
                        precision_class_list.append(proportion_results_dict[proportion][k][method][sensor] *
                                                    class_distributions[method])
                    precision_k_list.append(sum(precision_class_list))

            results[k].setdefault(sensor, round(statistics.mean(precision_k_list), 3))

    return results


def calculate_best_k_parameters(rank_method: str, average_method: str) -> Dict[str, int]:
    """
    Calculate k-parameters where precision@k == 1
    :param rank_method: Specify ranking-method ("score" or "rank")
    :param average_method: Specify class averaging-method ("mean" or "weighted-mean)
    :return: Dictionary with results
    """
    amount_subjects = len(get_subject_list())
    k_list = list(range(1, amount_subjects + 1))  # List with all possible k parameters
    results = calculate_sensor_precisions(k_list=k_list, rank_method=rank_method, average_method=average_method)
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


def get_best_sensor_configuration(res: Dict[int, Dict[str, float]], printable_version: bool = False) \
        -> Union[str, List[List[str]]]:
    """
    Calculate best sensor-combinations from given results
    :param res: Dictionary with sensor precision results
    :param printable_version: Set True if e.g. "acc+temp" instead of [["acc", "temp"]] should be returned
    :return: String or List with best sensor-combination
    """
    def get_best_sensor(sensors: Dict[str, float]) -> List[str]:
        """
        Get sensor-combinations with maximum precision score
        :param sensors: Dictionary with all sensor-combinations and precision scores
        :return: List with best sensor-combinations
        """
        max_value = max(sensors.values())
        max_sensors = [k for k, v in sensors.items() if v == max_value]
        return max_sensors

    best_sensor = str()
    best_sensors = list()
    adjusted_res = res.copy()
    for k in res:
        if len(best_sensors) == 0:
            best_sensors = get_best_sensor(sensors=res[k])
        else:
            adjusted_res[k] = {key: adjusted_res[k][key] for key in best_sensors}
            best_sensors = get_best_sensor(sensors=adjusted_res[k])

        if len(best_sensors) == 1:
            best_sensor = best_sensors[0]
            break

    if len(best_sensors) > 1:
        best_sensor = random.choice(best_sensors)

    if not printable_version:
        best_sensor = string_to_list(input_string=best_sensor)

    return best_sensor


def run_sensor_evaluation(rank_method: str = "score", average_method: str = "weighted-mean"):
    """
    Run and save evaluation for sensor-combinations
    :param rank_method: Specify rank-method "score" or "rank" (use best performing method)
    :param average_method: Specify averaging-method "mean" or "weighted-mean" (use best performing method)
    """
    results = calculate_sensor_precisions(rank_method=rank_method, average_method=average_method)
    best_sensors = get_best_sensor_configuration(res=results, printable_version=True)
    best_k_parameters = calculate_best_k_parameters(rank_method=rank_method, average_method=average_method)

    text = [create_md_precision_sensors(rank_method=rank_method, average_method=average_method, results=results,
                                        best_sensors=best_sensors, best_k_parameters=best_k_parameters)]

    # Save MD-File
    os.makedirs(EVALUATIONS_PATH, exist_ok=True)

    path_string = "/SW-DTW_evaluation_sensors.md"
    with open(EVALUATIONS_PATH + path_string, 'w') as outfile:
        for item in text:
            outfile.write("%s\n" % item)

    print("SW-DTW evaluation for sensors saved at: " + str(EVALUATIONS_PATH))