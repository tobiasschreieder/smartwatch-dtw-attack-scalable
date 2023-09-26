from alignments.dtw_attack import get_classes, get_proportions
from evaluation.metrics.calculate_precisions import calculate_precision_combinations
from evaluation.metrics.calculate_ranks import get_realistic_ranks_combinations
from evaluation.create_md_tables import create_md_precision_windows
from evaluation.optimization.class_evaluation import get_class_distribution
from evaluation.optimization.sensor_evaluation import list_to_string
from preprocessing.data_preparation import get_subject_list

from typing import Dict, List
import matplotlib.pyplot as plt
import statistics
import os
import random

MAIN_PATH = os.path.abspath(os.getcwd())
OUT_PATH = os.path.join(MAIN_PATH, "out")  # add /out to path
EVALUATIONS_PATH = os.path.join(OUT_PATH, "evaluations")  # add /evaluations to path


def calculate_window_precisions(rank_method: str = "score", average_method: str = "weighted-mean",
                                sensor_combination=None, subject_ids: List = None, k_list: List[int] = None) \
        -> Dict[int, Dict[float, float]]:
    """
    Calculate precisions per test-proportion ("baseline", "amusement", "stress"), mean over sensors and test-proportions
    :param rank_method: Specify rank-method "score" or "rank" (use beste rank-method)
    :param average_method: Specify averaging-method "mean" or "weighted-mean" (Choose best one)
    :param sensor_combination: Specify sensor-combination e.g. [["bvp", "acc", "temp"]] (Choose best on)
    :param subject_ids: Specify subject-ids, if None: all subjects are used
    :param k_list: Specify k parameters; if None: 1, 3, 5 are used
    :return: Dictionary with results
    """

    classes = get_classes()  # Get all classes
    proportions_test = get_proportions()  # Get all test-proportions
    if k_list is None:
        k_list = [1, 3, 5]  # List with all k for precision@k that should be considered
    class_distributions = get_class_distribution()  # Get class distributions

    if subject_ids is None:
        subject_ids = get_subject_list()
    if sensor_combination is None:
        sensor_combination = [["bvp", "eda", "acc", "temp"]]

    proportion_results_dict = dict()
    for proportion_test in proportions_test:
        results_class = dict()
        for k in k_list:
            results_class.setdefault(k, dict())
            for method in classes:
                # Calculate realistic ranks with specified rank-method
                realistic_ranks_comb = get_realistic_ranks_combinations(rank_method=rank_method,
                                                                        combinations=sensor_combination,
                                                                        method=method,
                                                                        proportion_test=proportion_test,
                                                                        subject_ids=subject_ids)
                # Calculate precision values with rank-method
                precision_comb = calculate_precision_combinations(realistic_ranks_comb=realistic_ranks_comb, k=k)

                # Save results in dictionary
                results_class[k].setdefault(method, statistics.mean(precision_comb.values()))

        proportion_results_dict.setdefault(proportion_test, results_class)

    # Calculate mean over classes
    results = dict()
    for proportion in proportions_test:
        for k in k_list:
            results.setdefault(k, dict())

            # averaging method "mean" -> unweighted mean
            if average_method == "mean":
                precision_class_list = list()
                for method in classes:
                    precision_class_list.append(proportion_results_dict[proportion][k][method])
                results[k].setdefault(proportion, round(statistics.mean(precision_class_list), 3))

            # averaging method "weighted mean" -> weighted mean
            else:
                precision_class_list = list()
                for method in classes:
                    precision_class_list.append(proportion_results_dict[proportion][k][method] *
                                                class_distributions[method])
                results[k].setdefault(proportion, round(sum(precision_class_list), 3))

    return results


def calculate_best_k_parameters(rank_method: str, average_method: str, sensor_combination: List[List[str]]) \
        -> Dict[float, int]:
    """
    Calculate k-parameters where precision@k == 1
    :param rank_method: Specify ranking-method ("score" or "rank")
    :param average_method: Specify class averaging-method ("mean" or "weighted-mean)
    :param sensor_combination: Specify sensor-combination e.g. [["bvp", "acc", "temp"]] (Choose best on)
    :return: Dictionary with results
    """
    amount_subjects = len(get_subject_list())
    k_list = list(range(1, amount_subjects + 1))  # List with all possible k parameters
    results = calculate_window_precisions(k_list=k_list, rank_method=rank_method, average_method=average_method,
                                          sensor_combination=sensor_combination)
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


def plot_window_precisions(results: Dict[int, Dict[float, float]], k_list: List[int]):
    """
    Plot precision@k over window sizes
    :param results: Results with precision values per class
    :param k_list: List with all k for precision@k that should be considered
    """
    plt.title(label="Precision@k over window-sizes", loc="center")
    plt.ylabel('precision@k')
    plt.xlabel('window-size')
    for k in k_list:
        data = results[k]
        x, y = zip(*sorted(data.items()))
        plt.plot(x, y, label="k=" + str(k))
    plt.legend()

    try:
        plt.savefig(fname=EVALUATIONS_PATH + "/SW-DTW_evaluation_windows.pdf", format="pdf")

    except FileNotFoundError:
        print("FileNotFoundError: Invalid directory structure!")

    plt.close()


def get_best_window_configuration(res: Dict[int, Dict[float, float]]) -> float:
    """
    Calculate best window configuration (test-proportion) from given results
    :param res: Dictionary with results
    :return: String with best window-size
    """
    def get_best_window(windows: Dict[float, float]) -> List[float]:
        """
        Get window with maximum precision score
        :param windows: Dictionary with all windows and precision scores
        :return: List with best windows
        """
        max_value = max(windows.values())
        max_windows = [k for k, v in windows.items() if v == max_value]
        return max_windows

    best_window = str()
    best_windows = list()
    adjusted_res = res.copy()
    for k in res:
        if len(best_windows) == 0:
            best_windows = get_best_window(windows=res[k])
        else:
            adjusted_res[k] = {key: adjusted_res[k][key] for key in best_windows}
            best_windows = get_best_window(windows=adjusted_res[k])

        if len(best_windows) == 1:
            best_window = best_windows[0]
            break

    if len(best_windows) > 1:
        best_window = random.choice(best_windows)

    return best_window


def run_window_evaluation(rank_method: str = "score", average_method: str = "weighted-mean",
                          sensor_combination=None):
    """
    Run and save evaluation for sensor-combinations
    :param rank_method: Specify rank-method "score" or "rank" (use best performing method)
    :param average_method: Specify averaging-method "mean" or "weighted-mean" (use best performing method)
    :param sensor_combination: Specify sensor-combination e.g. [["acc", "temp"]] (Choose best on)
    """
    if sensor_combination is None:
        sensor_combination = [["bvp", "eda", "acc", "temp"]]

    results = calculate_window_precisions(rank_method=rank_method, average_method=average_method,
                                          sensor_combination=sensor_combination)
    best_window = get_best_window_configuration(res=results)
    best_k_parameters = calculate_best_k_parameters(rank_method=rank_method, average_method=average_method,
                                                    sensor_combination=sensor_combination)

    text = [create_md_precision_windows(rank_method=rank_method, average_method=average_method, results=results,
                                        sensor_combination=list_to_string(input_list=sensor_combination[0]),
                                        best_window=best_window, best_k_parameters=best_k_parameters)]

    # Save MD-File
    os.makedirs(EVALUATIONS_PATH, exist_ok=True)

    path_string = "/SW-DTW_evaluation_windows.md"
    with open(EVALUATIONS_PATH + path_string, 'w') as outfile:
        for item in text:
            outfile.write("%s\n" % item)

    print("SW-DTW evaluation for windows saved at: " + str(EVALUATIONS_PATH))

    # Plot precision@k over window sizes
    plot_window_precisions(results=results, k_list=[1, 3, 5])
