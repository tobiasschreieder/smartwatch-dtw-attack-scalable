from alignments.dtw_attacks.dtw_attack import DtwAttack
from evaluation.metrics.calculate_precisions import calculate_precision_combinations
from evaluation.metrics.calculate_ranks import get_realistic_ranks_combinations
from evaluation.create_md_tables import create_md_precision_windows
from evaluation.optimization.class_evaluation import get_class_distribution
from evaluation.optimization.sensor_evaluation import list_to_string
from preprocessing.data_processing.data_processing import DataProcessing
from preprocessing.datasets.dataset import Dataset
from config import Config

from typing import Dict, List
import matplotlib.pyplot as plt
import statistics
import os
import random
import json


cfg = Config.get()


def calculate_window_precisions(dataset: Dataset, resample_factor: int, data_processing: DataProcessing,
                                dtw_attack: DtwAttack, rank_method: str = "score",
                                average_method: str = "weighted-mean", sensor_combination=None,
                                subject_ids: List = None, k_list: List[int] = None) -> Dict[int, Dict[int, float]]:
    """
    Calculate precisions per test-window-size, mean over sensors and test-window-size
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-attack
    :param rank_method: Specify rank-method "score" or "rank" (use beste rank-method)
    :param average_method: Specify averaging-method "mean" or "weighted-mean" (Choose best one)
    :param sensor_combination: Specify sensor-combination e.g. [["bvp", "acc", "temp"]] (Choose best on)
    :param subject_ids: Specify subject-ids, if None: all subjects are used
    :param k_list: Specify k parameters; if None: 1, 3, 5 are used
    :return: Dictionary with results
    """
    classes = dataset.get_classes()  # Get all classes
    windows_test = dtw_attack.get_windows()  # Get all test-windows

    # List with all k for precision@k that should be considered
    complete_k_list = [i for i in range(1, len(dataset.get_subject_list()) + 1)]
    # Get class distributions
    class_distributions = get_class_distribution(dataset=dataset, resample_factor=resample_factor,
                                                 data_processing=data_processing)

    if subject_ids is None:
        subject_ids = dataset.get_subject_list()
    if sensor_combination is None:
        sensor_combination = [["bvp", "eda", "acc", "temp"]]

    # Specify paths
    data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
    resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
    attack_path = os.path.join(resample_path, dtw_attack.get_attack_name())  # add /attack-name to path
    processing_path = os.path.join(attack_path, data_processing.name)  # add /data-processing to path
    evaluations_path = os.path.join(processing_path, "evaluations")  # add /evaluations to path
    results_path = os.path.join(evaluations_path, "results")  # add /results to path
    os.makedirs(results_path, exist_ok=True)
    path_string = ("SW-DTW_window-results_" + dataset.get_dataset_name() + "_" + str(resample_factor) + ".json")

    # Try to load existing results
    try:
        f = open(os.path.join(results_path, path_string), "r")
        results = json.loads(f.read())
        results = {int(k): v for k, v in results.items()}
        for k in results:
            results[k] = {int(k): v for k, v in results[k].items()}

    # Calculate results if not existing
    except FileNotFoundError:
        window_results_dict = dict()
        for test_window_size in windows_test:
            results_class = dict()
            for k in complete_k_list:
                results_class.setdefault(k, dict())
                for method in classes:
                    # Calculate realistic ranks with specified rank-method
                    realistic_ranks_comb = get_realistic_ranks_combinations(dataset=dataset,
                                                                            resample_factor=resample_factor,
                                                                            data_processing=data_processing,
                                                                            dtw_attack=dtw_attack,
                                                                            rank_method=rank_method,
                                                                            combinations=sensor_combination,
                                                                            method=method,
                                                                            test_window_size=test_window_size,
                                                                            subject_ids=subject_ids)
                    # Calculate precision values with rank-method
                    precision_comb = calculate_precision_combinations(dataset=dataset,
                                                                      realistic_ranks_comb=realistic_ranks_comb, k=k)

                    # Save results in dictionary
                    results_class[k].setdefault(method, statistics.mean(precision_comb.values()))

            window_results_dict.setdefault(test_window_size, results_class)

        # Calculate mean over classes
        results = dict()
        for test_window_size in windows_test:
            for k in complete_k_list:
                results.setdefault(k, dict())

                # averaging method "mean" -> unweighted mean
                if average_method == "mean":
                    precision_class_list = list()
                    for method in classes:
                        precision_class_list.append(window_results_dict[test_window_size][k][method])
                    results[k].setdefault(test_window_size, round(statistics.mean(precision_class_list), 3))

                # averaging method "weighted mean" -> weighted mean
                else:
                    precision_class_list = list()
                    for method in classes:
                        precision_class_list.append(window_results_dict[test_window_size][k][method] *
                                                    class_distributions[method])
                    results[k].setdefault(test_window_size, round(sum(precision_class_list), 3))

        # Save interim results as JSON-File
        with open(os.path.join(results_path, path_string), "w", encoding="utf-8") as outfile:
            json.dump(results, outfile)

        print("SW-DTW_window-results.json saved at: " + str(os.path.join(resample_path, path_string)))

    results = {int(k): v for k, v in results.items()}

    reduced_results = dict()
    if k_list is not None:
        for k in results:
            if k in k_list:
                reduced_results.setdefault(k, results[k])
        results = reduced_results

    return results


def calculate_best_k_parameters(dataset: Dataset, resample_factor: int, data_processing: DataProcessing,
                                dtw_attack: DtwAttack, rank_method: str, average_method: str,
                                sensor_combination: List[List[str]]) -> Dict[float, int]:
    """
    Calculate k-parameters where precision@k == 1
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-attack
    :param rank_method: Specify ranking-method ("score" or "rank")
    :param average_method: Specify class averaging-method ("mean" or "weighted-mean)
    :param sensor_combination: Specify sensor-combination e.g. [["bvp", "acc", "temp"]] (Choose best on)
    :return: Dictionary with results
    """
    amount_subjects = len(dataset.get_subject_list())
    k_list = list(range(1, amount_subjects + 1))  # List with all possible k parameters
    results = calculate_window_precisions(dataset=dataset, resample_factor=resample_factor,
                                          data_processing=data_processing, dtw_attack=dtw_attack, k_list=k_list,
                                          rank_method=rank_method, average_method=average_method,
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


def plot_window_precisions(results: Dict[int, Dict[float, float]], k_list: List[int], evaluations_path: os.path):
    """
    Plot precision@k over window sizes
    :param results: Results with precision values per class
    :param k_list: List with all k for precision@k that should be considered
    :param evaluations_path: Specify path to save plot
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
        plt.savefig(fname=os.path.join(evaluations_path, "SW-DTW_evaluation_windows.pdf"), format="pdf")

    except FileNotFoundError:
        print("FileNotFoundError: Invalid directory structure!")

    plt.close()


def get_best_window_configuration(res: Dict[int, Dict[int, float]]) -> int:
    """
    Calculate best window configuration (test-window-size) from given results
    :param res: Dictionary with results
    :return: String with best window-size
    """
    def get_best_window(windows: Dict[int, float]) -> List[int]:
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


def run_window_evaluation(dataset: Dataset, resample_factor: int, data_processing: DataProcessing,
                          dtw_attack: DtwAttack, rank_method: str = "score",  average_method: str = "weighted-mean",
                          sensor_combination=None, k_list: List[int] = None):
    """
    Run and save evaluation for sensor-combinations
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param dtw_attack: Specify DTW-attack
    :param rank_method: Specify rank-method "score" or "rank" (use best performing method)
    :param average_method: Specify averaging-method "mean" or "weighted-mean" (use best performing method)
    :param sensor_combination: Specify sensor-combination e.g. [["acc", "temp"]] (Choose best on)
    :param k_list: Specify k-parameters
    """
    # Specify k-parameters
    if k_list is None:
        k_list = [1, 3, 5]

    if sensor_combination is None:
        sensor_combination = [["bvp", "eda", "acc", "temp"]]

    results = calculate_window_precisions(dataset=dataset, resample_factor=resample_factor,
                                          data_processing=data_processing, dtw_attack=dtw_attack,
                                          rank_method=rank_method, average_method=average_method,
                                          sensor_combination=sensor_combination, k_list=k_list)
    best_window = get_best_window_configuration(res=results)
    best_k_parameters = calculate_best_k_parameters(dataset=dataset, resample_factor=resample_factor,
                                                    data_processing=data_processing, dtw_attack=dtw_attack,
                                                    rank_method=rank_method, average_method=average_method,
                                                    sensor_combination=sensor_combination)

    text = [create_md_precision_windows(rank_method=rank_method, average_method=average_method, results=results,
                                        sensor_combination=list_to_string(input_list=sensor_combination[0]),
                                        best_window=best_window, best_k_parameters=best_k_parameters)]

    # Save MD-File
    data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
    resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
    attack_path = os.path.join(resample_path, dtw_attack.get_attack_name())  # add /attack-name to path
    processing_path = os.path.join(attack_path, data_processing.name)  # add /data-processing to path
    evaluations_path = os.path.join(processing_path, "evaluations")  # add /evaluations to path
    os.makedirs(evaluations_path, exist_ok=True)

    path_string = "SW-DTW_evaluation_windows.md"
    with open(os.path.join(evaluations_path, path_string), 'w') as outfile:
        for item in text:
            outfile.write("%s\n" % item)

    print("SW-DTW evaluation for windows saved at: " + str(evaluations_path))

    # Plot precision@k over window sizes
    plot_window_precisions(results=results, k_list=[1, 3, 5], evaluations_path=evaluations_path)
