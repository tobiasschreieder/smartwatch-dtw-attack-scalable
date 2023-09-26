from alignments.dtw_attack import get_classes, get_proportions
from evaluation.metrics.calculate_precisions import calculate_max_precision
from evaluation.metrics.calculate_ranks import run_calculate_ranks, get_realistic_ranks
from evaluation.create_md_tables import create_md_distances, create_md_ranks, create_md_precision_combinations
from evaluation.optimization.class_evaluation import run_class_evaluation
from evaluation.optimization.overall_evaluation import calculate_best_configurations
from evaluation.optimization.rank_method_evaluation import run_rank_method_evaluation
from evaluation.optimization.sensor_evaluation import run_sensor_evaluation
from evaluation.optimization.window_evaluation import run_window_evaluation
from preprocessing.data_preparation import get_subject_list
from preprocessing.process_results import load_results

from typing import List
import os
import matplotlib.pyplot as plt


MAIN_PATH = os.path.abspath(os.getcwd())
OUT_PATH = os.path.join(MAIN_PATH, "out")  # add /out to path
SUBJECT_PLOT_PATH = os.path.join(OUT_PATH, "subject-plots")  # add /subject-plots to path
PRECISION_PATH = os.path.join(OUT_PATH, "precision")  # add /precision to path
EVALUATIONS_PATH = os.path.join(OUT_PATH, "evaluations")  # add /evaluations to path


def run_calculate_max_precision(k_list: List[int], methods: List = None, proportions_test: List = None,
                                step_width: float = 0.1):
    """
    Run calculations of maximum-precisions for specified k's, methods and test-proportions
    :param k_list: List with all k parameter
    :param methods: List with all methods ("baseline", "amusement", "stress")
    :param proportions_test: List with all test_proportions
    :param step_width: Specify step-width for weights
    """
    if methods is None:
        methods = get_classes()
    if proportions_test is None:
        proportions_test = get_proportions()

    for k in k_list:
        for method in methods:
            for proportion_test in proportions_test:
                calculate_max_precision(k=k, step_width=step_width, method=method, proportion_test=proportion_test)


def plot_realistic_ranks(path: os.path, method: str, proportion_test: float):
    """
    Plot and save realistic-rank-plot
    :param path: Path to save boxplot
    :param method: Specify method of results ("baseline", "amusement", "stress")
    :param proportion_test: Specify test-proportion
    """
    real_ranks_1 = get_realistic_ranks(rank_method="rank", method=method, proportion_test=proportion_test)
    real_ranks_2 = get_realistic_ranks(rank_method="score", method=method, proportion_test=proportion_test)

    real_ranks = [real_ranks_1, real_ranks_2]
    fig1, ax1 = plt.subplots()
    ax1.set_title('Realistic-Rank boxplot')
    plt.ylabel("Ranks")
    plt.xlabel("Methods: 1 = rank | 2 = score")
    ax1.boxplot(real_ranks)
    plt.savefig(fname=path, format="pdf")
    plt.close()


def subject_evaluation(plot_ranks: bool = True, methods: List[str] = None, proportions_test: List[float] = None,
                       subject_list: List[int] = None):
    """
    Create distance and rank-table for each subject
    :param methods: List with methods ("baseline", "amusement", "stress")
    :param plot_ranks: If True: realistic ranks will be plotted and saved
    :param proportions_test: List with test-proportions
    :param subject_list: Specify subject-ids if None: all subjects are used
    """
    if methods is None:
        methods = get_classes()
    if subject_list is None:
        subject_list = get_subject_list()
    if proportions_test is None:
        proportions_test = get_proportions()

    for method in methods:
        for proportion_test in proportions_test:
            text = list()
            text.append("# Subject Rank and Distance Table")
            text.append("* method: " + str(method))
            text.append("* test-proportion: " + str(proportion_test))

            for subject in subject_list:
                results = load_results(subject_id=subject, method=method, proportion_test=proportion_test)
                overall_ranks_rank, individual_ranks_rank = run_calculate_ranks(results=results, method="rank")
                overall_ranks_score, individual_ranks_score = run_calculate_ranks(results=results, method="score")

                text_distances = create_md_distances(results=results, subject_id=subject)
                text_ranks_rank = create_md_ranks(overall_ranks=overall_ranks_rank,
                                                  individual_ranks=individual_ranks_rank, subject_id=subject)
                text_ranks_score = create_md_ranks(overall_ranks=overall_ranks_score,
                                                   individual_ranks=individual_ranks_score, subject_id=subject)

                text.append("## Subject: " + str(subject))
                text.append(text_distances)
                text.append(text_ranks_rank)
                text.append(text_ranks_score)

            # Save MD-File
            path = os.path.join(SUBJECT_PLOT_PATH, method)
            path = os.path.join(path, "test=" + str(proportion_test))
            os.makedirs(path, exist_ok=True)

            path_string = "/SW-DTW_subject-plot_" + str(method) + "_" + str(proportion_test) + ".md"
            with open(path + path_string, 'w') as outfile:
                for item in text:
                    outfile.write("%s\n" % item)

            print("SW-DTW subject-plot for method = " + str(method) + " and test-proportion = " + str(proportion_test) +
                  " saved at: " + str(path))

            # Plot realistic ranks as boxplot
            if plot_ranks:
                plot_realistic_ranks(path=path + "/SW-DTW_realistic-rank-plot_" + str(method) + "_" +
                                     str(proportion_test) + ".pdf", method=method, proportion_test=proportion_test)

            print("SW-DTW realistic-rank-plot for method = " + str(method) + " and test-proportion = " +
                  str(proportion_test) + " saved at: " + str(path))


def precision_evaluation(methods: List[str] = None, proportions_test: List[float] = None, k_list: List[int] = None):
    """
    Evaluate DTW alignments with precision@k
    :param methods: List with methods ("baseline", "amusement", "stress")
    :param proportions_test: List with test-proportions
    :param k_list: Specify k parameters in precision table; if None: all k [1 - len(subjects)] are shown
    """
    if methods is None:
        methods = get_classes()
    if proportions_test is None:
        proportions_test = get_proportions()

    for method in methods:
        for proportion_test in proportions_test:
            text = list()
            text.append("# Evaluation with precision@k")
            text.append("* method: " + str(method))
            text.append("* test-proportion: " + str(proportion_test))
            text.append(create_md_precision_combinations(rank_method="rank", method=method,
                                                         proportion_test=proportion_test, k_list=k_list))
            text.append(create_md_precision_combinations(rank_method="score", method=method,
                                                         proportion_test=proportion_test, k_list=k_list))

            # Save MD-File
            path = os.path.join(PRECISION_PATH, method)
            path = os.path.join(path, "test=" + str(proportion_test))
            os.makedirs(path, exist_ok=True)

            path_string = "/SW-DTW_precision-plot_" + str(method) + "_" + str(proportion_test) + ".md"
            with open(path + path_string, 'w') as outfile:
                for item in text:
                    outfile.write("%s\n" % item)

            print("SW-DTW precision-plot for method = " + str(method) + " and test-proportion = " + str(proportion_test)
                  + " saved at: " + str(path))


def run_optimization_evaluation():
    """
    Run complete optimizations evaluation, Evaluation of: rank-methods, classes, sensors, windows
    """
    # Get best configurations
    best_configurations = calculate_best_configurations()

    # Evaluation of rank-method
    run_rank_method_evaluation()

    # Evaluation of classes
    run_class_evaluation(rank_method=best_configurations["rank_method"])

    # Evaluation of sensor-combinations
    run_sensor_evaluation(rank_method=best_configurations["rank_method"], average_method=best_configurations["class"])

    # Evaluation of windows
    run_window_evaluation(rank_method=best_configurations["rank_method"], average_method=best_configurations["class"],
                          sensor_combination=best_configurations["sensor"])
