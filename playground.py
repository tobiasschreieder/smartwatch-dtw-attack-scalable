from preprocessing.data_processing.standard_processing import StandardProcessing
from preprocessing.data_processing.dba_processing import DbaProcessing
from preprocessing.data_processing.pca_processing import PcaProcessing
from preprocessing.datasets.load_wesad import Wesad
from preprocessing.datasets.load_wesad_private import WesadPrivate
from preprocessing.datasets.load_cgan import WesadCGan
from preprocessing.datasets.load_dgan import WesadDGan
from preprocessing.datasets.load_combined import WesadCombined
from alignments.dtw_attacks.single_dtw_attack import SingleDtwAttack
from alignments.dtw_attacks.multi_dtw_attack import MultiDtwAttack
from alignments.dtw_attacks.slicing_dtw_attack import SlicingDtwAttack
from alignments.dtw_attacks.multi_slicing_dtw_attack import MultiSlicingDtwAttack
from alignments.run_dtw_attacks import run_dtw_attack, simulate_isolated_dtw_attack
from alignments.dtw_alignment import run_dtw_alignments
from evaluation.analysis.exploratory_data_analysis import plot_distance_heatmap, plot_subject_data
from evaluation.evaluation import (subject_evaluation, precision_evaluation, run_optimization_evaluation,
                                   run_calculate_max_precision)
from evaluation.optimization.overall_evaluation import run_overall_evaluation

import time


"""
Example Calculations
------------------------------------------------------------------------------------------------------------------------
"""
# Specify parameters

resample_factor = 1000
data_processing = StandardProcessing()
datasets = [Wesad(dataset_size=15), WesadCGan(dataset_size=15), WesadDGan(dataset_size=15)]

start = time.perf_counter()

for dataset in datasets:
    dtw_attack = SingleDtwAttack()
    test_window_sizes = [i for i in range(1, 37)]
    result_selection_method = "min"
    run_dtw_attack(dtw_attack=dtw_attack, dataset=dataset, data_processing=data_processing,
                   test_window_sizes=test_window_sizes, resample_factor=resample_factor, multi=3)
    run_optimization_evaluation(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                                dtw_attack=dtw_attack, result_selection_method=result_selection_method)
    run_calculate_max_precision(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                                dtw_attack=dtw_attack, result_selection_method=result_selection_method,
                                use_existing_weightings=False)
    run_overall_evaluation(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                           dtw_attack=dtw_attack, result_selection_method=result_selection_method, save_weightings=True)

    dtw_attack = MultiDtwAttack()
    test_window_sizes = [i for i in range(1, 13)]
    result_selection_method = "mean"
    run_dtw_attack(dtw_attack=dtw_attack, dataset=dataset, data_processing=data_processing,
                   test_window_sizes=test_window_sizes, resample_factor=resample_factor, multi=3)
    run_optimization_evaluation(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                                dtw_attack=dtw_attack, result_selection_method=result_selection_method)
    run_calculate_max_precision(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                                dtw_attack=dtw_attack, result_selection_method=result_selection_method,
                                use_existing_weightings=False)
    run_overall_evaluation(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                           dtw_attack=dtw_attack, result_selection_method=result_selection_method, save_weightings=True)

    dtw_attack = SlicingDtwAttack()
    test_window_sizes = [i for i in range(1, 37)]
    result_selection_method = "min"
    run_dtw_attack(dtw_attack=dtw_attack, dataset=dataset, data_processing=data_processing,
                   test_window_sizes=test_window_sizes, resample_factor=resample_factor, multi=3)
    run_optimization_evaluation(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                                dtw_attack=dtw_attack, result_selection_method=result_selection_method)
    run_calculate_max_precision(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                                dtw_attack=dtw_attack, result_selection_method=result_selection_method,
                                use_existing_weightings=False)
    run_overall_evaluation(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                           dtw_attack=dtw_attack, result_selection_method=result_selection_method, save_weightings=True)

    dtw_attack = MultiSlicingDtwAttack()
    test_window_sizes = [i for i in range(1, 13)]
    result_selection_method = "mean"
    run_dtw_attack(dtw_attack=dtw_attack, dataset=dataset, data_processing=data_processing,
                   test_window_sizes=test_window_sizes, resample_factor=resample_factor, multi=3)
    run_optimization_evaluation(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                                dtw_attack=dtw_attack, result_selection_method=result_selection_method)
    run_calculate_max_precision(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                                dtw_attack=dtw_attack, result_selection_method=result_selection_method,
                                use_existing_weightings=False)
    run_overall_evaluation(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                           dtw_attack=dtw_attack, result_selection_method=result_selection_method, save_weightings=True)

end = time.perf_counter()
print("Runtime: " + str(round(end - start, 2)) + "s")
