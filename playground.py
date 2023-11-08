from preprocessing.data_processing.standard_processing import StandardProcessing
from preprocessing.data_processing.dba_processing import DbaProcessing
from preprocessing.datasets.load_wesad import Wesad
from preprocessing.datasets.load_cgan import WesadCGan
from preprocessing.datasets.load_dgan import WesadDGan
from alignments.dtw_attacks.single_dtw_attack import SingleDtwAttack
from alignments.dtw_attacks.multi_dtw_attack import MultiDtwAttack
from alignments.dtw_attacks.slicing_dtw_attack import SlicingDtwAttack
from alignments.run_dtw_attacks import run_dtw_attack
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
dataset = WesadCGan(dataset_size=100)
resample_factor = 1000
data_processing = StandardProcessing()
dtw_attack = SlicingDtwAttack()


start = time.perf_counter()

"""1. Calculate DTW-alignments and save results to /out/alignments"""
# run_dtw_attack(dtw_attack=dtw_attack, dataset=dataset, data_processing=data_processing,
#                test_window_sizes=[10, 20, 30], resample_factor=resample_factor, multi=3)

"""2. Calculate DTW-alignments over complete sensor signals and save results to /out/alignments/complete"""
# run_dtw_alignments(dataset=dataset, data_processing=data_processing, resample_factor=resample_factor)

"""3. Plot DTW alignment subject distance heatmap and save plot to /out/eda"""
# plot_distance_heatmap(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing)

"""4. Plot exploratory data analysis to /out/eda"""
# plot_subject_data(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing)

"""5. Evaluate DTW-alignment results per subject; save MD-tables with distance and rank results and realistic-rank-plots
to /out/subject-plots"""
# subject_evaluation(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
#                    dtw_attack=dtw_attack)

"""6. Evaluation DTW-alignment results overall mit precision@k; save MD-tables with precision values"""
# precision_evaluation(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
#                      dtw_attack=dtw_attack)

"""7. Complete optimization evaluation, save precision@k values as MD-table"""
# run_optimization_evaluation(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
#                             dtw_attack=dtw_attack)

"""8. Calculate maximum precisions, save precision@k values as json file"""
# run_calculate_max_precision(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
#                             dtw_attack=dtw_attack, use_existing_weightings=True)

"""9. Overall evaluation with (DTW-results, maximum results, random guess results), save precision@k values as
MD-table"""
# run_overall_evaluation(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
#                        dtw_attack=dtw_attack, save_weightings=True)

end = time.perf_counter()
print("Runtime: " + str(round(end - start, 2)) + "s")
