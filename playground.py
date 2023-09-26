from alignments.dtw_attack import run_calculations
from alignments.dtw_alignment import run_dtw_alignments
from evaluation.analysis.exploratory_data_analysis import plot_subject_data, plot_alignment_heatmap
from evaluation.evaluation import subject_evaluation, precision_evaluation, run_optimization_evaluation, \
    run_calculate_max_precision
from evaluation.optimization.class_evaluation import run_class_evaluation
from evaluation.optimization.rank_method_evaluation import run_rank_method_evaluation
from evaluation.optimization.sensor_evaluation import run_sensor_evaluation
from evaluation.optimization.window_evaluation import run_window_evaluation
from preprocessing.data_preparation import preprocess_data
from evaluation.optimization.overall_evaluation import run_overall_evaluation


"""
Example Calculations
------------------------------------------------------------------------------------------------------------------------
"""

"""1. Preprocess WESAD dataset and save dataset as data_dict.pickle to /dataset"""
# preprocess_data()

"""2. Plot exploratory data analysis to /out/eda"""
# plot_subject_data()

"""3. Calculate DTW-alignments and save results to /out/alignments"""
# run_calculations(methods=["baseline", "amusement", "stress"], proportions=[0.15])

"""4. Calculate DTW-alignments over complete sensor signals and save results to /out/alignments/complete"""
# run_dtw_alignments(resample_factor=4)

"""5. Plot DTW alignment subject distance heatmap and save plot to /out/eda"""
# plot_alignment_heatmap()

"""5. Evaluate DTW-alignment results per subject; save MD-tables with distance and rank results and realistic-rank-plots
to /out/subject-plots"""
# subject_evaluation()

"""6. Evaluation DTW-alignment results overall mit precision@k; save MD-tables with precision values"""
# precision_evaluation(k_list=[1, 3, 5])

"""7. Evaluation of rank-method; save precision@k values as MD-table"""
# run_rank_method_evaluation()

"""8. Evaluation of classes, save precision@k values as MD-table"""
# run_class_evaluation(rank_method="score")

"""9. Evaluation of sensor-combinations, save precision@k values as MD-table"""
# run_sensor_evaluation(rank_method="score", average_method="weighted-mean")

"""10. Evaluation of windows, save precision@k values as MD-table"""
# run_window_evaluation(rank_method="score", average_method="weighted-mean",
#                       sensor_combination=[["bvp", "acc", "temp"]])

"""11. Complete optimization evaluation, save precision@k values as MD-table"""
# run_optimization_evaluation()

"""12. Calculate maximum precisions, save precision@k values as json file"""
# run_calculate_max_precision(k_list=list(range(1, 16)))

"""13. Overall evaluation with (DTW-results, maximum results, random guess results), save precision@k values as 
MD-table"""
# run_overall_evaluation(save_weightings=True)
