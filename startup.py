from alignments.dtw_attack import run_calculations
from alignments.dtw_alignment import run_dtw_alignments
from evaluation.analysis.exploratory_data_analysis import plot_subject_data, plot_alignment_heatmap
from evaluation.evaluation import subject_evaluation, precision_evaluation, run_optimization_evaluation, \
    run_calculate_max_precision
from evaluation.optimization.overall_evaluation import run_overall_evaluation


def main():
    """
    Run DTW-Attack and Evaluation
    """
    try:
        """1. Plot exploratory data analysis to /out/eda"""
        plot_subject_data()

        """2. Calculate DTW-alignments over complete sensor signals and save results to /out/alignments/complete"""
        run_dtw_alignments(resample_factor=4)

        """3. Plot DTW alignment subject distance heatmap and save plot to /out/eda"""
        plot_alignment_heatmap()

        """4. Calculate DTW-alignments and save results to /out/alignments"""
        run_calculations(methods=["baseline", "amusement", "stress"], test_window_sizes=[1, 10, 100])

        """5. Evaluate DTW-alignment results per subject; save MD-tables with distance and rank results and 
        realistic-rank-plots to /out/subject-plots"""
        subject_evaluation()

        """6. Evaluation DTW-alignment results overall mit precision@k; save MD-tables with precision values"""
        precision_evaluation(k_list=[1, 3, 5])

        """7. Complete optimization evaluation, save precision@k values as MD-table"""
        run_optimization_evaluation()

        """8. Calculate maximum precisions, save precision@k values as json file"""
        run_calculate_max_precision(k_list=list(range(1, 16)))

        """9. Overall evaluation with (DTW-results, maximum results, random guess results), save precision@k values as 
        MD-table"""
        run_overall_evaluation(save_weightings=True)

    except Exception as e:
        print(e)

    pass


if __name__ == '__main__':
    main()
