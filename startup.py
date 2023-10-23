from alignments.dtw_attack import run_calculations
from alignments.dtw_alignment import run_dtw_alignments
from evaluation.analysis.exploratory_data_analysis import plot_subject_data, plot_alignment_heatmap
from evaluation.evaluation import subject_evaluation, precision_evaluation, run_optimization_evaluation, \
    run_calculate_max_precision
from evaluation.optimization.overall_evaluation import run_overall_evaluation
from preprocessing.datasets.load_wesad import Wesad


def main():
    """
    Run DTW-Attack and Evaluation
    """
    # Specify parameters
    dataset = Wesad()
    resample_factor = 1000

    try:
        """1. Calculate DTW-alignments over complete sensor signals and save results to /out/alignments/complete"""
        run_dtw_alignments(dataset=dataset, resample_factor=resample_factor)

        """2. Plot DTW alignment subject distance heatmap and save plot to /out/dataset/resample-factor=x/eda"""
        plot_alignment_heatmap(dataset=dataset, resample_factor=resample_factor)

        """3. Calculate DTW-alignments and save results to /out/dataset/resample-factor=x/alignments"""
        run_calculations(dataset=dataset, resample_factor=resample_factor, test_window_sizes=[1, 10, 100])

        """4. Plot exploratory data analysis to /out/eda"""
        plot_subject_data(dataset=dataset, resample_factor=resample_factor)

        """5. Evaluate DTW-alignment results per subject; save MD-tables with distance and rank results and 
        realistic-rank-plots to /out/subject-plots"""
        subject_evaluation(dataset=dataset, resample_factor=resample_factor)

        """6. Evaluation DTW-alignment results overall mit precision@k; save MD-tables with precision values"""
        precision_evaluation(dataset=dataset, resample_factor=resample_factor, k_list=[1, 3, 5])

        """7. Complete optimization evaluation, save precision@k values as MD-table"""
        run_optimization_evaluation(dataset=dataset, resample_factor=resample_factor)

        """8. Calculate maximum precisions, save precision@k values as json file"""
        run_calculate_max_precision(dataset=dataset, resample_factor=resample_factor)

        """9. Overall evaluation with (DTW-results, maximum results, random guess results), save precision@k values as 
        MD-table"""
        run_overall_evaluation(dataset=dataset, resample_factor=resample_factor, save_weightings=True)

    except Exception as e:
        print(e)

    pass


if __name__ == '__main__':
    main()
