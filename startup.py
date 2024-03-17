from alignments.run_dtw_attacks import run_dtw_attack
from alignments.dtw_attacks.slicing_dtw_attack import SlicingDtwAttack
from evaluation.evaluation import run_optimization_evaluation, run_calculate_max_precision
from evaluation.optimization.overall_evaluation import run_overall_evaluation
from preprocessing.data_processing.standard_processing import StandardProcessing
from preprocessing.datasets.load_wesad import Wesad


def main():
    """
    Exemplary execution of a DTW attack
    """
    # Specify parameters
    dataset = Wesad(dataset_size=15)
    resample_factor = 1000
    data_processing = StandardProcessing()
    dtw_attack = SlicingDtwAttack()
    result_selection_method = "min"

    try:
        """1. Calculate DTW-alignments and save results to /out_private/dataset/resample-factor=x/alignments"""
        run_dtw_attack(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                       dtw_attack=dtw_attack, test_window_sizes=[i for i in range(1, 37)])

        """2. Complete optimization evaluation, save precision@k values as MD-table"""
        run_optimization_evaluation(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                                    dtw_attack=dtw_attack, result_selection_method=result_selection_method)

        """3. Calculate maximum precisions, save precision@k values as json file"""
        run_calculate_max_precision(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                                    dtw_attack=dtw_attack, result_selection_method=result_selection_method,
                                    use_existing_weightings=False)

        """4. Overall evaluation with (DTW-results, maximum results, random guess results), save precision@k values as 
        MD-table"""
        run_overall_evaluation(dataset=dataset, resample_factor=resample_factor, data_processing=data_processing,
                               dtw_attack=dtw_attack, result_selection_method=result_selection_method,
                               save_weightings=True)

    except Exception as e:
        print(e)

    pass


if __name__ == '__main__':
    main()
