from preprocessing.data_processing.data_processing import DataProcessing
from preprocessing.datasets.dataset import Dataset
from alignments.dtw_attacks.dtw_attack import DtwAttack
from alignments.dtw_attacks.single_dtw_attack import SingleDtwAttack
from alignments.dtw_attacks.multi_dtw_attack import MultiDtwAttack


from typing import List


def run_dtw_attack(dtw_attack: DtwAttack, dataset: Dataset, resample_factor: int, data_processing: DataProcessing,
                   test_window_sizes: List[int], multi: int = 3, additional_windows: int = 1000, n_jobs: int = -1,
                   methods: List[str] = None, subject_ids: List[int] = None):
    """
    Run DTW-attack with all given parameters and save results as json
    :param dtw_attack: Specify DTW-attack (SingleDtwAttack, MultiDtwAttack)
    :param dataset: Specify dataset, which should be used
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    :param test_window_sizes: List with all test windows that should be used (int)
    :param multi: Specify number of combined single attacks
    :param additional_windows: Specify amount of additional windows to be removed around test-window
    :param n_jobs: Number of processes to use (parallelization)
    :param methods: List with all method that should be used -> "non-stress" / "stress" (str)
    :param subject_ids: List with all subjects that should be used as test subjects (int) -> None = all subjects
    """
    if dtw_attack.name == SingleDtwAttack().name:
        print("Starting Single-DTW-Attack!")
        single_dtw_attack = SingleDtwAttack()
        single_dtw_attack.run_calculations(dataset=dataset, test_window_sizes=test_window_sizes,
                                           resample_factor=resample_factor, data_processing=data_processing,
                                           additional_windows=additional_windows, n_jobs=n_jobs, methods=methods,
                                           subject_ids=subject_ids)

    elif dtw_attack.name == MultiDtwAttack().name:
        print("Starting Multi-DTW-Attack!")
        multi_dtw_attack = MultiDtwAttack()
        multi_dtw_attack.run_calculations(dataset=dataset, test_window_sizes=test_window_sizes,
                                          data_processing=data_processing, resample_factor=resample_factor,
                                          multi=multi, additional_windows=additional_windows, n_jobs=n_jobs,
                                          methods=methods, subject_ids=subject_ids)

    else:
        print("Please specify a valid DTW-Attack!")
