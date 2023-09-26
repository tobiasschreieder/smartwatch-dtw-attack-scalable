# Smartwatch-DTW-Attack

## Dataset:
Please download the WESAD dataset from https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx and save it unzipped under /dataset/WESAD

## Code calculations:
1. Preprocess WESAD dataset and save dataset as data_dict.pickle to /dataset
* preprocess_data()

2. Plot exploratory data analysis to /out/eda
* plot_subject_data()

3. Calculate DTW-alignments over complete sensor signals and save results to /out/alignments/complete
* run_dtw_alignments(resample_factor=4)

4. Plot DTW alignment subject distance heatmap and save plot to /out/eda
* plot_alignment_heatmap()

5. Calculate DTW-alignments and save results to /out/alignments
* run_calculations(methods=["baseline", "amusement", "stress"], proportions=[0.0001, 0.001, 0.01, 0.05, 0.1])

6. Evaluate DTW-alignment results per subject; save MD-tables with distance and rank results and realistic-rank-plots to /out/subject-plots
* subject_evaluation()

7. Evaluation DTW-alignment results overall mit precision@k; save MD-tables with precision values
* precision_evaluation(k_list=[1, 3, 5])

8. Complete optimization evaluation, save precision@k values as MD-table
* run_optimization_evaluation()

9. Calculate maximum precisions, save precision@k values as json file
* run_calculate_max_precision(k_list=list(range(1, 16)))

10. Overall evaluation with (DTW-results, maximum results, random guess results), save precision@k values as MD-table
* run_overall_evaluation(save_weightings=True)

**Run startup.py for complete DTW-attack and evaluations**
* Please make sure that there is enough RAM available (>= 64 GB) or downsample the dataset!

## Evaluations:
* All evaluations are saved to /out
* The final evaluations can be found at /out/evaluations