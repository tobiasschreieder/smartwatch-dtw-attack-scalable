# Smartwatch-DTW-Attack-Scalable
* Git repository addressing the paper "Privacy at Risk: Exploiting Similarities in Health Data for Identity Inference" and further analysis with the purpose of scalability
* Link to the paper: https://arxiv.org/abs/2308.08310

## General Information:
Smartwatches enable the efficient collection of health data that can be used for research and comprehensive analysis to improve the health of individuals. In addition to the analysis capabilities, ensuring privacy when handling health data is a critical concern as the collection and analysis of such data become pervasive. Since health data contains sensitive information, it should be handled with responsibility and is therefore often treated anonymously. However, also the data itself can be exploited to reveal information and break anonymity. We propose a novel similarity-based re-identification attack on time-series health data and thereby unveil a significant vulnerability. Despite privacy measures that remove identifying information, our attack demonstrates that a brief amount of various sensor data from a target individual is adequate to possibly identify them within a database of other samples, solely based on sensor-level similarities. In our example scenario, where data owners leverage health data from smartwatches, findings show that we are able to correctly link the target data in two out of three cases. User privacy is thus already inherently threatened by the data itself and even when removing personal information. 

## Dataset:
Please download the WESAD dataset from https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx and save it unzipped under /dataset/WESAD

## Code calculations:
1. Calculate DTW-alignments over complete sensor signals and save results to /out/dataset/resample-factor=x/alignments/complete
* run_dtw_alignments()

2. Plot DTW alignment subject distance heatmap and save plot to /out/dataset/resample-factor=x/eda
* plot_distance_heatmap()

3. Calculate DTW-alignments and save results to /out/dataset/resample-factor=x/alignments
* run_calculations()

4. Plot exploratory data analysis to /out/eda
* plot_subject_data()

5. Evaluate DTW-alignment results per subject; save MD-tables with distance and rank results and realistic-rank-plots to /out/dataset/resample-factor=x/subject-plots
* subject_evaluation()

6. Evaluation DTW-alignment results overall mit precision@k; save MD-tables with precision values
* precision_evaluation()

7. Complete optimization evaluation, save precision@k values as MD-table
* run_optimization_evaluation()

8. Calculate maximum precisions, save precision@k values as json file
* run_calculate_max_precision()

9. Overall evaluation with (DTW-results, maximum results, random guess results), save precision@k values as MD-table
* run_overall_evaluation()

**Run startup.py for complete DTW-attack and evaluations**
* Please make sure that there is enough RAM available (>= 64 GB) or downsample the dataset!

## Evaluations:
* All evaluations are saved to /out
* The final evaluations can be found at /out/dataset/resample-factor=x/evaluations
