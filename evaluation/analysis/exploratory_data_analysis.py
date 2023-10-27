from preprocessing.data_processing.data_processing import DataProcessing
from preprocessing.datasets.dataset import Dataset
from preprocessing.process_results import load_complete_alignment_results
from config import Config

import os
import matplotlib.pyplot as plt
import numpy as np


cfg = Config.get()


def plot_subject_data(dataset: Dataset, resample_factor: int, data_processing: DataProcessing):
    """
    Plot sensor-value distribution for all subjects
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    """
    data_dict = dataset.load_dataset(resample_factor=resample_factor, data_processing=data_processing)  # read data_dict

    data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
    resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
    eda_path = os.path.join(resample_path, "eda")  # add /eda to path
    processing_path = os.path.join(eda_path, data_processing.name)  # add /data-processing to path
    os.makedirs(processing_path, exist_ok=True)

    print("Plotting subject-data! PNG-File saved at " + processing_path)

    for subject in data_dict:
        plt.plot(data_dict[subject])
        plt.legend(data_dict[subject].keys(), loc="center left")
        plt.title(label="Sensor-value distribution for subject: " + str(subject), loc="center")
        plt.ylabel('normalized sensor values (label=1: stress)')
        plt.xlabel('index | time')

        try:
            os.makedirs(eda_path, exist_ok=True)
            file_name = "eda_plot_S" + str(subject) + ".png"
            plt.savefig(fname=os.path.join(processing_path, file_name))

        except FileNotFoundError:
            print("FileNotFoundError: Invalid directory structure!")

        plt.close()


def plot_distance_heatmap(dataset: Dataset, resample_factor: int, data_processing: DataProcessing):
    """
    Plot complete subject distance as heatmap
    :param dataset: Specify dataset
    :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
    :param data_processing: Specify type of data-processing
    """
    subject_ids = dataset.get_subject_list()
    data = dict()
    data_array = list()

    # Load data
    for subject_id in subject_ids:
        results = load_complete_alignment_results(dataset=dataset, resample_factor=resample_factor,
                                                  data_processing=data_processing, subject_id=subject_id)
        data.setdefault(subject_id, list(results.values()))

    for subject_id in data:
        data_array.append(data[subject_id])

    data_array = np.array(data_array)

    # Plot heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(data_array)  # cmap=plt.cm.Blues
    axis = list(range(1, len(subject_ids) + 1))
    ax.set_xticks(np.arange(len(axis)), labels=axis)
    ax.set_yticks(np.arange(len(axis)), labels=axis)

    for i in range(len(subject_ids)):
        for j in range(len(subject_ids)):
            text = ax.text(j, i, "", ha="center", va="center", color="w")

    fig.tight_layout()
    plt.colorbar(im)
    plt.rc("font", size=24)

    # Save heatmap as png
    data_path = os.path.join(cfg.out_dir, dataset.get_dataset_name())  # add /dataset to path
    resample_path = os.path.join(data_path, "resample-factor=" + str(resample_factor))  # add /rs-factor to path
    eda_path = os.path.join(resample_path, "eda")  # add /eda to path
    processing_path = os.path.join(eda_path, data_processing.name)  # add /data-processing to path
    os.makedirs(processing_path, exist_ok=True)

    try:
        file_name = "eda_dtw_alignment_heatmap.pdf"
        plt.savefig(fname=os.path.join(processing_path, file_name), format="pdf", transparent=True, bbox_inches="tight")
        print("Distance-Heatmap saved at: " + str(os.path.join(processing_path, file_name)))

    except FileNotFoundError:
        print("FileNotFoundError: Invalid directory structure!")

    plt.close()
