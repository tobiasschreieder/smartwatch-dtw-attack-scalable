from preprocessing.data_preparation import load_dataset, get_subject_list
from preprocessing.process_results import load_complete_alignment_results

import os
import matplotlib.pyplot as plt
import numpy as np


MAIN_PATH = os.path.abspath(os.getcwd())
EDA_PATH = os.path.join(MAIN_PATH, "out")  # add /out to path
EDA_PATH = os.path.join(EDA_PATH, "eda")  # add /eda to path


def plot_subject_data():
    """
    Plot sensor-value distribution for all subjects
    :return:
    """
    data_dict = load_dataset()  # read data_dict

    for subject in data_dict:
        plt.plot(data_dict[subject])
        plt.legend(data_dict[subject].keys(), loc="center left")
        plt.title(label="Sensor-value distribution for subject: " + str(subject), loc="center")
        plt.ylabel('normalized sensor values (label=1: stress)')
        plt.xlabel('index | time')

        try:
            os.makedirs(EDA_PATH, exist_ok=True)
            plt.savefig(fname=EDA_PATH + "/eda_plot_S" + str(subject) + ".png")

        except FileNotFoundError:
            print("FileNotFoundError: Invalid directory structure!")

        plt.close()


def plot_alignment_heatmap(normalized_data: bool = True):
    """
    Plot complete subject alignments as heatmap
    :param normalized_data: If True -> use normalized results
    """
    subject_ids = get_subject_list()
    data = dict()
    data_array = list()

    # Load data
    for subject_id in subject_ids:
        results = load_complete_alignment_results(subject_id=subject_id, normalized_data=normalized_data)
        # Change distance to similarity
        for res in results:
            results[res] = 1 - results[res]
        data.setdefault(subject_id, list(results.values()))

    for subject_id in data:
        data_array.append(data[subject_id])

    data_array = np.array(data_array)

    # Plot heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(data_array, cmap=plt.cm.Blues)
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
    try:
        os.makedirs(EDA_PATH, exist_ok=True)
        plt.savefig(fname=EDA_PATH + "/eda_dtw_alignment_heatmap.pdf", format="pdf", transparent=True,
                    bbox_inches="tight")

    except FileNotFoundError:
        print("FileNotFoundError: Invalid directory structure!")

    plt.close()
