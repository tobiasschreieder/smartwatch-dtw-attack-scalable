from typing import Dict, List
import os
import pickle
import pandas as pd
import scipy.signal


# Specify path
MAIN_PATH = os.path.abspath(os.getcwd())
DATA_PATH = os.path.join(MAIN_PATH, "dataset")  # add /dataset to path

# List with all available subject_ids
SUBJECT_LIST = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

# List with all sensor combinations
SENSOR_COMBINATIONS = [["bvp"], ["eda"], ["acc"], ["temp"], ["bvp", "eda"], ["bvp", "temp"], ["bvp", "acc"],
                       ["eda", "acc"], ["eda", "temp"], ["acc", "temp"], ["bvp", "eda", "acc"], ["bvp", "eda", "temp"],
                       ["bvp", "acc", "temp"], ["eda", "acc", "temp"], ["bvp", "eda", "acc", "temp"]]


class Subject:
    """
    Subject of the WESAD dataset.
    Subject Class inspired by: https://github.com/WJMatthew/WESAD
    Preprocessing based on Gil-Martin et al. 2022: Human stress detection with wearable sensors using convolutional
    neural networks: https://ieeexplore.ieee.org/document/9669993
    """
    def __init__(self, data_path, subject_number):
        """
        Load WESAD dataset
        :param data_path: Path of WESAD dataset
        :param subject_number: Specify current subject number
        """
        self.name = f'S{subject_number}'
        self.subject_keys = ['signal', 'label', 'subject']
        self.signal_keys = ['chest', 'wrist']
        self.chest_keys = ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']
        self.wrist_keys = ['ACC', 'BVP', 'EDA', 'TEMP']

        with open(os.path.join(data_path, self.name) + '/' + self.name + '.pkl', 'rb') as file:
            self.data = pickle.load(file, encoding='latin1')
        self.labels = self.data['label']

    def get_wrist_data(self) -> pd.DataFrame:
        """
        Method to get wrist data
        :return: Data measured by the E4 Empatica
        """
        data = self.data['signal']['wrist']
        return data

    def get_subject_dataframe(self) -> pd.DataFrame:
        """
        Preprocess and upsample WESAD dataset
        :return: Dataframe with the preprocessed data of the subject
        """
        wrist_data = self.get_wrist_data()
        bvp_signal = wrist_data['BVP'][:, 0]
        eda_signal = wrist_data['EDA'][:, 0]
        acc_x_signal = wrist_data['ACC'][:, 0]
        acc_y_signal = wrist_data['ACC'][:, 1]
        acc_z_signal = wrist_data['ACC'][:, 2]
        temp_signal = wrist_data['TEMP'][:, 0]

        # Upsampling data to match BVP data sampling rate using fourier method as described in Paper/dataset
        eda_upsampled = scipy.signal.resample(eda_signal, len(bvp_signal))
        temp_upsampled = scipy.signal.resample(temp_signal, len(bvp_signal))
        acc_x_upsampled = scipy.signal.resample(acc_x_signal, len(bvp_signal))
        acc_y_upsampled = scipy.signal.resample(acc_y_signal, len(bvp_signal))
        acc_z_upsampled = scipy.signal.resample(acc_z_signal, len(bvp_signal))
        label_df = pd.DataFrame(self.labels, columns=['label'])
        label_df.index = [(1 / 700) * i for i in range(len(label_df))]  # 700 is the sampling rate of the label
        label_df.index = pd.to_datetime(label_df.index, unit='s')

        data_arrays = zip(bvp_signal, eda_upsampled, acc_x_upsampled, acc_y_upsampled, acc_z_upsampled, temp_upsampled)
        df = pd.DataFrame(data=data_arrays, columns=['bvp', 'eda', 'acc_x', 'acc_y', 'acc_z', 'temp'])
        df.index = [(1 / 64) * i for i in range(len(df))]  # 64 = sampling rate of BVP
        df.index = pd.to_datetime(df.index, unit='s')
        df = df.join(label_df)
        df['label'] = df['label'].fillna(method='ffill')
        df.reset_index(drop=True, inplace=True)
        df.drop(df[df['label'].isin([0.0, 4.0, 5.0, 6.0, 7.0])].index, inplace=True)
        df['label'] = df['label'].replace([1.0, 2.0, 3.0], [0, 1, 0.5])
        df.reset_index(drop=True, inplace=True)

        # Normalize data (no train test leakage since data frame per subject)
        df = (df - df.min()) / (df.max() - df.min())
        return df


def preprocess_data():
    """
    Load WESAD dataset, preprocess dataset and save dataset as data_dict.pickle
    """
    # Load data of all subjects in subject_list
    data_dict = dict()
    for i in SUBJECT_LIST:
        subject = Subject(os.path.join(DATA_PATH, "WESAD"), i)
        data = subject.get_subject_dataframe()
        data_dict.setdefault(i, data)

    # save data_dict
    try:
        with open(DATA_PATH + '/data_dict.pickle', 'wb') as f:
            pickle.dump(data_dict, f)

    except FileNotFoundError:
        print("FileNotFoundError: Invalid directory structure! Please make sure that /dataset exists.")


def load_dataset() -> Dict[int, pd.DataFrame]:
    """
    Load preprocessed dataset from /dataset
    :return: Dictionary with preprocessed data
    """
    data_dict = dict()
    try:
        with open(DATA_PATH + '/data_dict.pickle', 'rb') as f:
            data_dict = pickle.load(f)

    except FileNotFoundError:
        print("FileNotFoundError: Invalid directory structure or data_dict.pickle does not exist.")
        preprocess_data()
        print("Generating data_dict.pickle from WESAD dataset!")

    return data_dict


def get_subject_list() -> List[int]:
    """
    Get list with all available subjects
    :return: List with subject-ids
    """
    return SUBJECT_LIST


def get_sensor_combinations() -> List[List[str]]:
    """
    Get sensor-combinations
    :return: sensor-combinations
    """
    return SENSOR_COMBINATIONS
