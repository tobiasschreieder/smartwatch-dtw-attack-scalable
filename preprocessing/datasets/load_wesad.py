from preprocessing.datasets.dataset import Dataset
from preprocessing.data_processing.data_processing import DataProcessing
from config import Config

from typing import Dict, List
import os
import pickle
import pandas as pd
from scipy import signal


cfg = Config.get()


# List with all available subject_ids
SUBJECT_LIST = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

# All available classes
CLASSES = ["non-stress", "stress"]


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
        eda_upsampled = signal.resample(eda_signal, len(bvp_signal))
        temp_upsampled = signal.resample(temp_signal, len(bvp_signal))
        acc_x_upsampled = signal.resample(acc_x_signal, len(bvp_signal))
        acc_y_upsampled = signal.resample(acc_y_signal, len(bvp_signal))
        acc_z_upsampled = signal.resample(acc_z_signal, len(bvp_signal))
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
        df['label'] = df['label'].replace([1.0, 2.0, 3.0], [0, 1, 0])
        df.reset_index(drop=True, inplace=True)

        # Normalize data (no train test leakage since data frame per subject)
        df = (df - df.min()) / (df.max() - df.min())
        return df


class Wesad(Dataset):
    """
    Class to generate, load and preprocess WESAD dataset
    """
    def __init__(self):
        """
        Try to load preprocessed WESAD dataset (wesad_data.pickle); if not available -> generate wesad_data.pickle
        """
        super().__init__()

        self.name = "WESAD"

        try:
            with open(os.path.join(cfg.data_dir, 'wesad_data.pickle'), "rb") as f:
                self.data = pickle.load(f)

        except FileNotFoundError:
            print("FileNotFoundError: Invalid directory structure! Please make sure that /dataset exists.")
            print("Creating wesad_data.pickle from WESAD dataset.")

            # Load data of all subjects in subject_list
            data_dict = dict()
            for i in SUBJECT_LIST:
                subject = Subject(os.path.join(cfg.data_dir, "WESAD"), i)
                data = subject.get_subject_dataframe()
                data_dict.setdefault(i, data)
            self.data = data_dict

            # Save data_dict
            try:
                with open(os.path.join(cfg.data_dir, 'wesad_data.pickle'), 'wb') as f:
                    pickle.dump(data_dict, f)

            except FileNotFoundError:
                print("FileNotFoundError: Invalid directory structure! Please make sure that /dataset exists.")

    def load_dataset(self, data_processing: DataProcessing, resample_factor: int = None) -> Dict[int, pd.DataFrame]:
        """
        Load preprocessed dataset from /dataset
        :param data_processing: Specify type of data-processing
        :param resample_factor: Specify down-sample factor (1: no down-sampling; 2: half-length)
        :return: Dictionary with preprocessed data
        """
        data_dict = dict()
        data = self.data

        if resample_factor is not None:
            for subject_id in data:
                sensor_data = data[subject_id]
                label_data = data[subject_id]["label"]
                sensor_data = sensor_data.drop("label", axis=1)
                column_names = sensor_data.columns.values.tolist()

                sensor_data = signal.resample(sensor_data, round(len(data[subject_id]) / resample_factor))
                sensor_data = pd.DataFrame(data=sensor_data, columns=column_names)

                label_data.index = [(1 / resample_factor) * i for i in range(len(label_data))]
                label_data.index = pd.to_datetime(label_data.index, unit='s')

                sensor_data.index = pd.to_datetime(sensor_data.index, unit='s')
                sensor_data = sensor_data.join(label_data)
                sensor_data['label'] = sensor_data['label'].fillna(method='ffill')
                sensor_data.reset_index(drop=True, inplace=True)

                data_dict.setdefault(subject_id, sensor_data)

        else:
            data_dict = data

        # Run data-processing
        data_dict = data_processing.process_data(data_dict=data_dict)

        return data_dict

    def get_dataset_name(self) -> str:
        """
        Get name of dataset
        :return: String with name
        """
        return self.name

    def get_subject_list(self) -> List[int]:
        """
        Get list with all available subjects
        :return: List with subject-ids
        """
        return SUBJECT_LIST

    def get_classes(self) -> List[str]:
        """
        Get classes ("baseline", "amusement", "stress")
        :return: List with all classes
        """
        return CLASSES
