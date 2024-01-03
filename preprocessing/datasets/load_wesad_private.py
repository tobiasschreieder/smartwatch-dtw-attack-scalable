from preprocessing.data_processing.data_processing import DataProcessing
from preprocessing.data_processing.standard_processing import StandardProcessing
from preprocessing.datasets.dataset import Dataset
from preprocessing.datasets.load_wesad import Wesad
from privacy.laplace_noise import create_noisy_data
from config import Config

from typing import Dict, List
import os
import pickle
import pandas as pd
from scipy import signal


cfg = Config.get()


# All available classes
CLASSES = ["non-stress", "stress"]


class WesadPrivate(Dataset):
    """
    Class to generate, load and preprocess WESAD dataset
    """
    def __init__(self, dataset_size: int, noise_multiplier: float):
        """
        Try to load preprocessed WESAD dataset (wesad_data.pickle); if not available -> generate wesad_data.pickle
        :param dataset_size: Specify amount of subjects in dataset
        :param noise_multiplier: Specify noise multiplier (scale parameter) of laplace distribution (> 0.0)
        """
        super().__init__(dataset_size=dataset_size)

        self.name = "WESAD-p" + str(noise_multiplier)

        # List with all available subject_ids
        subject_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
        if dataset_size > 15:
            print("Size of the data set is too large! Set size to 15.")
            dataset_size = 15
        subject_list = subject_list[:dataset_size]
        self.subject_list = subject_list

        filename = "wesad_p" + str(noise_multiplier) + "_" + str(dataset_size) + ".pickle"

        try:
            with open(os.path.join(cfg.data_dir, filename), "rb") as f:
                self.data = pickle.load(f)

        except FileNotFoundError:
            print("FileNotFoundError: Invalid directory structure! Please make sure that /dataset exists.")
            print("Creating wesad_private_data.pickle from WESAD dataset.")

            wesad_data = Wesad(dataset_size=15).load_dataset(resample_factor=1, data_processing=StandardProcessing())
            wesad_private = dict()
            for subject_id in wesad_data:
                label = wesad_data[subject_id]["label"]
                signal = wesad_data[subject_id].drop("label", axis=1)  # Label should not be changed wth noise
                columns = signal.columns
                signal = signal.to_numpy()
                signal = signal.transpose()

                noisy_signal = create_noisy_data(data=signal, noise_multiplier=noise_multiplier)
                noisy_signal = noisy_signal.transpose()
                noisy_data = pd.DataFrame(data=noisy_signal, columns=columns)
                noisy_data["label"] = label
                wesad_private.setdefault(subject_id, noisy_data)

            self.data = wesad_private

            # Save data_dict
            try:
                with open(os.path.join(cfg.data_dir, filename), 'wb') as f:
                    pickle.dump(wesad_private, f)

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

    def get_classes(self) -> List[str]:
        """
        Get classes ("baseline", "amusement", "stress")
        :return: List with all classes
        """
        return CLASSES