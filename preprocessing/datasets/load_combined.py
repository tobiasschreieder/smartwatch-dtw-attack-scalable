from preprocessing.datasets.dataset import Dataset
from preprocessing.datasets.load_wesad import Wesad
from preprocessing.datasets.load_gan import WesadGan
from config import Config

from typing import Dict, List
import os
import pickle
import pandas as pd
from scipy import signal


cfg = Config.get()


# List with all available subject_ids
SUBJECT_LIST = Wesad().get_subject_list() + WesadGan().get_subject_list()

# All available classes
CLASSES = Wesad().get_classes()

# List with all sensor combinations
SENSOR_COMBINATIONS = Wesad().get_sensor_combinations()


class WesadCombined(Dataset):
    """
    Class to generate, load and preprocess Combined WESAD dataset (WESAD + WESAD-GAN)
    """
    def __init__(self):
        """
        Try to load preprocessed WESAD dataset (wesad_data.pickle); if not available -> generate wesad_data.pickle
        """
        super().__init__()

        self.name = "WESAD-Combined"

        try:
            with open(os.path.join(cfg.data_dir, 'wesad_combined_data.pickle'), "rb") as f:
                self.data = pickle.load(f)

        except FileNotFoundError:
            print("FileNotFoundError: Invalid directory structure! Please make sure that /dataset exists.")
            print("Creating wesad_combined_data.pickle from WESAD and WESAD-GAN dataset.")

            # Load data of all subjects in subject_list
            wesad_data = Wesad().load_dataset(resample_factor=1)
            wesad_gan_data = WesadGan().load_dataset(resample_factor=1)
            data_dict = wesad_data
            for k, v in wesad_gan_data.items():
                data_dict.setdefault(k, v)
            self.data = data_dict

            # Save data_dict
            try:
                with open(os.path.join(cfg.data_dir, 'wesad_combined_data.pickle'), 'wb') as f:
                    pickle.dump(data_dict, f)

            except FileNotFoundError:
                print("FileNotFoundError: Invalid directory structure! Please make sure that /dataset exists.")

    def load_dataset(self, resample_factor: int = None) -> Dict[int, pd.DataFrame]:
        """
        Load preprocessed dataset from /dataset
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

    def get_sensor_combinations(self) -> List[List[str]]:
        """
        Get sensor-combinations
        :return: sensor-combinations
        """
        return SENSOR_COMBINATIONS

    def get_classes(self) -> List[str]:
        """
        Get classes ("baseline", "amusement", "stress")
        :return: List with all classes
        """
        return CLASSES

