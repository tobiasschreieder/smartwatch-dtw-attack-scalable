from preprocessing.data_processing.data_processing import DataProcessing

import numpy as np
import pandas as pd
from dtaidistance import dtw_barycenter
from typing import Dict


class DbaProcessing(DataProcessing):
    """
    Class to process dataset with method Dynamic Time Warping Barycenter Averaging
    """
    def __init__(self):
        """
        Init DBA-processing
        """
        super().__init__()

        self.name = "DBA-Processing"

    @classmethod
    def process_data(cls, data_dict: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """
        Run Dynamic Time Warping Barycenter Averaging for given dataset (averaging over sensors per subject)
        :param data_dict: Dictionary with dataset
        :return: Dictionary with averaged signals
        """
        dba_dict = dict()
        for subject in data_dict:
            series_wesad = list()
            for sensor in data_dict[subject]:
                if sensor != "label":
                    sensor_data = data_dict[subject][sensor]
                    series_wesad.append(sensor_data)

            series_wesad = np.array(series_wesad)
            average_series = dtw_barycenter.dba(s=series_wesad, c=series_wesad[0])
            labels = data_dict[subject]["label"].to_frame()
            average_dba = pd.DataFrame(average_series, columns=["dba"])
            average_dba = pd.concat([average_dba, labels], axis=1)
            dba_dict.setdefault(subject, average_dba)

        return dba_dict
