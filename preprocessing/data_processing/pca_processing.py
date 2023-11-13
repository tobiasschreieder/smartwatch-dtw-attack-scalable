from preprocessing.data_processing.data_processing import DataProcessing

from typing import Dict
import pandas as pd
from sklearn.decomposition import PCA


class PcaProcessing(DataProcessing):
    """
    Class to process dataset with method Principal Component Analysis (PCA)
    """
    def __init__(self):
        """
        Init pca-processing
        """
        super().__init__()

        self.name = "PCA-Processing"

    @classmethod
    def process_data(cls, data_dict: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """
        Run pca-processing for given dataset
        :param data_dict: Dictionary with dataset
        :return: Dictionary with processed data
        """
        pca_components = 1  # Specify number of components (PCA)

        pca_dict = dict()

        pca_columns = list()
        for i in range(1, pca_components + 1):
            pca_columns.append("pca-" + str(i))

        for subject in data_dict:
            label = data_dict[subject].label
            data = data_dict[subject].drop(columns=["label"])
            data = (data - data.mean()) / data.std()
            pca = PCA(n_components=pca_components)
            data_pca = pca.fit_transform(data)
            data_pca = pd.DataFrame(data_pca, columns=pca_columns).assign(label=label)
            pca_dict.setdefault(subject, data_pca)

        return pca_dict
