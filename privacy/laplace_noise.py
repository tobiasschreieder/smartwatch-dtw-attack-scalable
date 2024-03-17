import numpy as np


def create_noisy_data(data: np.ndarray, noise_multiplier: float = None, noise_type: str = "laplace",
                      clip_max: bool = False) -> np.array:
    """
    Add noise to data based on multiplier or epsilon
    :param data: Data to noise, two dimensional np.array of signals e.g. (6,100) for six signals and 100 data points
    :param noise_multiplier: Scale parameter of Laplace distribution -> Specify level of noise >= 0.0
    :param noise_type: Noise distribution ("laplace" or "gaussian")
    :param clip_max: Clipping to max of each signal for dp, if false clip to 1 as sensitivity of similarity func.
    :return: noised data
    """
    noisy_data = np.empty(data.shape)
    for idx, signal_data in enumerate(data):
        # 1 as sensitivity of similarity func or max of each signal as clip for DP
        clip = 1 if not clip_max else np.max(signal_data)
        if noise_type == "laplace":
            noisy_data[idx] = signal_data + np.random.laplace(scale=clip*noise_multiplier, size=signal_data.shape)
        elif noise_type == "gaussian":
            noisy_data[idx] = signal_data + np.random.normal(scale=clip*noise_multiplier, size=signal_data.shape)
    return noisy_data
