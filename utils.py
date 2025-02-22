import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Вычисляет среднюю квадратичную ошибку (MSE) между y_true и y_pred.
    MSE = (1/N) * sum((y_pred - y_true)^2)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_pred - y_true) ** 2)
