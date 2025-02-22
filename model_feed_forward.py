import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_feed_forward(hidden_layers):
    """
    Создает последовательную (Sequential) Feed Forward сеть с указанным списком hidden_layers,
    где каждый элемент списка = количество нейронов в слое.
    Предполагается вход размером (2,) и выход 1.
    """
    model = keras.Sequential()
    # Входной слой: 2 признака (x, y)
    model.add(layers.Input(shape=(2,)))
    
    # Добавляем скрытые слои
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu'))
    
    # Выходной слой (1 нейрон, линейная активация)
    model.add(layers.Dense(1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mse')
    return model
