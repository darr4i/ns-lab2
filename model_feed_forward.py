import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_feed_forward(hidden_layers):
    """
    Створює Feed Forward нейромережу з вказаною кількістю нейронів у прихованих шарах.
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(2,)))  # Вхідний шар на 2 нейрони (x, y)
    
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu'))
    
    model.add(layers.Dense(1))  # Вихідний шар (1 нейрон)
    
    # Додаємо loss='mse', щоб зберігати loss у history
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    
    return model
