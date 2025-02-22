import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_elman(hidden_layers):
    """
    Эмуляция Elman RNN через SimpleRNN.
    Каждый элемент hidden_layers задает количество нейронов в соответствующем RNN-слое.
    Если слоев более одного, для первых слоев используется return_sequences=True.
    """
    model = keras.Sequential()
    model.add(layers.SimpleRNN(
        hidden_layers[0],
        activation='tanh',
        return_sequences=(len(hidden_layers) > 1),
        input_shape=(1, 2)  # (timesteps=1, features=2)
    ))
    
    for i in range(1, len(hidden_layers)):
        model.add(layers.SimpleRNN(
            hidden_layers[i],
            activation='tanh',
            return_sequences=(i < len(hidden_layers) - 1)
        ))
    
    model.add(layers.Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model
