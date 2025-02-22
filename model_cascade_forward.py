import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_cascade_forward(hidden_layers):
    """
    Створює Cascade Forward нейромережу.
    Вхідний шар з'єднується з кожним прихованим шаром.
    """
    inputs = keras.Input(shape=(2,))
    x = inputs
    
    # Додаємо всі приховані шари та з'єднуємо з входом
    for units in hidden_layers:
        x = layers.Concatenate()([x, inputs])  # З'єднання входу з кожним шаром
        x = layers.Dense(units, activation='relu')(x)
    
    outputs = layers.Dense(1)(x)  # Вихідний шар (1 нейрон)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    
    return model
