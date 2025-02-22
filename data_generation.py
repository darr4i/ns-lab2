import numpy as np
from sklearn.model_selection import train_test_split

def generate_data(n_points=100, test_size=0.2, random_state=42):
    """
    Генерує дані для f(x,y) = 2x^2 + y^2 на відрізку [0, 10],
    а потім масштабує входи (діленням на 10) та виходи (діленням на 300),
    щоб привести їх до діапазону [0,1].
    Повертає (X_train, X_test, y_train, y_test).
    """
    # Генеруємо x та y з відрізку [0,10]
    x_vals = np.linspace(0, 10, n_points)
    y_vals = np.linspace(0, 10, n_points)

    # Створюємо сітку
    X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)

    # Обчислюємо f(x,y) = 2x^2 + y^2
    Z = 2 * (X_mesh ** 2) + (Y_mesh ** 2)

    # Формуємо дані: (N,2) для входів і (N,) для виходів
    X_data = np.column_stack((X_mesh.ravel(), Y_mesh.ravel()))
    y_data = Z.ravel()

    # Масштабування:
    X_data = X_data / 10.0    # входи тепер в [0,1]
    y_data = y_data / 300.0     # максимальне значення при x=y=10: f(10,10)=2*100+100=300

    # Розбиваємо на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

