import numpy as np
import matplotlib.pyplot as plt

from data_generation import generate_data
from utils import mean_squared_error

from model_feed_forward import build_feed_forward
from model_cascade_forward import build_cascade_forward
from model_elman import build_elman

def main():
    # 1) Генерація масштабованих даних для f(x,y) = 2x^2 + y^2 на відрізку [0,10]
    X_train, X_test, y_train, y_test = generate_data(n_points=100, test_size=0.2)

    # 2) Для Elman (RNN) перетворюємо дані у форму (N, 1, 2)
    X_train_rnn = X_train.reshape((X_train.shape[0], 1, 2))
    X_test_rnn  = X_test.reshape((X_test.shape[0], 1, 2))

    # Параметри навчання
    EPOCHS = 200
    BATCH_SIZE = 128

    # === Навчання моделей ===

    # Feed Forward
    model_ff_1 = build_feed_forward([10])
    history_ff_1 = model_ff_1.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    y_pred_ff_1 = model_ff_1.predict(X_test)
    mse_ff_1 = mean_squared_error(y_test, y_pred_ff_1)

    model_ff_2 = build_feed_forward([20])
    history_ff_2 = model_ff_2.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    y_pred_ff_2 = model_ff_2.predict(X_test)
    mse_ff_2 = mean_squared_error(y_test, y_pred_ff_2)

    # Cascade Forward
    model_cf_1 = build_cascade_forward([20])
    history_cf_1 = model_cf_1.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    y_pred_cf_1 = model_cf_1.predict(X_test)
    mse_cf_1 = mean_squared_error(y_test, y_pred_cf_1)

    model_cf_2 = build_cascade_forward([10, 10])
    history_cf_2 = model_cf_2.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    y_pred_cf_2 = model_cf_2.predict(X_test)
    mse_cf_2 = mean_squared_error(y_test, y_pred_cf_2)

    # Elman (RNN)
    model_elman_1 = build_elman([15])
    history_elman_1 = model_elman_1.fit(X_train_rnn, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    y_pred_elman_1 = model_elman_1.predict(X_test_rnn)
    mse_elman_1 = mean_squared_error(y_test, y_pred_elman_1)

    model_elman_2 = build_elman([5, 5, 5])
    history_elman_2 = model_elman_2.fit(X_train_rnn, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    y_pred_elman_2 = model_elman_2.predict(X_test_rnn)
    mse_elman_2 = mean_squared_error(y_test, y_pred_elman_2)

    # Вивід MSE
    print("=== Feed Forward ===")
    print(f"1 шар (10 нейронів): MSE = {mse_ff_1:.6f}")
    print(f"1 шар (20 нейронів): MSE = {mse_ff_2:.6f}")

    print("\n=== Cascade-Forward ===")
    print(f"1 шар (20 нейронів): MSE = {mse_cf_1:.6f}")
    print(f"2 шари (10+10 нейронів): MSE = {mse_cf_2:.6f}")

    print("\n=== Elman (RNN) ===")
    print(f"1 шар (15 нейронів): MSE = {mse_elman_1:.6f}")
    print(f"3 шари (5+5+5 нейронів): MSE = {mse_elman_2:.6f}")

    # 5) Графіки MSE від епох для всіх моделей
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Feed Forward
    axes[0].plot(history_ff_1.history['loss'], label='Feed Forward [10 нейронів]')
    axes[0].plot(history_ff_2.history['loss'], label='Feed Forward [20 нейронів]')
    axes[0].set_xlabel("Епоха")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Feed Forward: Залежність помилки від епох")
    axes[0].legend()

    # Cascade Forward
    axes[1].plot(history_cf_1.history['loss'], label='Cascade Forward [1x20]')
    axes[1].plot(history_cf_2.history['loss'], label='Cascade Forward [2x10]')
    axes[1].set_xlabel("Епоха")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Cascade Forward: Залежність помилки від епох")
    axes[1].legend()

    # Elman RNN
    axes[2].plot(history_elman_1.history['loss'], label='Elman RNN [1x15]')
    axes[2].plot(history_elman_2.history['loss'], label='Elman RNN [3x5]')
    axes[2].set_xlabel("Епоха")
    axes[2].set_ylabel("MSE")
    axes[2].set_title("Elman RNN: Залежність помилки від епох")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    # 6) 3D-графіки для всіх моделей
    from mpl_toolkits.mplot3d import Axes3D  

    x_plot = np.linspace(0, 10, 50)
    y_plot = np.linspace(0, 10, 50)
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
    Z_true = 2 * (X_plot ** 2) + (Y_plot ** 2)

    models = {
        "Feed Forward [10 нейронів]": model_ff_1,
        "Feed Forward [20 нейронів]": model_ff_2,
        "Cascade Forward [1x20]": model_cf_1,
        "Cascade Forward [2x10]": model_cf_2,
        "Elman RNN [1x15]": model_elman_1,
        "Elman RNN [3x5]": model_elman_2
    }

    fig = plt.figure(figsize=(12, 5 * len(models)))

    for i, (name, model) in enumerate(models.items(), start=1):
        X_input = np.column_stack((X_plot.ravel() / 10.0, Y_plot.ravel() / 10.0))
        if "Elman" in name:
            X_input = X_input.reshape(-1, 1, 2)
        
        Z_pred = model.predict(X_input).reshape(X_plot.shape) * 300.0

        ax_true = fig.add_subplot(len(models), 2, 2 * i - 1, projection='3d')
        ax_true.plot_surface(X_plot, Y_plot, Z_true, cmap='viridis')
        ax_true.set_title(f"Істинна функція для {name}")

        ax_pred = fig.add_subplot(len(models), 2, 2 * i, projection='3d')
        ax_pred.plot_surface(X_plot, Y_plot, Z_pred, cmap='viridis')
        ax_pred.set_title(f"Передбачення {name}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
