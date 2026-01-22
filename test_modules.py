import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from simulation import SimulationEngine


def analyze_results(sim, title):
    """
    Возвращает данные для графиков: время, <r^2>, rms
    """
    # Превращаем историю в матрицы
    X = np.array(sim.history_x)
    Y = np.array(sim.history_y)

    # Считаем R^2 для каждой частицы на каждом шаге
    # (X - X_start)^2 + ... (считаем, что старт в 0, иначе вычесть X[0])
    R2 = (X - X[0]) ** 2 + (Y - Y[0]) ** 2

    # Среднее по ансамблю
    mean_r2 = np.mean(R2, axis=1)
    rms = np.sqrt(mean_r2)
    steps = np.arange(len(mean_r2)) * sim.history_step

    return steps, mean_r2, rms


def plot_trajectories(ax, sim, num_to_plot=20):
    """Рисует цветные спагетти траекторий"""
    hist_x = np.array(sim.history_x)
    hist_y = np.array(sim.history_y)
    colors = cm.rainbow(np.linspace(0, 1, num_to_plot))

    for i in range(num_to_plot):
        ax.plot(hist_x[:, i], hist_y[:, i], lw=1, color=colors[i], alpha=0.7)
        ax.plot(hist_x[-1, i], hist_y[-1, i], "ko", markersize=2)  # Конечная точка

    ax.set_title("Trajectories (subset)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")


def run_full_test():
    # Параметры
    n_part = 5000
    n_steps = 10000

    print("--- Running Normal Simulation ---")
    sim = SimulationEngine(
        num_trajectories=n_part, num_steps=n_steps, movement_type="normal"
    )
    sim.history_step = 10  # Пишем часто для гладких графиков
    sim.run()

    # Анализ
    steps, mean_r2, rms = analyze_results(sim, "Normal")

    # --- ОТРИСОВКА ---
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        f"Validation: Normal Motion (N={n_part}, Steps={n_steps})", fontsize=16
    )

    # 1. Траектории
    ax1 = fig.add_subplot(2, 2, 1)
    plot_trajectories(ax1, sim)

    # 2. График <r^2>
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(steps, mean_r2, label="Simulation <r^2>", color="blue", linewidth=2)
    ax2.plot(
        steps, steps, "k--", label="Theoretical N (Slope=1)", linewidth=2
    )  # Теория y=x
    ax2.set_title("Mean Squared Displacement")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("<r^2>")
    ax2.legend()
    ax2.grid(True)

    # 3. График RMS
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(steps, rms, label="Simulation RMS", color="green", linewidth=2)
    ax3.plot(steps, np.sqrt(steps), "k--", label="Theoretical sqrt(N)", linewidth=2)
    ax3.set_title("Root Mean Square Displacement")
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Distance")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_full_test()
