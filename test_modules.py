import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from simulation import SimulationEngine


def plot_trajectories(sim, title, num_to_plot=15):
    """
    Рисует полные пути для num_to_plot случайных частиц.
    """
    # Превращаем список массивов [step, particle] в матрицу [step, particle]
    hist_x = np.array(sim.history_x)
    hist_y = np.array(sim.history_y)

    # Выбираем цвета из палитры (радуга)
    colors = cm.rainbow(np.linspace(0, 1, num_to_plot))

    plt.title(title)

    # Рисуем пути для первых N частиц
    for i in range(num_to_plot):
        # hist_x[:, i] берет координату i-й частицы на всех шагах
        path_x = hist_x[:, i]
        path_y = hist_y[:, i]

        plt.plot(path_x, path_y, lw=1, color=colors[i], alpha=0.8)

        # Отметим начало (зеленый) и конец (красный)
        plt.plot(path_x[0], path_y[0], "g+", markersize=5)  # Start
        plt.plot(path_x[-1], path_y[-1], "ro", markersize=3)  # End

    plt.grid(True, alpha=0.3)
    plt.xlabel("X")
    plt.ylabel("Y")


def test_simulation():
    # Параметры теста
    n_part = 1000
    n_steps = 2000  # Побольше шагов, чтобы пути были длинными

    print("--- Running Normal Simulation ---")
    sim_normal = SimulationEngine(
        num_trajectories=n_part, num_steps=n_steps, movement_type="normal"
    )
    sim_normal.run()

    print("--- Running Maxwell Simulation ---")
    sim_maxwell = SimulationEngine(
        num_trajectories=n_part, num_steps=n_steps, movement_type="maxwell", beta=0.5
    )
    sim_maxwell.run()

    # Визуализация
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plot_trajectories(sim_normal, "Normal Motion Paths (Random Walk)")

    plt.subplot(1, 2, 2)
    plot_trajectories(sim_maxwell, "Maxwell Motion Paths")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_simulation()
