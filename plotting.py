import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


class SimulationPlotter:
    @staticmethod
    def _get_round_limit(value, step=10.0):
        """
        Округляет значение вверх до ближайшего числа, кратного step.
        """
        if value == 0:
            return step
        return np.ceil(value / step) * step

    @staticmethod
    def plot_trajectories(sim, title="Simulation Results", num_trajectories=20):
        """
        Визуализация траекторий частиц и препятствий с адаптивным масштабом.
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # --- 1. Автоматическое определение границ графика ---
        all_x = sim.x
        all_y = sim.y

        # Максимальное удаление от центра
        max_abs_x = np.max(np.abs(all_x))
        max_abs_y = np.max(np.abs(all_y))
        global_max = max(max_abs_x, max_abs_y)

        # Адаптивный шаг сетки: 50 для больших масштабов, 10 для малых
        step = 50.0 if global_max > 50 else 10.0
        limit = SimulationPlotter._get_round_limit(global_max, step=step)

        # Симметричные квадратные границы
        xlim = (-limit, limit)
        ylim = (-limit, limit)

        # --- 2. Отрисовка геометрии (стен/препятствий) ---
        if hasattr(sim, "geo_strategy"):
            sim.geo_strategy.draw(ax, xlim, ylim)

        # --- 3. Отрисовка траекторий ---
        hist_x = np.array(sim.history_x)
        hist_y = np.array(sim.history_y)

        colors = cm.rainbow(np.linspace(0, 1, num_trajectories))
        count = min(num_trajectories, sim.num_trajectories)

        for i in range(count):
            # Линия траектории
            ax.plot(hist_x[:, i], hist_y[:, i], lw=1, alpha=0.7, color=colors[i])
            # Точка финиша
            ax.scatter(
                hist_x[-1, i],
                hist_y[-1, i],
                s=15,
                color=colors[i],
                marker="o",
                edgecolors="white",
                linewidth=0.5,
            )

        # Оформление
        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_aspect("equal")  # Фиксация пропорций (круг остается кругом)
        ax.grid(True, alpha=0.2, linestyle="--")

        return fig

    @staticmethod
    def plot_statistics(sim):
        """
        График среднеквадратичного смещения (MSD) <r^2> от времени.
        Используется для проверки линейности диффузии.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        X = np.array(sim.history_x)
        Y = np.array(sim.history_y)

        # MSD для ансамбля частиц
        R2 = (X - X[0]) ** 2 + (Y - Y[0]) ** 2
        mean_r2 = np.mean(R2, axis=1)
        steps = np.arange(len(mean_r2)) * sim.history_step

        ax.plot(steps, mean_r2, label="Simulation <r^2>", color="blue", lw=2)

        # Теоретический эталон для свободного пространства (наклон = 1)
        ax.plot(
            steps, steps, "k--", alpha=0.5, label="Theoretical Free Space (Slope=1)"
        )

        ax.set_title("Mean Squared Displacement vs Time")
        ax.set_xlabel("Steps")
        ax.set_ylabel("<r^2>")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    @staticmethod
    def plot_concentration_profile(centers, density, title="Concentration Profile"):
        """
        График радиальной концентрации C(r).
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(
            centers,
            density,
            "o-",
            color="purple",
            lw=2,
            markersize=4,
            label="Measured C(r)",
        )

        ax.fill_between(centers, density, alpha=0.3, color="purple")

        ax.set_title(title)
        ax.set_xlabel("Radius r (Distance from center)")
        ax.set_ylabel("Concentration C(r) [Particles / Area]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig
