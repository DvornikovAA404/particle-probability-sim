import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


class SimulationPlotter:
    @staticmethod
    def _get_round_limit(value, step=10.0):
        """
        Округляет значение вверх до ближайшего числа, кратного step.
        Пример: value=43, step=10 -> 50.
        """
        if value == 0:
            return step
        return np.ceil(value / step) * step

    @staticmethod
    def plot_trajectories(sim, title="Simulation Results", num_trajectories=20):
        """
        Рисует траектории и геометрию с красивыми "круглыми" границами.
        """
        fig, ax = plt.subplots(figsize=(10, 10))  # Квадратный размер фигуры

        # 1. Анализ границ (Auto-scaling to round numbers)
        # Берем финальные координаты всех частиц
        all_x = sim.x
        all_y = sim.y

        # Находим максимальное удаление от центра по любой оси
        max_abs_x = np.max(np.abs(all_x))
        max_abs_y = np.max(np.abs(all_y))
        global_max = max(max_abs_x, max_abs_y)

        # Определяем шаг округления в зависимости от масштаба
        # Если разлет < 50 -> шаг 10. Если > 50 -> шаг 50.
        step = 50.0 if global_max > 50 else 10.0

        limit = SimulationPlotter._get_round_limit(global_max, step=step)

        # Делаем границы симметричными и квадратными
        xlim = (-limit, limit)
        ylim = (-limit, limit)

        # 2. Рисуем Геометрию
        # Передаем вычисленные красивые границы, чтобы геометрия знала, сколько рисовать
        if hasattr(sim, "geo_strategy"):
            sim.geo_strategy.draw(ax, xlim, ylim)

        # 3. Рисуем Траектории
        hist_x = np.array(sim.history_x)
        hist_y = np.array(sim.history_y)

        colors = cm.rainbow(np.linspace(0, 1, num_trajectories))

        # Рисуем только N случайных (или первых) траекторий
        count = min(num_trajectories, sim.num_trajectories)
        for i in range(count):
            ax.plot(hist_x[:, i], hist_y[:, i], lw=1, alpha=0.7, color=colors[i])
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
        ax.set_aspect("equal")  # Важно: чтобы круги были кругами, а квадраты квадратами
        ax.grid(True, alpha=0.2, linestyle="--")

        return fig

    @staticmethod
    def plot_statistics(sim):
        """
        Рисует валидационный график <r^2>
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        X = np.array(sim.history_x)
        Y = np.array(sim.history_y)

        # Квадрат смещения
        R2 = (X - X[0]) ** 2 + (Y - Y[0]) ** 2
        mean_r2 = np.mean(R2, axis=1)
        steps = np.arange(len(mean_r2)) * sim.history_step

        ax.plot(steps, mean_r2, label="Simulation <r^2>", color="blue", lw=2)

        # Теоретическая справка
        ax.plot(
            steps, steps, "k--", alpha=0.5, label="Theoretical Free Space (Slope=1)"
        )

        ax.set_title("Mean Squared Displacement vs Time")
        ax.set_xlabel("Steps")
        ax.set_ylabel("<r^2>")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig
