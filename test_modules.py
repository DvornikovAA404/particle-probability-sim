import matplotlib.pyplot as plt
import numpy as np

from plotting import SimulationPlotter
from simulation import SimulationEngine


def run_test():
    """
    Обычный запуск одной симуляции.
    """
    print("Initializing simulation...")
    # Варианты: 'empty', 'parallel', 'circle', 'random'
    geo_type = "circle"

    sim = SimulationEngine(
        num_trajectories=5000,
        num_steps=10000,
        movement_type="normal",
        # Передаем выбранный тип
        geometry_type=geo_type,
        # --- Параметры для геометрии ---
        barrier_dist=20.0,  # Расстояние между стенами (или кольцами)
        hole_size=10.0,  # Размер дырки (или радиус препятствия в random)
        num_obstacles=50,  # Только для 'random'
    )

    sim.history_step = 10
    print(f"Running with geometry: {geo_type}...")
    sim.run()

    print("Plotting results...")

    # 1. Траектории и стены
    fig1 = SimulationPlotter.plot_trajectories(
        sim, title=f"Simulation: {geo_type.capitalize()}", num_trajectories=50
    )

    # 2. График статистики (r^2)
    fig2 = SimulationPlotter.plot_statistics(sim)

    plt.show()


def test_all_geometries():
    """
    Сравнение 4-х геометрий (для тестов).
    """
    geometries = ["empty", "parallel", "circle", "random"]
    params = {
        "num_trajectories": 300,
        "num_steps": 1000,
        "movement_type": "normal",
        "hole_size": 10.0,
        "barrier_dist": 25.0,
        "num_obstacles": 40,
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i, geo in enumerate(geometries):
        print(f"Testing: {geo}...")
        sim = SimulationEngine(geometry_type=geo, **params)
        sim.history_step = 10
        sim.run()

        ax = axes[i]
        final_x, final_y = sim.x, sim.y
        max_val = max(np.max(np.abs(final_x)), np.max(np.abs(final_y)), 10.0)
        limit = np.ceil(max_val / 10.0) * 10.0
        if geo == "empty" and limit < 100:
            limit = 100

        if hasattr(sim, "geo_strategy"):
            sim.geo_strategy.draw(ax, (-limit, limit), (-limit, limit))

        ax.scatter(final_x, final_y, s=5, alpha=0.6, c="blue")
        hx, hy = np.array(sim.history_x), np.array(sim.history_y)
        for p in range(min(10, sim.num_trajectories)):
            ax.plot(hx[:, p], hy[:, p], lw=0.5, alpha=0.4, c="red")

        ax.set_title(geo)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_test()
    # test_all_geometries()
