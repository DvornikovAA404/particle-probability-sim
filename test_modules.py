import matplotlib.pyplot as plt
import numpy as np

from analytics import PhysicsAnalyzer
from plotting import SimulationPlotter
from simulation import SimulationEngine


def run_test():
    print("Initializing scientific simulation...")

    # --- ПАРАМЕТРЫ ЭКСПЕРИМЕНТА ---
    # Тип геометрии: 'parallel', 'circle', 'random' или 'empty'
    geo_type = "parallel"

    sim = SimulationEngine(
        num_trajectories=5000,  # Количество частиц (для статистики)
        num_steps=10000,  # Длительность симуляции в шагах
        movement_type="normal",
        geometry_type=geo_type,
        barrier_dist=20.0,  # Расстояние между барьерами или кольцами
        hole_size=8.0,  # Размер отверстия или препятствия
    )
    sim.history_step = 10  # Шаг записи истории (оптимизация памяти)

    print(f"Running simulation [Geometry: {geo_type}]...")
    sim.run()

    # --- АНАЛИТИКА ---
    print("\n--- ANALYTICS REPORT ---")
    analyzer = PhysicsAnalyzer()

    # 1. Расчет диффузии и извилистости
    slope, r2 = analyzer.calculate_diffusion_coefficient(sim)

    # Теоретический D0 = 1.0 для свободного пространства (при текущей физике)
    theoretical_slope = 1.0
    # Извилистость (Tortuosity): во сколько раз среда замедляет диффузию
    tortuosity = theoretical_slope / slope

    print(f"Diffusion Slope (D_eff): {slope:.4f}")
    print(f"Linearity (R^2):         {r2:.4f}")
    print(f"Tortuosity (τ):          {tortuosity:.4f}")

    if tortuosity < 1.0:
        print("Note: τ < 1 means super-diffusion (unlikely for barriers)")

    # 2. Расчет концентрации
    dr = 4.0  # Шаг радиуса для построения профиля концентрации
    r_centers, counts, density = analyzer.calculate_radial_concentration(sim, dr=dr)

    # --- ОТРИСОВКА ---
    print("Plotting results...")

    # 1. Карта траекторий и геометрии
    fig1 = SimulationPlotter.plot_trajectories(
        sim, title=f"Map: {geo_type} (τ = {tortuosity:.2f})", num_trajectories=50
    )

    # 2. График MSD (среднеквадратичное смещение)
    fig2 = SimulationPlotter.plot_statistics(sim)

    # 3. Радиальный профиль концентрации
    fig3 = SimulationPlotter.plot_concentration_profile(
        r_centers, density, title=f"Radial Concentration Profile (Geo: {geo_type})"
    )

    plt.show()


def test_all_geometries():
    """
    Тестовое сравнение 4-х типов геометрии на одном холсте.
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

        # Расчет границ графика
        final_x, final_y = sim.x, sim.y
        max_val = max(np.max(np.abs(final_x)), np.max(np.abs(final_y)), 10.0)
        limit = np.ceil(max_val / 10.0) * 10.0
        if geo == "empty" and limit < 100:
            limit = 100

        # Отрисовка стен
        if hasattr(sim, "geo_strategy"):
            sim.geo_strategy.draw(ax, (-limit, limit), (-limit, limit))

        # Отрисовка частиц
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
