import matplotlib.pyplot as plt

from plotting import SimulationPlotter
from simulation import SimulationEngine


def run_test():
    # 1. Настройка и запуск
    print("Initializing simulation with Parallel Barriers...")
    sim = SimulationEngine(
        num_trajectories=5000,
        num_steps=10000,
        movement_type="normal",
        geometry_type="parallel",
        barrier_dist=20.0,  # Расстояние между стенами
        hole_size=10.0,  # Размер дырки
    )

    # Чтобы графики были плавными
    sim.history_step = 10

    print("Running...")
    sim.run()

    # 2. Отрисовка через новый модуль
    print("Plotting results...")

    # График траекторий + стены
    fig1 = SimulationPlotter.plot_trajectories(
        sim, title="Normal Diffusion in Porous Media", num_trajectories=30
    )

    # График статистики
    fig2 = SimulationPlotter.plot_statistics(sim)

    plt.show()


if __name__ == "__main__":
    run_test()
