from abc import ABC, abstractmethod

import numpy as np

# --- STRATEGY PATTERN ---


class MovementStrategy(ABC):
    @abstractmethod
    def get_displacement(self, num_particles, dt=1.0):
        pass


class NormalMovement(MovementStrategy):
    """
    Обычное движение.
    """

    def get_displacement(self, num_particles, dt=1.0):
        scale = np.sqrt(0.5)
        dx = np.random.normal(loc=0.0, scale=scale, size=num_particles) * np.sqrt(dt)
        dy = np.random.normal(loc=0.0, scale=scale, size=num_particles) * np.sqrt(dt)
        return dx, dy


class MaxwellMovement(MovementStrategy):
    """
    Движение по Максвеллу.
    """

    def __init__(self, beta=0.5):
        self.beta = beta

    def get_displacement(self, num_particles, dt=1.0):
        # Если мы хотим сопоставимый масштаб с Normal, нужно учитывать нормировку
        # Но у Максвелла свой физический смысл (температура).
        # Пока оставим beta как параметр масштаба.

        # Генерируем компоненты скорости
        vx = np.random.normal(0, self.beta, num_particles)
        vy = np.random.normal(0, self.beta, num_particles)
        vz = np.random.normal(0, self.beta, num_particles)
        speed = np.sqrt(vx**2 + vy**2 + vz**2)

        angle = np.random.uniform(0, 2 * np.pi, num_particles)

        dx = speed * np.cos(angle) * np.sqrt(dt)
        dy = speed * np.sin(angle) * np.sqrt(dt)
        return dx, dy


# --- FACTORY ---


class StrategyFactory:
    @staticmethod
    def create(movement_type, **kwargs):
        if movement_type == "normal":
            return NormalMovement()
        elif movement_type == "maxwell":
            beta = kwargs.get("beta", 0.5)
            return MaxwellMovement(beta=beta)
        else:
            raise ValueError(f"Unknown movement type: {movement_type}")


class SimulationEngine:
    def __init__(
        self, num_trajectories=10000, num_steps=10000, movement_type="normal", **kwargs
    ):
        self.num_trajectories = num_trajectories
        self.num_steps = num_steps

        # Используем фабрику для получения стратегии
        self.strategy = StrategyFactory.create(movement_type, **kwargs)

        # Состояние системы
        self.x = np.zeros(self.num_trajectories)
        self.y = np.zeros(self.num_trajectories)

        # История (оптимизированная)
        self.history_step = 100
        self.history_x = []
        self.history_y = []

    def run(self):
        print(f"Simulation started. Strategy: {self.strategy.__class__.__name__}")

        # Сохраняем старт
        self.history_x.append(self.x.copy())
        self.history_y.append(self.y.copy())

        for step in range(1, self.num_steps + 1):
            # ДЕЛЕГИРУЕМ расчет смещения стратегии
            dx, dy = self.strategy.get_displacement(self.num_trajectories)

            # Обновляем координаты
            self.x += dx
            self.y += dy

            if step % self.history_step == 0:
                self.history_x.append(self.x.copy())
                self.history_y.append(self.y.copy())

        print("Simulation finished.")
