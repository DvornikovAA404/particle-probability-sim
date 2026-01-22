from abc import ABC, abstractmethod

import numpy as np

from geometry import GeometryFactory

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
        self,
        num_trajectories=10000,
        num_steps=10000,
        movement_type="normal",
        geometry_type="parallel",  # Добавили geometry_type
        **kwargs,
    ):
        self.num_trajectories = num_trajectories
        self.num_steps = num_steps

        # 1. Стратегия Движения (Физика)
        self.move_strategy = StrategyFactory.create(movement_type, **kwargs)

        # 2. Стратегия Геометрии (Стены) - создаем через Фабрику
        # Передаем kwargs дальше, чтобы стратегия могла забрать barrier_dist и hole_size
        self.geo_strategy = GeometryFactory.create(geometry_type, **kwargs)

        self.x = np.zeros(self.num_trajectories)
        self.y = np.zeros(self.num_trajectories)

        self.history_step = 100
        self.history_x = []
        self.history_y = []

    def run(self):
        # Сохраняем начальное состояние
        self.history_x.append(self.x.copy())
        self.history_y.append(self.y.copy())

        print(
            f"Simulating: {self.num_trajectories} particles, "
            f"Movement: {self.move_strategy.__class__.__name__}, "
            f"Geometry: {self.geo_strategy.__class__.__name__}"
        )

        for step in range(1, self.num_steps + 1):
            # A. Считаем "желаемое" смещение (Physics)
            dx, dy = self.move_strategy.get_displacement(self.num_trajectories)

            # Предварительные координаты
            current_x = self.x
            current_y = self.y

            proposed_x = current_x + dx
            proposed_y = current_y + dy

            # B. Применяем ограничения геометрии (Geometry)
            # Вся логика коллизий и дырок теперь делегирована стратегии
            final_x, final_y = self.geo_strategy.apply_boundaries(
                current_x, current_y, proposed_x, proposed_y
            )

            # C. Обновляем состояние
            self.x = final_x
            self.y = final_y

            if step % self.history_step == 0:
                self.history_x.append(self.x.copy())
                self.history_y.append(self.y.copy())

        print("Done.")
