from abc import ABC, abstractmethod

import numpy as np


class GeometryStrategy(ABC):
    @abstractmethod
    def apply_boundaries(self, old_x, old_y, new_x, new_y):
        pass

    @abstractmethod
    def draw(self, ax, x_lim, y_lim):
        """Рисует препятствия на переданных осях ax в границах x_lim, y_lim"""
        pass


class ParallelLinesGeometry(GeometryStrategy):
    def __init__(self, barrier_dist=10.0, hole_size=5.0):
        self.barrier_dist = barrier_dist
        self.hole_size = hole_size

    def apply_boundaries(self, old_x, old_y, new_x, new_y):
        # (Весь код apply_boundaries остается без изменений, как в прошлом шаге)
        # ... (скопируй логику коллизий сюда) ...
        # Для краткости я его свернул, но он должен быть тут.

        # --- (Копия логики из предыдущего сообщения) ---
        idx_old = np.floor(old_y / self.barrier_dist)
        idx_new = np.floor(new_y / self.barrier_dist)
        crossing_mask = idx_old != idx_new

        if not np.any(crossing_mask):
            return new_x, new_y

        moving_up = new_y > old_y
        barrier_levels = np.zeros_like(old_y)
        mask_up = crossing_mask & moving_up
        barrier_levels[mask_up] = (idx_old[mask_up] + 1) * self.barrier_dist
        mask_down = crossing_mask & (~moving_up)
        barrier_levels[mask_down] = idx_old[mask_down] * self.barrier_dist

        barrier_indices = np.round(barrier_levels / self.barrier_dist).astype(int)
        L = self.hole_size * 4.0

        row_offsets = np.zeros_like(barrier_indices, dtype=float)
        is_odd_barrier = barrier_indices % 2 != 0
        row_offsets[is_odd_barrier] = L / 2.0

        shifted_x = new_x - row_offsets
        k = np.round(shifted_x / L)
        hole_center = k * L + row_offsets
        dist_to_hole = np.abs(new_x - hole_center)
        hit_hole = dist_to_hole <= (self.hole_size / 2.0)

        to_reflect = crossing_mask & (~hit_hole)
        final_y = new_y.copy()
        final_y[to_reflect] = 2 * barrier_levels[to_reflect] - new_y[to_reflect]
        return new_x, final_y

    def draw(self, ax, x_lim, y_lim):
        """
        Рисуем линии с разрывами (дырками).
        """
        ymin, ymax = y_lim
        xmin, xmax = x_lim

        # Определяем диапазон линий, видимых на экране
        min_k = int(np.floor(ymin / self.barrier_dist))
        max_k = int(np.ceil(ymax / self.barrier_dist))

        L = self.hole_size * 4.0  # Период

        # Рисуем каждую линию
        for k in range(min_k, max_k + 1):
            if k == 0:
                continue  # Обычно в нуле нет стены, если это центр

            y_pos = k * self.barrier_dist

            # Сдвиг (шахматный порядок)
            offset = (L / 2.0) if (k % 2 != 0) else 0.0

            # Находим дырки, которые попадают в кадр по X
            # Нам нужно нарисовать сегменты СТЕН между дырками.
            # Стена идет от (Center_i + R_hole) до (Center_i+1 - R_hole)

            start_n = int(np.floor((xmin - offset) / L)) - 1
            end_n = int(np.ceil((xmax - offset) / L)) + 1

            for n in range(start_n, end_n):
                center = n * L + offset
                # Координаты стены (сегмент справа от текущей дырки)
                wall_start_x = center + self.hole_size / 2.0
                wall_end_x = (center + L) - self.hole_size / 2.0

                # Оптимизация: не рисовать, если далеко за границей
                if wall_end_x < xmin or wall_start_x > xmax:
                    continue

                ax.plot(
                    [wall_start_x, wall_end_x],
                    [y_pos, y_pos],
                    color="black",
                    linewidth=1.5,
                    alpha=0.7,
                )


class GeometryFactory:
    @staticmethod
    def create(geo_type, **kwargs):
        if geo_type == "parallel":
            dist = kwargs.get("barrier_dist", 10.0)
            hole = kwargs.get("hole_size", 5.0)
            return ParallelLinesGeometry(barrier_dist=dist, hole_size=hole)
        else:
            raise ValueError(f"Unknown: {geo_type}")
