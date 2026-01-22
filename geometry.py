from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class GeometryStrategy(ABC):
    """
    Абстрактная стратегия геометрии.
    """

    @abstractmethod
    def apply_boundaries(self, old_x, old_y, new_x, new_y):
        pass

    @abstractmethod
    def draw(self, ax, x_lim, y_lim):
        pass


# --- 1. ПАРАЛЛЕЛЬНЫЕ ЛИНИИ (Классика) ---
class ParallelLinesGeometry(GeometryStrategy):
    def __init__(self, barrier_dist=10.0, hole_size=5.0):
        self.barrier_dist = barrier_dist
        self.hole_size = hole_size

    def apply_boundaries(self, old_x, old_y, new_x, new_y):
        # Логика полос (как была)
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

        # Дырки
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

        # Отражение
        to_reflect = crossing_mask & (~hit_hole)
        final_y = new_y.copy()
        final_y[to_reflect] = 2 * barrier_levels[to_reflect] - new_y[to_reflect]

        return new_x, final_y

    def draw(self, ax, x_lim, y_lim):
        ymin, ymax = y_lim
        xmin, xmax = x_lim
        min_k = int(np.floor(ymin / self.barrier_dist))
        max_k = int(np.ceil(ymax / self.barrier_dist))
        L = self.hole_size * 4.0

        for k in range(min_k, max_k + 1):
            if k == 0:
                continue
            y_pos = k * self.barrier_dist
            offset = (L / 2.0) if (k % 2 != 0) else 0.0

            start_n = int(np.floor((xmin - offset) / L)) - 1
            end_n = int(np.ceil((xmax - offset) / L)) + 1

            for n in range(start_n, end_n):
                center = n * L + offset
                wall_start_x = center + self.hole_size / 2.0
                wall_end_x = (center + L) - self.hole_size / 2.0

                if wall_end_x < xmin or wall_start_x > xmax:
                    continue
                ax.plot(
                    [wall_start_x, wall_end_x], [y_pos, y_pos], color="black", lw=1.5
                )


# --- 2. ПУСТОЕ ПРОСТРАНСТВО (Без пор) ---
class EmptyGeometry(GeometryStrategy):
    def apply_boundaries(self, old_x, old_y, new_x, new_y):
        # Ничего не делаем, частицы летят свободно
        return new_x, new_y

    def draw(self, ax, x_lim, y_lim):
        # Ничего не рисуем
        pass


# --- 3. КОНЦЕНТРИЧЕСКИЕ КРУГИ (Radial/Tangential) ---
class ConcentricCirclesGeometry(GeometryStrategy):
    def __init__(self, radius_step=20.0, hole_size=10.0):
        self.radius_step = radius_step
        self.hole_size = hole_size

    def apply_boundaries(self, old_x, old_y, new_x, new_y):
        # 1. Переходим в полярные координаты (r, theta)
        r_old = np.hypot(old_x, old_y)
        r_new = np.hypot(new_x, new_y)

        # 2. Логика барьеров по радиусу (аналогично линиям по Y)
        idx_old = np.floor(r_old / self.radius_step)
        idx_new = np.floor(r_new / self.radius_step)

        crossing_mask = idx_old != idx_new
        if not np.any(crossing_mask):
            return new_x, new_y

        # Определяем радиус барьера
        moving_out = r_new > r_old
        barrier_r = np.zeros_like(r_old)

        mask_out = crossing_mask & moving_out
        barrier_r[mask_out] = (idx_old[mask_out] + 1) * self.radius_step

        mask_in = crossing_mask & (~moving_out)
        barrier_r[mask_in] = idx_old[mask_in] * self.radius_step

        # 3. Проверка дырок (по углу theta)
        # theta = arctan2(y, x) -> [-pi, pi]
        theta = np.arctan2(new_y, new_x)

        # Длина окружности барьера C = 2 * pi * R
        # Период дырок по дуге (L)
        L = self.hole_size * 4.0

        # Переводим угол в длину дуги: arc_pos = theta * R
        arc_pos = theta * barrier_r

        # Сдвиг дырок: четные кольца - 0, нечетные - полпериода
        barrier_indices = np.round(barrier_r / self.radius_step).astype(int)
        offsets = np.zeros_like(barrier_indices, dtype=float)
        offsets[barrier_indices % 2 != 0] = L / 2.0

        shifted_arc = arc_pos - offsets
        k = np.round(shifted_arc / L)
        hole_arc_center = k * L + offsets

        # Расстояние по дуге до центра дырки
        dist_arc = np.abs(arc_pos - hole_arc_center)

        hit_hole = dist_arc <= (self.hole_size / 2.0)

        # 4. Отражение (по радиусу)
        to_reflect = crossing_mask & (~hit_hole)

        final_r = r_new.copy()
        # Зеркальное отражение r относительно barrier_r
        final_r[to_reflect] = 2 * barrier_r[to_reflect] - r_new[to_reflect]

        # Обратно в декартовы (угол не меняем при отражении от кольца)
        # Важно: используем theta от new координат
        final_x = final_r * np.cos(theta)
        final_y = final_r * np.sin(theta)

        # Для тех, кто не отразился, оставляем new_x/y (они прошли или не пересекали)
        # Но так как мы меняли final_r, надо собрать результат
        # Проще так: берем исходные new_x/y, и меняем только у отраженных
        out_x = new_x.copy()
        out_y = new_y.copy()

        out_x[to_reflect] = final_x[to_reflect]
        out_y[to_reflect] = final_y[to_reflect]

        return out_x, out_y

    def draw(self, ax, x_lim, y_lim):
        max_dim = max(abs(x_lim[1]), abs(y_lim[1]))
        max_k = int(np.ceil(max_dim / self.radius_step))
        L = self.hole_size * 4.0

        for k in range(1, max_k + 1):
            r = k * self.radius_step
            circumference = 2 * np.pi * r

            # Сколько дырок влезает?
            n_holes = int(circumference / L)
            if n_holes == 0:
                n_holes = 1

            # Реальный угловой шаг (чтобы замкнулось красиво)
            d_theta = (2 * np.pi) / n_holes

            offset_angle = (d_theta / 2.0) if (k % 2 != 0) else 0.0

            # Угловой размер дырки
            hole_angle = self.hole_size / r

            # Рисуем дуги (стены) между дырками
            for i in range(n_holes):
                center_angle = i * d_theta + offset_angle

                # Стена начинается после дырки и заканчивается перед следующей
                # Дырка от center - hole/2 до center + hole/2
                # Стена идет от (center + hole/2) до (next_center - hole/2)

                start_angle = center_angle + hole_angle / 2.0
                end_angle = (center_angle + d_theta) - hole_angle / 2.0

                # Переводим в градусы для matplotlib
                theta1 = np.degrees(start_angle)
                theta2 = np.degrees(end_angle)

                # Коррекция для Arc (matplotlib рисует против часовой)
                arc = matplotlib.patches.Arc(
                    (0, 0),
                    2 * r,
                    2 * r,
                    theta1=theta1,
                    theta2=theta2,
                    color="black",
                    lw=1.5,
                )
                ax.add_patch(arc)


import matplotlib.patches  # Нужен для рисования дуг


# --- 4. СЛУЧАЙНЫЕ ПРЕПЯТСТВИЯ (Random) ---
class RandomObstaclesGeometry(GeometryStrategy):
    def __init__(self, num_obstacles=50, obstacle_radius=5.0, field_size=200.0):
        self.num_obstacles = num_obstacles
        self.r_obs = obstacle_radius
        self.field_size = field_size

        # Генерируем препятствия один раз при создании
        # Чтобы карта не менялась каждый шаг
        self.centers_x = np.random.uniform(-field_size, field_size, num_obstacles)
        self.centers_y = np.random.uniform(-field_size, field_size, num_obstacles)

    def apply_boundaries(self, old_x, old_y, new_x, new_y):
        # Векторизированная проверка сложная (N частиц * M препятствий).
        # Для 10000 частиц и 50 препятствий это 500к операций, Python потянет.

        out_x = new_x.copy()
        out_y = new_y.copy()

        # Проходим по каждому препятствию (цикл по препятствиям, векторизация по частицам)
        for cx, cy in zip(self.centers_x, self.centers_y):
            # Расстояние от центра препятствия до частицы
            dx = out_x - cx
            dy = out_y - cy
            dist_sq = dx**2 + dy**2
            min_dist_sq = self.r_obs**2

            # Кто попал внутрь?
            mask_hit = dist_sq < min_dist_sq

            if np.any(mask_hit):
                # "Выталкиваем" частицу наружу (упругое отражение по нормали)
                # Нормаль n = (dx, dy) / dist
                # Новая позиция = Center + (Radius + epsilon) * n

                dist = np.sqrt(dist_sq[mask_hit])
                # Избегаем деления на 0 (если частица прямо в центре, пуляем рандомно)
                dist[dist == 0] = 0.001

                norm_x = dx[mask_hit] / dist
                norm_y = dy[mask_hit] / dist

                # Ставим на границу + чуть-чуть
                out_x[mask_hit] = cx + norm_x * (self.r_obs + 0.01)
                out_y[mask_hit] = cy + norm_y * (self.r_obs + 0.01)

        return out_x, out_y

    def draw(self, ax, x_lim, y_lim):
        for cx, cy in zip(self.centers_x, self.centers_y):
            circle = plt.Circle((cx, cy), self.r_obs, color="black", alpha=0.5)
            ax.add_patch(circle)

            # Опционально: крестик в центре
            # ax.plot(cx, cy, 'k+', markersize=2)


# --- ФАБРИКА ---
class GeometryFactory:
    @staticmethod
    def create(geo_type, **kwargs):
        if geo_type == "parallel":
            return ParallelLinesGeometry(
                barrier_dist=kwargs.get("barrier_dist", 10.0),
                hole_size=kwargs.get("hole_size", 5.0),
            )
        elif geo_type == "empty":
            return EmptyGeometry()
        elif geo_type == "circle":
            return ConcentricCirclesGeometry(
                radius_step=kwargs.get(
                    "barrier_dist", 20.0
                ),  # Используем barrier_dist как шаг радиуса
                hole_size=kwargs.get("hole_size", 10.0),
            )
        elif geo_type == "random":
            return RandomObstaclesGeometry(
                num_obstacles=kwargs.get("num_obstacles", 50),
                obstacle_radius=kwargs.get(
                    "hole_size", 5.0
                ),  # Используем hole_size как радиус препятствия
                field_size=200.0,  # Можно вынести в настройки
            )
        else:
            raise ValueError(f"Unknown geometry type: {geo_type}")
