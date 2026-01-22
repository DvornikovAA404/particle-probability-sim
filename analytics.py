import numpy as np
from scipy.stats import linregress


class PhysicsAnalyzer:
    @staticmethod
    def calculate_diffusion_coefficient(sim):
        """
        Вычисляет коэффициент диффузии D_eff как наклон графика MSD (<r^2>).
        Возвращает: slope (наклон), r2_score (коэффициент детерминации).
        """
        X = np.array(sim.history_x)
        Y = np.array(sim.history_y)

        # Квадрат смещения от начальной точки для каждой частицы
        R2 = (X - X[0]) ** 2 + (Y - Y[0]) ** 2

        # Средний квадрат смещения (MSD) по ансамблю
        mean_r2 = np.mean(R2, axis=1)

        steps = np.arange(len(mean_r2)) * sim.history_step

        # Отбрасываем первую половину симуляции (переходный процесс)
        start_idx = len(steps) // 2

        # Линейная регрессия: MSD = 4 * D * t
        slope, intercept, r_value, p_value, std_err = linregress(
            steps[start_idx:], mean_r2[start_idx:]
        )

        return slope, r_value**2

    @staticmethod
    def calculate_radial_concentration(sim, dr=5.0):
        """
        Вычисляет профиль радиальной концентрации C(r).
        dr: ширина кольца пространственной дискретизации.
        """
        final_x = sim.x
        final_y = sim.y

        r = np.sqrt(final_x**2 + final_y**2)

        # Определение границ кольцевых зон (бинов)
        max_r = np.max(r)
        if max_r == 0:
            max_r = dr
        bins = np.arange(0, max_r + dr, dr)

        # Гистограмма распределения частиц по радиусу
        counts, bin_edges = np.histogram(r, bins=bins)

        # Координаты центров колец для графика
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Площадь каждого кольца: S = pi * (R_out^2 - R_in^2)
        r_inner = bin_edges[:-1]
        r_outer = bin_edges[1:]
        areas = np.pi * (r_outer**2 - r_inner**2)

        # Концентрация = Число частиц / Площадь кольца
        density = counts / areas

        return centers, counts, density
