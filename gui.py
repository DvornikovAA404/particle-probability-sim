import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from analytics import PhysicsAnalyzer
from plotting import SimulationPlotter

# Импорт наших модулей
from simulation import SimulationEngine


class ScientificApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Particle Diffusion Simulator v1.0")
        self.geometry("1400x900")

        # Настройка стилей
        style = ttk.Style()
        style.theme_use("clam")  # Более современный вид, чем default

        # Главный контейнер (3 колонки)
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- ЛЕВАЯ ПАНЕЛЬ (Настройки) ---
        self.left_panel = ttk.LabelFrame(main_frame, text="Settings", padding=10)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.create_settings_widgets()

        # --- ПРАВАЯ ПАНЕЛЬ (Результаты) ---
        self.right_panel = ttk.LabelFrame(
            main_frame, text="Analytics Report", padding=10
        )
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        self.create_results_widgets()

        # --- ЦЕНТРАЛЬНАЯ ПАНЕЛЬ (Графики) ---
        self.center_panel = ttk.LabelFrame(main_frame, text="Visualization", padding=10)
        self.center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.create_plot_area()

    def create_settings_widgets(self):
        """Создает поля ввода настроек"""

        # Helper для создания полей
        def add_param(label_text, default_val, row):
            lbl = ttk.Label(self.left_panel, text=label_text)
            lbl.grid(row=row, column=0, sticky="w", pady=5)
            entry = ttk.Entry(self.left_panel, width=15)
            entry.insert(0, str(default_val))
            entry.grid(row=row, column=1, sticky="e", pady=5)
            return entry

        # Параметры
        self.inp_particles = add_param("Particles (N):", "2000", 0)
        self.inp_steps = add_param("Steps (t):", "1500", 1)
        self.inp_barrier = add_param("Barrier Dist:", "20.0", 2)
        self.inp_hole = add_param("Hole/Obs Size:", "8.0", 3)

        # Выпадающий список: Геометрия
        ttk.Label(self.left_panel, text="Geometry Type:").grid(
            row=4, column=0, sticky="w", pady=10
        )
        self.combo_geo = ttk.Combobox(
            self.left_panel,
            values=["parallel", "circle", "random", "empty"],
            state="readonly",
        )
        self.combo_geo.current(0)  # parallel
        self.combo_geo.grid(row=4, column=1, pady=10)

        # Выпадающий список: Движение
        ttk.Label(self.left_panel, text="Movement Type:").grid(
            row=5, column=0, sticky="w", pady=5
        )
        self.combo_move = ttk.Combobox(
            self.left_panel, values=["normal", "maxwell"], state="readonly"
        )
        self.combo_move.current(0)  # normal
        self.combo_move.grid(row=5, column=1, pady=5)

        # КНОПКА ЗАПУСКА
        # Делаем её большой и заметной
        self.btn_run = ttk.Button(
            self.left_panel, text="▶ RUN SIMULATION", command=self.run_simulation
        )
        self.btn_run.grid(row=6, column=0, columnspan=2, pady=20, sticky="ew")

    def create_results_widgets(self):
        """Создает текстовое поле для вывода цифр"""
        self.txt_results = tk.Text(
            self.right_panel,
            width=30,
            height=40,
            state="disabled",
            font=("Consolas", 10),
        )
        self.txt_results.pack(fill=tk.BOTH, expand=True)

    def create_plot_area(self):
        """Создает область для Matplotlib"""
        # Создаем фигуру заранее
        self.fig = plt.figure(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.center_panel)

        # Тулбар (кнопки сохранения, зума)
        toolbar = NavigationToolbar2Tk(self.canvas, self.center_panel)
        toolbar.update()

        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def log_result(self, text):
        """Вывод текста в правую панель"""
        self.txt_results.config(state="normal")
        self.txt_results.insert(tk.END, text + "\n")
        self.txt_results.see(tk.END)
        self.txt_results.config(state="disabled")

    def run_simulation(self):
        """Основная логика при нажатии кнопки"""
        try:
            # 1. Считываем данные
            n_part = int(self.inp_particles.get())
            n_steps = int(self.inp_steps.get())
            dist = float(self.inp_barrier.get())
            hole = float(self.inp_hole.get())
            geo = self.combo_geo.get()
            move = self.combo_move.get()

            # Очистка результатов
            self.txt_results.config(state="normal")
            self.txt_results.delete(1.0, tk.END)
            self.log_result("--- STARTING SIMULATION ---")
            self.log_result(f"Geo: {geo}, N={n_part}, T={n_steps}")
            self.update()  # Обновить интерфейс, чтобы текст появился

            # 2. Запуск движка
            sim = SimulationEngine(
                num_trajectories=n_part,
                num_steps=n_steps,
                movement_type=move,
                geometry_type=geo,
                barrier_dist=dist,
                hole_size=hole,
            )
            sim.history_step = 10  # Оптимизация для графиков
            sim.run()

            self.log_result("Simulation finished.")
            self.log_result("Calculating metrics...")

            # 3. Аналитика
            analyzer = PhysicsAnalyzer()
            slope, r2 = analyzer.calculate_diffusion_coefficient(sim)
            theo_slope = 1.0
            tortuosity = theo_slope / slope

            dr = 4.0
            r_centers, counts, density = analyzer.calculate_radial_concentration(
                sim, dr=dr
            )

            # Вывод цифр
            self.log_result("\n--- METRICS ---")
            self.log_result(f"D_eff Slope: {slope:.4f}")
            self.log_result(f"Linearity R2: {r2:.4f}")
            self.log_result(f"Tortuosity (tau): {tortuosity:.4f}")
            self.log_result("----------------")

            # 4. Отрисовка
            self.fig.clear()

            # Создаем сетку графиков 2x2 (можно настроить иначе)
            ax1 = self.fig.add_subplot(2, 2, 1)  # Траектории
            ax2 = self.fig.add_subplot(2, 2, 2)  # Статистика
            ax3 = self.fig.add_subplot(2, 1, 2)  # Концентрация (на всю ширину снизу)

            # Используем логику из Plotter, но перенаправляем на наши оси (axes)
            # Придется немного адаптировать Plotter, или рисовать вручную.
            # Для надежности вызовем методы отрисовки вручную, используя данные из sim

            # График 1: Траектории
            # Повторяем логику Plotter.plot_trajectories но для конкретного ax
            limit = SimulationPlotter._get_round_limit(
                max(np.max(np.abs(sim.x)), 10), step=20
            )
            if hasattr(sim, "geo_strategy"):
                sim.geo_strategy.draw(ax1, (-limit, limit), (-limit, limit))

            colors = plt.cm.rainbow(np.linspace(0, 1, 50))
            hx, hy = np.array(sim.history_x), np.array(sim.history_y)
            for i in range(min(50, n_part)):
                ax1.plot(hx[:, i], hy[:, i], lw=0.5, alpha=0.6, color=colors[i])
            ax1.set_title(f"Trajectory Map (tau={tortuosity:.2f})")
            ax1.set_xlim(-limit, limit)
            ax1.set_ylim(-limit, limit)
            ax1.set_aspect("equal")

            # График 2: MSD
            X, Y = np.array(sim.history_x), np.array(sim.history_y)
            R2_arr = (X - X[0]) ** 2 + (Y - Y[0]) ** 2
            mean_r2 = np.mean(R2_arr, axis=1)
            steps = np.arange(len(mean_r2)) * sim.history_step
            ax2.plot(steps, mean_r2, "b-", label="Simulation")
            ax2.plot(steps, steps, "k--", alpha=0.5, label="Theory D0")
            ax2.set_title("MSD vs Time")
            ax2.set_xlabel("Steps")
            ax2.set_ylabel("<r^2>")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # График 3: Концентрация
            ax3.plot(r_centers, density, "o-", color="purple", lw=2)
            ax3.fill_between(r_centers, density, alpha=0.3, color="purple")
            ax3.set_title("Radial Concentration Profile")
            ax3.set_xlabel("Radius")
            ax3.set_ylabel("Density")
            ax3.grid(True)

            self.fig.tight_layout()
            self.canvas.draw()

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(e)


if __name__ == "__main__":
    app = ScientificApp()
    app.mainloop()
