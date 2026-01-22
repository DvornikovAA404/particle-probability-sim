import json  # <--- Для работы с конфигами
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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

        self.title("Симулятор Диффузии Частиц v1.3 (Config Support)")
        self.geometry("1400x950")

        # Переменные
        self.current_sim = None
        self.current_analytics_data = {}

        # Стили
        style = ttk.Style()
        style.theme_use("clam")

        # --- МЕНЮ (Верхняя панель) ---
        self.create_menu_bar()

        # Главный контейнер
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 1. ЛЕВАЯ ПАНЕЛЬ (Настройки)
        self.left_panel = ttk.LabelFrame(main_frame, text="Настройки", padding=10)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.create_settings_widgets()

        # 2. ПРАВАЯ ПАНЕЛЬ (Результаты)
        self.right_panel = ttk.LabelFrame(
            main_frame, text="Результаты и Экспорт", padding=10
        )
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        self.create_results_and_export_widgets()

        # 3. ЦЕНТРАЛЬНАЯ ПАНЕЛЬ (Графики)
        self.center_panel = ttk.LabelFrame(main_frame, text="Графики", padding=10)
        self.center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.create_plot_area()

    def create_menu_bar(self):
        """Создает верхнее меню Файл"""
        menubar = tk.Menu(self)

        # Меню "Файл"
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(
            label="📂 Загрузить настройки (JSON)...", command=self.load_config
        )
        file_menu.add_command(
            label="💾 Сохранить настройки (JSON)...", command=self.save_config
        )
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.quit)

        menubar.add_cascade(label="Файл", menu=file_menu)
        self.config(menu=menubar)

    def create_settings_widgets(self):
        def add_param(label_text, default_val, row):
            lbl = ttk.Label(self.left_panel, text=label_text)
            lbl.grid(row=row, column=0, sticky="w", pady=5)
            entry = ttk.Entry(self.left_panel, width=15)
            entry.insert(0, str(default_val))
            entry.grid(row=row, column=1, sticky="e", pady=5)
            return entry

        self.inp_particles = add_param("Частиц (N):", "2000", 0)
        self.inp_steps = add_param("Шагов (t):", "1500", 1)
        self.inp_barrier = add_param("Шаг барьера:", "20.0", 2)
        self.inp_hole = add_param("Размер пор:", "8.0", 3)

        ttk.Label(self.left_panel, text="Геометрия:").grid(
            row=4, column=0, sticky="w", pady=10
        )
        self.combo_geo = ttk.Combobox(
            self.left_panel,
            values=["parallel", "circle", "random", "empty"],
            state="readonly",
        )
        self.combo_geo.current(0)
        self.combo_geo.grid(row=4, column=1, pady=10)

        ttk.Label(self.left_panel, text="Движение:").grid(
            row=5, column=0, sticky="w", pady=5
        )
        self.combo_move = ttk.Combobox(
            self.left_panel, values=["normal", "maxwell"], state="readonly"
        )
        self.combo_move.current(0)
        self.combo_move.grid(row=5, column=1, pady=5)

        self.btn_run = ttk.Button(
            self.left_panel, text="▶ ЗАПУСК", command=self.run_simulation
        )
        self.btn_run.grid(row=6, column=0, columnspan=2, pady=20, sticky="ew")

    def create_results_and_export_widgets(self):
        self.txt_results = tk.Text(
            self.right_panel,
            width=35,
            height=25,
            state="disabled",
            font=("Consolas", 10),
        )
        self.txt_results.pack(fill=tk.BOTH, expand=True)

        btn_copy = ttk.Button(
            self.right_panel, text="Копировать текст", command=self.copy_to_clipboard
        )
        btn_copy.pack(fill=tk.X, pady=5)

        ttk.Separator(self.right_panel, orient="horizontal").pack(fill="x", pady=15)

        lbl_export = ttk.Label(
            self.right_panel,
            text="Сохранить отдельные графики:",
            font=("Arial", 10, "bold"),
        )
        lbl_export.pack(pady=(0, 10))

        self.btn_save_map = ttk.Button(
            self.right_panel,
            text="💾 Карта Частиц (.pdf/.svg)",
            command=self.save_map_plot,
            state="disabled",
        )
        self.btn_save_map.pack(fill=tk.X, pady=2)

        self.btn_save_diff = ttk.Button(
            self.right_panel,
            text="💾 График Диффузии MSD",
            command=self.save_diffusion_plot,
            state="disabled",
        )
        self.btn_save_diff.pack(fill=tk.X, pady=2)

        self.btn_save_conc = ttk.Button(
            self.right_panel,
            text="💾 Профиль Концентрации",
            command=self.save_concentration_plot,
            state="disabled",
        )
        self.btn_save_conc.pack(fill=tk.X, pady=2)

    def create_plot_area(self):
        self.fig = plt.figure(figsize=(9, 9))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.center_panel)
        toolbar = NavigationToolbar2Tk(self.canvas, self.center_panel)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- ЛОГИКА КОНФИГУРАЦИИ (JSON) ---
    def save_config(self):
        """Сохраняет текущие значения полей в JSON"""
        config = {
            "particles": self.inp_particles.get(),
            "steps": self.inp_steps.get(),
            "barrier_dist": self.inp_barrier.get(),
            "hole_size": self.inp_hole.get(),
            "geometry": self.combo_geo.get(),
            "movement": self.combo_move.get(),
        }

        filename = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON Config", "*.json")]
        )
        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=4)
                messagebox.showinfo("Успех", "Настройки сохранены!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")

    def load_config(self):
        """Загружает настройки из JSON и обновляет поля"""
        filename = filedialog.askopenfilename(filetypes=[("JSON Config", "*.json")])
        if filename:
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    config = json.load(f)

                # Обновляем поля (удаляем старое -> вставляем новое)
                if "particles" in config:
                    self.inp_particles.delete(0, tk.END)
                    self.inp_particles.insert(0, config["particles"])

                if "steps" in config:
                    self.inp_steps.delete(0, tk.END)
                    self.inp_steps.insert(0, config["steps"])

                if "barrier_dist" in config:
                    self.inp_barrier.delete(0, tk.END)
                    self.inp_barrier.insert(0, config["barrier_dist"])

                if "hole_size" in config:
                    self.inp_hole.delete(0, tk.END)
                    self.inp_hole.insert(0, config["hole_size"])

                if "geometry" in config:
                    self.combo_geo.set(config["geometry"])

                if "movement" in config:
                    self.combo_move.set(config["movement"])

                messagebox.showinfo("Успех", "Настройки загружены!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Неверный файл конфигурации:\n{e}")

    # --- ЛОГИКА СИМУЛЯЦИИ И ЭКСПОРТА (без изменений) ---
    def log_result(self, text):
        self.txt_results.config(state="normal")
        self.txt_results.insert(tk.END, text + "\n")
        self.txt_results.see(tk.END)
        self.txt_results.config(state="disabled")

    def copy_to_clipboard(self):
        self.clipboard_clear()
        self.clipboard_append(self.txt_results.get("1.0", tk.END))
        messagebox.showinfo("Инфо", "Скопировано!")

    def _enable_export_buttons(self):
        self.btn_save_map.config(state="normal")
        self.btn_save_diff.config(state="normal")
        self.btn_save_conc.config(state="normal")

    def run_simulation(self):
        try:
            n_part = int(self.inp_particles.get())
            n_steps = int(self.inp_steps.get())
            dist = float(self.inp_barrier.get())
            hole = float(self.inp_hole.get())
            geo = self.combo_geo.get()
            move = self.combo_move.get()

            self.txt_results.config(state="normal")
            self.txt_results.delete(1.0, tk.END)
            self.log_result("--- СТАРТ ---")
            self.update()

            sim = SimulationEngine(
                num_trajectories=n_part,
                num_steps=n_steps,
                movement_type=move,
                geometry_type=geo,
                barrier_dist=dist,
                hole_size=hole,
            )
            sim.history_step = 10
            sim.run()

            self.current_sim = sim

            analyzer = PhysicsAnalyzer()
            slope, r2 = analyzer.calculate_diffusion_coefficient(sim)
            tortuosity = 1.0 / slope
            dr = 4.0
            r_centers, counts, density = analyzer.calculate_radial_concentration(
                sim, dr=dr
            )

            self.current_analytics_data = {
                "r_centers": r_centers,
                "density": density,
                "tortuosity": tortuosity,
                "slope": slope,
                "geo": geo,
            }

            self.log_result("\n--- ИТОГИ ---")
            self.log_result(f"Геометрия: {geo}")
            self.log_result(f"Tortuosity (τ): {tortuosity:.4f}")
            self.log_result(f"D_eff slope: {slope:.4f}")

            self._enable_export_buttons()

            self.fig.clear()

            ax1 = self.fig.add_subplot(2, 2, 1)
            limit = SimulationPlotter._get_round_limit(
                max(np.max(np.abs(sim.x)), 10), step=20
            )
            if hasattr(sim, "geo_strategy"):
                sim.geo_strategy.draw(ax1, (-limit, limit), (-limit, limit))
            colors = plt.cm.rainbow(np.linspace(0, 1, 50))
            hx, hy = np.array(sim.history_x), np.array(sim.history_y)
            for i in range(min(50, n_part)):
                ax1.plot(hx[:, i], hy[:, i], lw=0.5, alpha=0.6, color=colors[i])
            ax1.set_title(f"Карта (τ={tortuosity:.2f})")
            ax1.set_xlim(-limit, limit)
            ax1.set_ylim(-limit, limit)
            ax1.set_aspect("equal")

            ax2 = self.fig.add_subplot(2, 2, 2)
            X, Y = np.array(sim.history_x), np.array(sim.history_y)
            mean_r2 = np.mean((X - X[0]) ** 2 + (Y - Y[0]) ** 2, axis=1)
            steps = np.arange(len(mean_r2)) * sim.history_step
            ax2.plot(steps, mean_r2, "b-", label="Sim")
            ax2.plot(steps, steps, "k--", alpha=0.5, label="Theory")
            ax2.set_title("MSD")
            ax2.legend()

            ax3 = self.fig.add_subplot(2, 1, 2)
            ax3.plot(r_centers, density, "o-", color="purple", lw=2)
            ax3.fill_between(r_centers, density, alpha=0.3, color="purple")
            ax3.set_title("Концентрация C(r)")
            ax3.grid(True)

            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _save_plot_helper(self, plot_func, default_name):
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[("PDF", "*.pdf"), ("SVG", "*.svg"), ("PNG", "*.png")],
        )
        if not filename:
            return
        temp_fig = plt.figure(figsize=(8, 6))
        ax = temp_fig.add_subplot(111)
        plot_func(ax)
        try:
            temp_fig.savefig(filename, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Ок", f"Сохранено: {filename}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
        finally:
            plt.close(temp_fig)

    def save_map_plot(self):
        def draw(ax):
            sim = self.current_sim
            limit = SimulationPlotter._get_round_limit(
                max(np.max(np.abs(sim.x)), 10), step=20
            )
            if hasattr(sim, "geo_strategy"):
                sim.geo_strategy.draw(ax, (-limit, limit), (-limit, limit))
            colors = plt.cm.rainbow(np.linspace(0, 1, 100))
            hx, hy = np.array(sim.history_x), np.array(sim.history_y)
            for i in range(min(100, sim.num_trajectories)):
                ax.plot(hx[:, i], hy[:, i], lw=0.8, alpha=0.6, color=colors[i])
            ax.set_title(
                f"Траектории (τ = {self.current_analytics_data['tortuosity']:.3f})"
            )
            ax.set_aspect("equal")

        self._save_plot_helper(draw, "map_plot")

    def save_diffusion_plot(self):
        def draw(ax):
            sim = self.current_sim
            X, Y = np.array(sim.history_x), np.array(sim.history_y)
            mean_r2 = np.mean((X - X[0]) ** 2 + (Y - Y[0]) ** 2, axis=1)
            steps = np.arange(len(mean_r2)) * sim.history_step
            ax.plot(steps, mean_r2, "b-", lw=2)
            ax.plot(steps, steps, "k--", alpha=0.5)
            ax.set_title("MSD")

        self._save_plot_helper(draw, "msd_plot")

    def save_concentration_plot(self):
        def draw(ax):
            r = self.current_analytics_data["r_centers"]
            rho = self.current_analytics_data["density"]
            ax.plot(r, rho, "o-", color="purple")
            ax.fill_between(r, rho, alpha=0.3, color="purple")
            ax.set_title("Концентрация C(r)")

        self._save_plot_helper(draw, "concentration_plot")


if __name__ == "__main__":
    app = ScientificApp()
    app.mainloop()
