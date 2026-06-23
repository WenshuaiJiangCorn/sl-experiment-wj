import numpy as np

import threading
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from binding_classes import LinearTrackFunctions
from ataraxis_base_utilities import ensure_directory_exists
from main_experiment_for_GUI import ExperimentControl, run_experiment


DEFAULT_EXPERIMENT_DIR = "C:\\Users\\yapici\\Desktop\\lineartrack_data\\10_percent_sucrose\\2026June_DAT_FoodRestricted\\raw_data"
DEFAULT_REWARD_VOLUME = "10"
DEFAULT_CALIBRATION_PULSE = "60000"


class ExperimentGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Experiment Launcher")
        self.resizable(True, True)
        self._control: ExperimentControl | None = None
        self._exp_thread: threading.Thread | None = None
        self._maint_thread: threading.Thread | None = None
        self._build_config_panel()

    # ------------------------------------------------------------------
    # Config panel
    # ------------------------------------------------------------------

    def _build_config_panel(self):
        self._config_frame = tk.Frame(self)
        self._config_frame.pack(fill="both", expand=True)
        f = self._config_frame

        tk.Label(f, text="Experiment Directory:").grid(row=0, column=0, sticky="w", padx=10, pady=6)
        self.exp_dir_var = tk.StringVar(value=DEFAULT_EXPERIMENT_DIR)
        tk.Entry(f, textvariable=self.exp_dir_var, width=52).grid(row=0, column=1, padx=10, pady=6)
        tk.Button(f, text="Browse…", command=self._browse_dir).grid(row=0, column=2, padx=(0, 10))

        tk.Label(f, text="Reward Volume (µL):").grid(row=1, column=0, sticky="w", padx=10, pady=6)
        self.reward_var = tk.StringVar(value=DEFAULT_REWARD_VOLUME)
        tk.Entry(f, textvariable=self.reward_var, width=10).grid(row=1, column=1, sticky="w", padx=10, pady=6)

        tk.Label(f, text="Mouse ID:").grid(row=2, column=0, sticky="w", padx=10, pady=6)
        self.mouse_var = tk.StringVar()
        tk.Entry(f, textvariable=self.mouse_var, width=20).grid(row=2, column=1, sticky="w", padx=10, pady=6)
        tk.Label(f, text="(e.g. DATM1)", fg="grey").grid(row=2, column=2, sticky="w", padx=(0, 10))

        tk.Label(f, text="Experiment Day:").grid(row=3, column=0, sticky="w", padx=10, pady=6)
        self.day_var = tk.StringVar()
        tk.Entry(f, textvariable=self.day_var, width=20).grid(row=3, column=1, sticky="w", padx=10, pady=6)
        tk.Label(f, text="(e.g. day_1)", fg="grey").grid(row=3, column=2, sticky="w", padx=(0, 10))

        tk.Label(f, text="Output Directory:").grid(row=4, column=0, sticky="w", padx=10, pady=6)
        self.output_preview = tk.Label(f, text="", fg="grey", anchor="w", wraplength=480, justify="left")
        self.output_preview.grid(row=4, column=1, columnspan=2, sticky="w", padx=10, pady=6)

        for var in (self.exp_dir_var, self.mouse_var, self.day_var):
            var.trace_add("write", lambda *_: self._update_preview())
        self._update_preview()

        ttk.Separator(f, orient="horizontal").grid(row=5, column=0, columnspan=3, sticky="ew", pady=4)

        btn_row = tk.Frame(f)
        btn_row.grid(row=6, column=0, columnspan=3, pady=(4, 12))
        tk.Button(btn_row, text="Start Experiment", command=self._start,
                  bg="#4CAF50", fg="white", font=("", 11, "bold"),
                  padx=12, pady=4).pack(side="left", padx=12)
        tk.Button(btn_row, text="Maintenance Mode", command=self._open_maintenance,
                  bg="#FF9800", fg="white", font=("", 11, "bold"),
                  padx=12, pady=4).pack(side="left", padx=12)

    # ------------------------------------------------------------------
    # Control panel (shown while experiment is running)
    # ------------------------------------------------------------------

    def _build_control_panel(self, output_dir: Path):
        self._control_frame = tk.Frame(self)
        self._control_frame.pack(fill="both", expand=True, padx=20, pady=16)
        f = self._control_frame

        tk.Label(f, text="Experiment Running", font=("", 13, "bold"), fg="#4CAF50").pack(pady=(0, 4))
        tk.Label(f, text=str(output_dir), fg="grey", wraplength=460, justify="left").pack(pady=(0, 12))

        btn_frame = tk.Frame(f)
        btn_frame.pack(pady=8)

        self._btn_left = tk.Button(
            btn_frame, text="Dispense Left", width=16, height=4,
            bg="#2196F3", fg="white", font=("", 11, "bold"),
            command=self._dispense_left,
        )
        self._btn_left.grid(row=0, column=0, padx=12)

        self._btn_right = tk.Button(
            btn_frame, text="Dispense Right", width=16, height=4,
            bg="#2196F3", fg="white", font=("", 11, "bold"),
            command=self._dispense_right,
        )
        self._btn_right.grid(row=0, column=1, padx=12)

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=12)

        self._btn_proceed = tk.Button(
            f, text="Proceed (skip acclimation)", font=("", 10),
            command=self._proceed,
        )
        self._btn_proceed.pack(pady=(0, 6))

        self._btn_stop = tk.Button(
            f, text="Stop Experiment", width=24, height=2,
            bg="#f44336", fg="white", font=("", 11, "bold"),
            command=self._stop_experiment,
        )
        self._btn_stop.pack(pady=(0, 12))

    # ------------------------------------------------------------------
    # Maintenance panel
    # ------------------------------------------------------------------

    def _build_maintenance_panel(self):
        self._maint_frame = tk.Frame(self)
        self._maint_frame.pack(fill="both", expand=True, padx=16, pady=12)
        f = self._maint_frame

        tk.Label(f, text="Maintenance Mode", font=("", 13, "bold"), fg="#FF9800").grid(
            row=0, column=0, columnspan=4, pady=(0, 8))

        # Status bar
        self._maint_status_var = tk.StringVar(value="Idle")
        tk.Label(f, textvariable=self._maint_status_var, fg="grey", font=("", 9, "italic")).grid(
            row=1, column=0, columnspan=4, pady=(0, 6))

        ttk.Separator(f, orient="horizontal").grid(row=2, column=0, columnspan=4, sticky="ew", pady=4)

        # --- Open valve ------------------------------------------------
        tk.Label(f, text="Open Valve", font=("", 10, "bold")).grid(row=3, column=0, sticky="w", padx=8, pady=4)
        tk.Label(f, text="Side:").grid(row=3, column=1, sticky="e", padx=4)
        self._ov_side = tk.StringVar(value="left")
        ttk.Combobox(f, textvariable=self._ov_side, values=["left", "right"],
                     state="readonly", width=7).grid(row=3, column=2, sticky="w")
        tk.Label(f, text="Duration (s):").grid(row=3, column=3, sticky="e", padx=4)
        self._ov_dur = tk.StringVar(value="1")
        tk.Entry(f, textvariable=self._ov_dur, width=6).grid(row=3, column=4, sticky="w", padx=4)
        self._maint_btns: list[tk.Button] = []
        btn = tk.Button(f, text="Run", width=6,
                        command=lambda: self._run_maint(self._maint_open_valve))
        btn.grid(row=3, column=5, padx=8)
        self._maint_btns.append(btn)

        ttk.Separator(f, orient="horizontal").grid(row=4, column=0, columnspan=6, sticky="ew", pady=4)

        # --- Calibrate valve -------------------------------------------
        tk.Label(f, text="Calibrate Valve", font=("", 10, "bold")).grid(row=5, column=0, sticky="w", padx=8, pady=4)
        tk.Label(f, text="Side:").grid(row=5, column=1, sticky="e", padx=4)
        self._cal_side = tk.StringVar(value="left")
        ttk.Combobox(f, textvariable=self._cal_side, values=["left", "right"],
                     state="readonly", width=7).grid(row=5, column=2, sticky="w")
        tk.Label(f, text="Pulse (µs):").grid(row=5, column=3, sticky="e", padx=4)
        self._cal_pulse = tk.StringVar(value=DEFAULT_CALIBRATION_PULSE)
        tk.Entry(f, textvariable=self._cal_pulse, width=8).grid(row=5, column=4, sticky="w", padx=4)
        btn = tk.Button(f, text="Run", width=6,
                        command=lambda: self._run_maint(self._maint_calibrate_valve))
        btn.grid(row=5, column=5, padx=8)
        self._maint_btns.append(btn)

        ttk.Separator(f, orient="horizontal").grid(row=6, column=0, columnspan=6, sticky="ew", pady=4)

        # --- Delivery test ---------------------------------------------
        tk.Label(f, text="Delivery Test (40×)", font=("", 10, "bold")).grid(row=7, column=0, sticky="w", padx=8, pady=4)
        tk.Label(f, text="Side:").grid(row=7, column=1, sticky="e", padx=4)
        self._dt_side = tk.StringVar(value="left")
        ttk.Combobox(f, textvariable=self._dt_side, values=["left", "right"],
                     state="readonly", width=7).grid(row=7, column=2, sticky="w")
        btn = tk.Button(f, text="Run", width=6,
                        command=lambda: self._run_maint(self._maint_delivery_test))
        btn.grid(row=7, column=5, padx=8)
        self._maint_btns.append(btn)

        ttk.Separator(f, orient="horizontal").grid(row=8, column=0, columnspan=6, sticky="ew", pady=4)

        # --- Training --------------------------------------------------
        tk.Label(f, text="Training", font=("", 10, "bold")).grid(row=9, column=0, sticky="w", padx=8, pady=4)
        btn1 = tk.Button(f, text="Day 1", width=10,
                         command=lambda: self._run_maint(self._maint_day1_training))
        btn1.grid(row=9, column=1, columnspan=2, padx=8)
        self._maint_btns.append(btn1)
        btn2 = tk.Button(f, text="Day 2", width=10,
                         command=lambda: self._run_maint(self._maint_day2_training))
        btn2.grid(row=9, column=3, columnspan=2, padx=8)
        self._maint_btns.append(btn2)

        ttk.Separator(f, orient="horizontal").grid(row=10, column=0, columnspan=6, sticky="ew", pady=4)

        # --- Test noise ------------------------------------------------
        tk.Label(f, text="Test Noise", font=("", 10, "bold")).grid(row=11, column=0, sticky="w", padx=8, pady=4)
        btn = tk.Button(f, text="Run", width=6,
                        command=lambda: self._run_maint(self._maint_test_noise))
        btn.grid(row=11, column=5, padx=8)
        self._maint_btns.append(btn)

        ttk.Separator(f, orient="horizontal").grid(row=12, column=0, columnspan=6, sticky="ew", pady=8)

        self._maint_back_btn = tk.Button(f, text="← Back", command=self._close_maintenance)
        self._maint_back_btn.grid(row=13, column=0, columnspan=6, pady=(0, 8))

    # ------------------------------------------------------------------
    # Maintenance helpers
    # ------------------------------------------------------------------

    def _open_maintenance(self):
        self._config_frame.pack_forget()
        self._build_maintenance_panel()

    def _close_maintenance(self):
        self._maint_frame.pack_forget()
        self._config_frame.pack(fill="both", expand=True)

    def _run_maint(self, func):
        """Run a maintenance function in a background thread; disable all buttons while running."""
        if self._maint_thread and self._maint_thread.is_alive():
            messagebox.showwarning("Busy", "Another maintenance operation is already running.")
            return

        for btn in self._maint_btns:
            btn.config(state="disabled")
        self._maint_back_btn.config(state="disabled")
        self._maint_status_var.set("Running…")

        def _wrapper():
            try:
                func()
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Error", str(exc)))
            finally:
                self.after(0, self._maint_done)

        self._maint_thread = threading.Thread(target=_wrapper, daemon=True)
        self._maint_thread.start()

    def _maint_done(self):
        self._maint_status_var.set("Idle")
        for btn in self._maint_btns:
            btn.config(state="normal")
        self._maint_back_btn.config(state="normal")

    # --- Individual maintenance actions --------------------------------

    def _maint_open_valve(self):
        side = self._ov_side.get()
        try:
            duration = int(self._ov_dur.get())
            if duration <= 0:
                raise ValueError
        except ValueError:
            raise ValueError("Duration must be a positive integer (seconds).")
        exp = LinearTrackFunctions()
        exp.open_valve(valve_side=side, duration=duration)

    def _maint_calibrate_valve(self):
        side = self._cal_side.get()
        try:
            pulse = np.uint32(int(self._cal_pulse.get()))
        except (ValueError, OverflowError):
            raise ValueError("Pulse duration must be a positive integer (µs).")
        exp = LinearTrackFunctions()
        exp.calibrate_valve(valve_side=side, calibration_pulse_duration=pulse)

    def _maint_delivery_test(self):
        side = self._dt_side.get()
        exp = LinearTrackFunctions()
        exp.delivery_test(valve_side=side)

    def _maint_day1_training(self):
        exp = LinearTrackFunctions()
        exp.first_day_training()

    def _maint_day2_training(self):
        exp = LinearTrackFunctions()
        exp.second_day_training()

    def _maint_test_noise(self):
        exp = LinearTrackFunctions()
        exp.test_noise()

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _browse_dir(self):
        chosen = filedialog.askdirectory(initialdir=self.exp_dir_var.get() or "/")
        if chosen:
            self.exp_dir_var.set(chosen)

    def _update_preview(self):
        try:
            base = Path(self.exp_dir_var.get())
            mouse = self.mouse_var.get().strip()
            day = self.day_var.get().strip()
            if mouse and day:
                date = datetime.now().strftime("%Y%m%d")
                out = base / mouse / f"{day}_{date}"
                self.output_preview.config(text=str(out), fg="black")
            else:
                self.output_preview.config(text="(fill in Mouse ID and Experiment Day)", fg="grey")
        except Exception:
            self.output_preview.config(text="(invalid path)", fg="red")

    # ------------------------------------------------------------------
    # Experiment lifecycle
    # ------------------------------------------------------------------

    def _start(self):
        try:
            reward_volume = float(self.reward_var.get())
            if reward_volume <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Reward volume must be a positive number.")
            return

        mouse = self.mouse_var.get().strip()
        if not mouse:
            messagebox.showerror("Invalid Input", "Mouse ID cannot be empty.")
            return

        day = self.day_var.get().strip()
        if not day:
            messagebox.showerror("Invalid Input", "Experiment Day cannot be empty.")
            return

        date = datetime.now().strftime("%Y%m%d")
        output_dir = Path(self.exp_dir_var.get()) / mouse / f"{day}_{date}"

        if output_dir.exists():
            if not messagebox.askyesno(
                "Directory Exists",
                f"Output directory already exists:\n{output_dir}\n\nData may be overwritten. Continue?",
            ):
                return

        ensure_directory_exists(output_dir)

        self._control = ExperimentControl()
        self._exp_thread = threading.Thread(
            target=run_experiment,
            args=(output_dir, np.float64(reward_volume), self._control),
            daemon=True,
        )
        self._exp_thread.start()

        self._config_frame.pack_forget()
        self._build_control_panel(output_dir)
        self._poll_experiment()

    def _poll_experiment(self):
        if self._exp_thread and not self._exp_thread.is_alive():
            self._on_experiment_done()
        else:
            self.after(500, self._poll_experiment)

    def _on_experiment_done(self):
        for btn in (self._btn_left, self._btn_right, self._btn_stop, self._btn_proceed):
            btn.config(state="disabled")
        messagebox.showinfo("Experiment Ended", "Experiment has finished and data has been saved.")
        self._control_frame.pack_forget()
        self._config_frame.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Experiment control callbacks
    # ------------------------------------------------------------------

    def _dispense_left(self):
        if self._control:
            self._control.dispense_left.set()

    def _dispense_right(self):
        if self._control:
            self._control.dispense_right.set()

    def _proceed(self):
        if self._control:
            self._control.proceed.set()
            self._btn_proceed.config(state="disabled")

    def _stop_experiment(self):
        if self._control:
            if messagebox.askyesno("Stop Experiment", "Are you sure you want to stop the experiment?"):
                self._control.stop.set()
                self._btn_stop.config(state="disabled")


if __name__ == "__main__":
    app = ExperimentGUI()
    app.mainloop()
