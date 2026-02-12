from typing import Tuple
from structure.commands.Command import Command

import tkinter as tk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from matplotlib import patches
from matplotlib import transforms as mtransforms

from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator
from Robot.subsystems.sim_sensors.SimUWB import SimUWB
from Robot.MathUtil import MathUtil
import time


class PlotStateCmd(Command):
    """Command that plots the EKF position (x,y) in a Matplotlib figure embedded
    in a non-blocking Tkinter window.

    Usage: create and schedule this command from your RobotContainer or a script.
    The command will keep the window open until the command is cancelled or the
    user closes the window.
    """

    def __init__(self, max_points: int = 1000):
        super().__init__()
        self.max_points = max_points
        self._running = False

        # plot state
        self.xs = []
        self.ys = []

        # UWB individual tag positions (plotted as dots with gradient)
        # Store up to 2 tags with separate color traces
        self.max_tags = 2
        self.uwb_tag_data = [{'xs': [], 'ys': []} for _ in range(self.max_tags)]

        # GUI objects
        self.root = None
        self.figure = None
        self.ax = None
        self.line = None
        self.canvas = None
        # add a handle for the latest point
        self.last_dot = None

        # top-down widgets
        self.ax_top = None
        self.truck_patch = None
        self.yaw_text = None

        # estimator instance
        self.estimator = KalmanStateEstimator()
        # Use the simulated UWB playback for plotting in simulated mode
        self.uwb = SimUWB()

    def initialize(self):
        # Create Tk window and Matplotlib canvas. We do NOT call mainloop;
        # instead we call root.update() from execute() so the window is non-blocking.
        if tk is None:
            print("PlotStateCmd: tkinter not available; cannot create GUI.")
            self._running = False
            return

        try:
            self.root = tk.Tk()
        except Exception as e:  # pragma: no cover - depends on environment
            print(f"PlotStateCmd: failed to create Tk root: {e}")
            self.root = None
            self._running = False
            return

        self.root.wm_title("EKF Position Plot")

        # Use two subplots: main XY plot on left and a small top-down yaw view
        self.figure = Figure(figsize=(8, 4), dpi=100)
        gs = self.figure.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.3)

        # main XY plot
        self.ax = self.figure.add_subplot(gs[0, 0])
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True)

        # initial empty line
        self.line, = self.ax.plot([], [], "b.-", markersize=4)
        # green marker for the latest point
        self.last_dot, = self.ax.plot([], [], "go", markersize=8, zorder=5)

        # UWB dots: create gradient from red to orange for multiple tags
        self.uwb_plots = []
        for i in range(self.max_tags):
            # Gradient from red (#FF0000) to orange (#FF8800)
            ratio = i / max(1, self.max_tags - 1)
            r = 1.0
            g = 0.0 + (0.533 * ratio)  # 0x88/0xFF = 0.533
            b = 0.0
            color = (r, g, b)
            plot_line, = self.ax.plot([], [], 'o', color=color, markersize=5, linestyle="", alpha=0.7)
            print(f"PlotStateCmd: created UWB tag plot {i} with color {color}")
            self.uwb_plots.append(plot_line) 

        # top-down yaw view on right
        self.ax_top = self.figure.add_subplot(gs[0, 1])
        self.ax_top.set_title("Top-down (yaw)")
        self.ax_top.set_xlim(-1.5, 1.5)
        self.ax_top.set_ylim(-1.5, 1.5)
        self.ax_top.set_aspect("equal")
        self.ax_top.axis("off")

        # draw a simple truck shape as a rectangle + cabin triangle centered at origin
        truck_length = 1.0
        truck_width = 0.6
        # rectangle centered at origin (lower-left at -L/2, -W/2)
        rect = patches.Rectangle(
            (-truck_length / 2.0, -truck_width / 2.0),
            truck_length,
            truck_width,
            facecolor="#d4a017",
            edgecolor="k",
            linewidth=1.0,
        )
        # small cabin triangle at front
        cabin = patches.Polygon(
            [
                (truck_length / 2.0, 0.0),
                (truck_length / 2.0 - 0.15, truck_width / 4.0),
                (truck_length / 2.0 - 0.15, -truck_width / 4.0),
            ],
            closed=True,
            facecolor="#c77b00",
            edgecolor="k",
        )

        # add to axis and keep references for rotation updates
        self.truck_patch = rect
        self.ax_top.add_patch(rect)
        self.ax_top.add_patch(cabin)
        # text below the truck to show numeric yaw
        self.yaw_text = self.ax_top.text(0.0, -1.2, "Yaw: --\N{DEGREE SIGN}", ha="center", va="center")

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.draw()
        widget = self.canvas.get_tk_widget()
        widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self._plot_period = 0.2  # 5 Hz
        self._last_plot_time = 0.0

        # keep running until closed
        self._running = True

    def execute(self):
        # If GUI couldn't be created, nothing to do
        if not self._running or self.root is None:
            return

        # get current position from EKF
        try:
            pos = self.estimator.pos  # numpy array [x,y,z]
        except Exception as e:
            # If estimator fails for any reason, just skip this update
            print(f"PlotStateCmd: failed to read estimator: {e}")
            return

        # safe conversion (in case of None or invalid values)
        try:
            x = float(pos[0])
        except Exception:
            x = np.nan
        try:
            y = float(pos[1])
        except Exception:
            y = np.nan

        self.xs.append(x)
        self.ys.append(y)
        if len(self.xs) > self.max_points:
            self.xs = self.xs[-self.max_points :]
            self.ys = self.ys[-self.max_points :]

        # update line data and autoscale
        self.line.set_data(self.xs, self.ys) # type: ignore
        self.ax.relim() # type: ignore
        self.ax.autoscale_view() # type: ignore

        # update latest point (green)
        if self.xs and self.ys:
            x_last, y_last = self.xs[-1], self.ys[-1]
            if np.isfinite(x_last) and np.isfinite(y_last):
                self.last_dot.set_data([x_last], [y_last])  # type: ignore

        # draw and process Tk events in a non-blocking way
        now = time.time()
        if now - self._last_plot_time >= self._plot_period and self.canvas is not None:
            self.canvas.draw_idle()
            # optionally do a single draw() if you need synchronous update
            self.canvas.draw()
            try:
                self.root.update_idletasks()
                self.root.update()
            except Exception:
                self._running = False
            self._last_plot_time = now

        # update top-down yaw view (do this after drawing to avoid flicker)
        if self.ax_top is not None:
            euler = self.estimator.euler
            yaw = np.rad2deg(euler[2])

            # apply rotation to truck patches around origin
            trans = mtransforms.Affine2D().rotate(np.deg2rad(yaw)) + self.ax_top.transData
            # set same transform for all patches in axis
            for p in list(self.ax_top.patches):
                p.set_transform(trans)

            # update yaw text in degrees
            try:
                if self.yaw_text is not None:
                    self.yaw_text.set_text(f"Yaw: {yaw:.1f}\N{DEGREE SIGN}")
            except Exception:
                if self.yaw_text is not None:
                    self.yaw_text.set_text("Yaw: --\N{DEGREE SIGN}")

        # Update UWB dot positions from the simulated UWB subsystem (individual tags)
        try:
            individual_positions = self.uwb.get_individual_positions()
        except Exception:
            individual_positions = None

        if individual_positions is not None and len(individual_positions) > 0:
            try:
                # Update each tag's position data
                for tag_idx, pos in enumerate(individual_positions[:self.max_tags]):
                    uwb_x = float(pos.x)
                    uwb_y = float(pos.y)
                    self.uwb_tag_data[tag_idx]['xs'].append(uwb_x)
                    self.uwb_tag_data[tag_idx]['ys'].append(uwb_y)
                    
                    # Limit stored points
                    if len(self.uwb_tag_data[tag_idx]['xs']) > self.max_points:
                        self.uwb_tag_data[tag_idx]['xs'] = self.uwb_tag_data[tag_idx]['xs'][-self.max_points:]
                        self.uwb_tag_data[tag_idx]['ys'] = self.uwb_tag_data[tag_idx]['ys'][-self.max_points:]
                    
                    # Update plot data
                    self.uwb_plots[tag_idx].set_data(
                        self.uwb_tag_data[tag_idx]['xs'],
                        self.uwb_tag_data[tag_idx]['ys']
                    )
            except Exception as e:
                # ignore malformed simulated readings
                pass

    def end(self, interrupted):
        # close window and cleanup
        self._running = False
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None

    def is_finished(self) -> bool:
        # the command finishes when the window is closed or end() is called
        return not self._running
