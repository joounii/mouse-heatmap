import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_and_process_data():
    pos_df = pd.read_csv("mouse_positions.csv")
    clk_df = pd.read_csv("mouse_clicks.csv")

    # Drop any stagnant duplicate timestamps
    pos_df = pos_df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # Calculate velocity in pixels per second: sqrt(dx^2 + dy^2) / dt
    dx = pos_df["x"].diff()
    dy = pos_df["y"].diff()
    dt = pos_df["timestamp"].diff()

    distance = np.sqrt(dx**2 + dy**2)
    velocity = distance / dt

    # Filter out stationary noise and potential sleep-time jumps
    pos_df["velocity_px_s"] = np.clip(velocity, 0, 10000)
    pos_df["time_rel_s"] = pos_df["timestamp"] - pos_df["timestamp"].iloc[0]

    return pos_df, clk_df

def get_screen_bounds(pos_df, clk_df, margin=0.05):
    # Combine positions and clicks so neither gets cropped out
    all_x = pd.concat([pos_df["x"], clk_df["x"]]) if not clk_df.empty else pos_df["x"]
    all_y = pd.concat([pos_df["y"], clk_df["y"]]) if not clk_df.empty else pos_df["y"]

    min_x, max_x = all_x.min(), all_x.max()
    min_y, max_y = all_y.min(), all_y.max()

    pad_x = (max_x - min_x) * margin if max_x > min_x else 100
    pad_y = (max_y - min_y) * margin if max_y > min_y else 100

    # Keep bounds in natural ascending order (min < max)
    xlim = (min_x - pad_x, max_x + pad_x)
    ylim = (min_y - pad_y, max_y + pad_y)

    return xlim, ylim

def plot_dashboard():
    pos_df, clk_df = load_and_process_data()
    xlim, ylim = get_screen_bounds(pos_df, clk_df)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1])

    ax_heat = fig.add_subplot(gs[:, 0])
    ax_clicks = fig.add_subplot(gs[0, 1])
    ax_vel = fig.add_subplot(gs[1, 1])

    # 1. Density Heatmap
    # extent expects (xmin, xmax, ymin, ymax) with ymin < ymax
    hb = ax_heat.hexbin(
        pos_df["x"], 
        pos_df["y"], 
        gridsize=75, 
        cmap="inferno", 
        mincnt=1, 
        bins="log",
        extent=(xlim[0], xlim[1], ylim[0], ylim[1])
    )
    ax_heat.set_xlim(xlim)
    ax_heat.set_ylim(ylim)
    ax_heat.invert_yaxis()
    ax_heat.set_aspect("equal")
    ax_heat.set_title("Mouse Cursor Position Density (All Displays)", fontsize=12, pad=10)
    ax_heat.set_xlabel("X (Pixels)")
    ax_heat.set_ylabel("Y (Pixels)")
    fig.colorbar(hb, ax=ax_heat, orientation="horizontal", label="Sample Count (log)", pad=0.08, shrink=0.7)

    # 2. Clicks Distribution
    if not clk_df.empty:
        left_clicks = clk_df[clk_df["button"] == "left"]
        right_clicks = clk_df[clk_df["button"] == "right"]

        ax_clicks.scatter(left_clicks["x"], left_clicks["y"], color="#38bdf8", alpha=0.6, s=25, label="Left Click")
        ax_clicks.scatter(right_clicks["x"], right_clicks["y"], color="#f43f5e", alpha=0.7, s=35, marker="^", label="Right Click")
        
        ax_clicks.set_xlim(xlim)
        ax_clicks.set_ylim(ylim)
        ax_clicks.invert_yaxis()
        ax_clicks.set_aspect("equal")
        ax_clicks.set_title(f"Click Map (Total: {len(clk_df)})", fontsize=11)
        ax_clicks.legend(loc="upper right")
        ax_clicks.grid(True, linestyle="--", alpha=0.2)

    # 3. Velocity Over Time
    pos_df["vel_smooth"] = pos_df["velocity_px_s"].rolling(window=25, min_periods=1).mean()
    
    ax_vel.plot(pos_df["time_rel_s"], pos_df["vel_smooth"], color="#a78bfa", linewidth=1.2, label="Velocity (px/s)")
    ax_vel.fill_between(pos_df["time_rel_s"], pos_df["vel_smooth"], color="#a78bfa", alpha=0.15)
    
    total_px = np.sqrt(pos_df["x"].diff()**2 + pos_df["y"].diff()**2).sum()
    approx_meters = total_px * 0.000264

    ax_vel.set_title(f"Cursor Speed Profile | Total Odometer: {approx_meters:.1f} m", fontsize=11)
    ax_vel.set_xlabel("Session Elapsed Time (Seconds)")
    ax_vel.set_ylabel("Speed (px / s)")
    ax_vel.set_ylim(bottom=0)
    ax_vel.grid(True, linestyle="--", alpha=0.2)

    plt.show()

if __name__ == "__main__":
    plot_dashboard()