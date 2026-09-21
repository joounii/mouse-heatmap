import csv
import time
from threading import Lock
from pynput import mouse

OUTPUT_FILE = "mouse_data.csv"
SAMPLE_INTERVAL = 0.02

# Thread-safe container for current state
state_lock = Lock()
current_pos = (0, 0)
clicks_buffer = []

def on_move(x, y):
    global current_pos
    with state_lock:
        current_pos = (x, y)

def on_click(x, y, button, pressed):
    if pressed:
        btn_name = "left" if button == mouse.Button.left else "right" if button == mouse.Button.right else "other"
        with state_lock:
            clicks_buffer.append((time.time(), x, y, btn_name))

def main():
    print(f"Tracking mouse activity to '{OUTPUT_FILE}'. Press Ctrl+C to stop...")

    # Initialize CSV files
    with open("mouse_positions.csv", "w", newline="") as f_pos, \
         open("mouse_clicks.csv", "w", newline="") as f_clk:
        
        pos_writer = csv.writer(f_pos)
        clk_writer = csv.writer(f_clk)
        
        pos_writer.writerow(["timestamp", "x", "y"])
        clk_writer.writerow(["timestamp", "x", "y", "button"])

        listener = mouse.Listener(on_move=on_move, on_click=on_click)
        listener.start()

        try:
            while True:
                time.sleep(SAMPLE_INTERVAL)
                now = time.time()
                
                with state_lock:
                    pos = current_pos
                    clicks_to_write = list(clicks_buffer)
                    clicks_buffer.clear()

                pos_writer.writerow([now, pos[0], pos[1]])
                for clk in clicks_to_write:
                    clk_writer.writerow(clk)

        except KeyboardInterrupt:
            print("\nStopped tracking. Data saved.")
        finally:
            listener.stop()

if __name__ == "__main__":
    main()
