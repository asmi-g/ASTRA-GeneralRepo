import subprocess
import threading
import time
import os
import signal
from csv import convert_complex_to_csv
# from model_inference import run_model  # Optional
# from sensor_logger import log_sensor_data  # Optional

DATA_DIR = "Data/"
TX_SCRIPT = "tx_flowgraph.py"
RX_SCRIPT = "rx_flowgraph.py"

def run_flowgraph(script_path):
    return subprocess.Popen(["python3", script_path], preexec_fn=os.setsid)

def wait_for_data(file_path, timeout=30):
    for _ in range(timeout):
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return True
        time.sleep(1)
    return False

def main():
    print("Launching TX and RX scripts...")
    tx_proc = run_flowgraph(TX_SCRIPT)
    rx_proc = run_flowgraph(RX_SCRIPT)

    print("Waiting for data to be written...")
    rx_ok = wait_for_data(os.path.join(DATA_DIR, "rxdata.dat"))
    tx_ok = wait_for_data(os.path.join(DATA_DIR, "txdata.dat"))

    if not (rx_ok and tx_ok):
        print("Timed out waiting for signal files.")
        os.killpg(os.getpgid(tx_proc.pid), signal.SIGTERM)
        os.killpg(os.getpgid(rx_proc.pid), signal.SIGTERM)
        return

    print("Data detected. Running converters...")

    convert_complex_to_csv(f"{DATA_DIR}/rxdata.dat", f"{DATA_DIR}/rx_signal.csv", start=-10000)
    convert_complex_to_csv(f"{DATA_DIR}/txdata.dat", f"{DATA_DIR}/tx_signal.csv", start=0, end=10000)

    print("CSVs saved.")

    # Optional: ML
    # print("Running ML model...")
    # run_model("Data/rx_signal.csv")

    # Optional: Background sensor logging (if long-running)
    # threading.Thread(target=log_sensor_data, daemon=True).start()

    print("All tasks completed.")

    # Optional: keep running for streaming/sensor logging
    # input("Press Enter to terminate...")
    os.killpg(os.getpgid(tx_proc.pid), signal.SIGTERM)
    os.killpg(os.getpgid(rx_proc.pid), signal.SIGTERM)

if __name__ == "__main__":
    main()
