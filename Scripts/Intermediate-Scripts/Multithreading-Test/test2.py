import subprocess
import threading
import time
import sys
import os
import signal
import platform

import numpy as np
import matplotlib.pyplot as plt
import csv
# from model_inference import run_model  # Optional
# from sensor_logger import log_sensor_data  # Optional


DATA_DIR = "Data/"
TX_SCRIPT = "tx_flowgraph.py"
RX_SCRIPT = "rx_flowgraph.py"

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError:
        print("Failed to install some packages. Continuing anyway...")


def run_flowgraph(script_path):
    if platform.system() == "Windows":
        # Windows: no setsid, use python (or python3 if your system uses it)
        return subprocess.Popen(["python", script_path])
    else:
        # Linux/Unix: use setsid to start a new process group for easier termination
        return subprocess.Popen(["python3", script_path], preexec_fn=os.setsid)

def terminate_process(proc):
    if platform.system() == "Windows":
        proc.terminate()
    else:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

def wait_for_data(file_path, timeout=30):
    for _ in range(timeout):
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return True
        time.sleep(1)
    return False


def save_to_csv(rx_file_path, tx_file_path, csv_file_path):
    # Load data file
    rx_data = np.fromfile(open(rx_file_path), dtype=np.complex64)
    tx_data = np.fromfile(open(tx_file_path), dtype=np.complex64)

    # Save last 500000 samples if available (adjust size as needed)
    tx_data_last = tx_data[-500000:] if len(tx_data) >= 500000 else tx_data
    rx_data_last = rx_data[-500000:] if len(rx_data) >= 500000 else rx_data

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", 
                        "TX Real", "TX Imag", "TX Magnitude", 
                        "RX Real", "RX Imag", "RX Magnitude"])
        
        for i in range(len(tx_data_last)):
            tx = tx_data_last[i]
            rx = rx_data_last[i] if i < len(rx_data_last) else 0
            writer.writerow([
                i, 
                np.real(tx), np.imag(tx), np.abs(tx), 
                np.real(rx), np.imag(rx), np.abs(rx)
            ])

def main():
    install_requirements()
    
    print("Launching TX and RX scripts...")
    tx_proc = run_flowgraph(TX_SCRIPT)
    rx_proc = run_flowgraph(RX_SCRIPT)

    print("Waiting for data to be written...")
    rx_ok = wait_for_data(os.path.join(DATA_DIR, "rxdata.dat"))
    tx_ok = wait_for_data(os.path.join(DATA_DIR, "txdata.dat"))

    if not (rx_ok and tx_ok):
        print("Timed out waiting for signal files.")
        terminate_process(tx_proc)
        terminate_process(rx_proc)
        return

    print("Data detected. Running converters...")

    save_to_csv(os.path.join(DATA_DIR, "rxdata.dat"), 
                os.path.join(DATA_DIR, "txdata.dat"), 
                os.path.join(DATA_DIR, "signal.csv"))

    print("CSVs saved.")

    # Optional: ML
    # print("Running ML model...")
    # run_model("Data/rx_signal.csv")

    # Optional: Background sensor logging (if long-running)
    # threading.Thread(target=log_sensor_data, daemon=True).start()

    print("All tasks completed.")

    terminate_process(tx_proc)
    terminate_process(rx_proc)

if __name__ == "__main__":
    main()