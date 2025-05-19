import subprocess
import threading
import time
import os
import signal

import numpy as np
import matplotlib.pyplot as plt
import csv
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



#rx_file_path = 'C:/Users/asmig/OneDrive/Documents/GNURadio/Data/rxdata' # Insert here
#tx_file_path = 'C:/Users/asmig/OneDrive/Documents/GNURadio/Data/txdata' # Insert here

# Save to CSV
#csv_file_path = 'C:/Users/asmig/OneDrive/Documents/GNURadio/Data/rx_tx_data.csv'



def save_to_csv(rx_file_path, tx_file_path, csv_file_path):
    # Load data file
    rx_data = np.fromfile(open(rx_file_path), dtype=np.complex64)
    tx_data = np.fromfile(open(tx_file_path), dtype=np.complex64)

    # Save last 10,000 samples of both
    tx_data_last = tx_data[-500000:] if len(tx_data) >= 500000 else tx_data
    rx_data_last = rx_data[-500000:] if len(rx_data) >= 500000 else rx_data

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", 
                        "TX Real", "TX Imag", "TX Magnitude", 
                        "RX Real", "RX Imag", "RX Magnitude"])
        
        for i in range(len(tx_data_last)):  # should be 10,000 if data is long enough
            tx = tx_data_last[i]
            rx = rx_data_last[i] if i < len(rx_data_last) else 0
            writer.writerow([
                i, 
                np.real(tx), np.imag(tx), np.abs(tx), 
                np.real(rx), np.imag(rx), np.abs(rx)
            ])

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

    save_to_csv(f"{DATA_DIR}/rxdata.dat", f"{DATA_DIR}/txdata.dat", f"{DATA_DIR}/signal.csv")
    # convert_complex_to_csv(f"{DATA_DIR}/txdata.dat", f"{DATA_DIR}/tx_signal.csv", start=0, end=10000)

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
