import subprocess
import threading
import time
import sys
import os
import signal
import platform
import numpy as np
import csv

DATA_DIR = "Data/"
TX_SCRIPT = "TX.py"
RX_SCRIPT = "RX.py"
CSV_FILE_PATH = os.path.join(DATA_DIR, "signal.csv")
RUNTIME_SECONDS = 10  # duration to run TX/RX per cycle

# TO DO
# - Fix: "sink :warning: Soapy sink error: TIMEOUT"


def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError:
        print("Failed to install some packages. Continuing anyway...")


def run_flowgraph(script_path):
    if platform.system() == "Windows":
        return subprocess.Popen(["python", script_path])
    else:
        return subprocess.Popen(["python3", script_path], preexec_fn=os.setsid)


def terminate_process(proc):
    if platform.system() == "Windows":
        proc.terminate()
    else:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)


def save_to_csv(rx_file_path, tx_file_path, csv_file_path):
    rx_data = np.fromfile(open(rx_file_path), dtype=np.complex64)
    tx_data = np.fromfile(open(tx_file_path), dtype=np.complex64)

    tx_data_last = tx_data[-500000:] if len(tx_data) >= 500000 else tx_data
    rx_data_last = rx_data[-500000:] if len(rx_data) >= 500000 else rx_data

    write_header = not os.path.exists(csv_file_path)

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Index", "TX Real", "TX Imag", "TX Magnitude", "RX Real", "RX Imag", "RX Magnitude"])

        for i in range(len(tx_data_last)):
            tx = tx_data_last[i]
            rx = rx_data_last[i] if i < len(rx_data_last) else 0
            writer.writerow([
                i,
                np.real(tx), np.imag(tx), np.abs(tx),
                np.real(rx), np.imag(rx), np.abs(rx)
            ])


def cycle_once():
    print("Launching TX and RX scripts...")
    tx_proc = run_flowgraph(TX_SCRIPT)
    rx_proc = run_flowgraph(RX_SCRIPT)
    

    print(f"Running for {RUNTIME_SECONDS} seconds...")
    time.sleep(RUNTIME_SECONDS)

    print("Terminating scripts...")
    terminate_process(tx_proc)
    terminate_process(rx_proc)

    print("Saving to CSV...")
    save_to_csv(os.path.join(DATA_DIR, "rxdata.dat"),
                os.path.join(DATA_DIR, "txdata.dat"),
                CSV_FILE_PATH)
    print("Cycle complete.\n")


def main():
    install_requirements()

    while True:
        cycle_once()
        time.sleep(2)  # Optional delay between cycles


if __name__ == "__main__":
    main()