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
ML_SCRIPT = "../../RL-EnvConfig/inference.py"
TEMP_LOGGER_SCRIPT = "Scripts/SystemTesting/temperature_logger.py"
CSV_FILE_PATH = os.path.join(DATA_DIR, "signal.csv")
RUNTIME_SECONDS = 10  # duration to run TX/RX per cycle

# TO DO
# - Fix: "sink :warning: Soapy sink error: TIMEOUT"
# - Integrate Chelsea's comments from previous pr
# - Integrate AM scripts, address throttle block error and rerun on WSL
# - Integrate timed operation for flight



def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError:
        print("Failed to install some packages. Continuing anyway...")


def run_script(script_path):
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


def SDR_cycle():
    print("Launching TX and RX scripts...")
    tx_proc = run_script(TX_SCRIPT)
    rx_proc = run_script(RX_SCRIPT)
    

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

    # Launch temperature logging script
    print("Launching temperature logger...")
    temp_logger_proc = run_script(TEMP_LOGGER_SCRIPT)

    # Launch ML inference script once
    print("Launching ML model...")
    ml_proc = run_script(ML_SCRIPT)

    # SDR capture loop
    try:
        while True:
            SDR_cycle()
            time.sleep(2)  # Optional delay between cycles
    except KeyboardInterrupt:
        print("Terminating Model Operation...")
        terminate_process(ml_proc)

        print("Terminating temperature logging...")
        terminate_process(temp_logger_proc)


if __name__ == "__main__":
    main()