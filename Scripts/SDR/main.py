import subprocess
import multiprocessing as mp
import time
import sys
import os
import signal
import platform
try:
    import cupy as np # CuPy is a GPU-accelerated library similar to NumPy
    print("Using CuPy (GPU)")
except ImportError:
    import numpy as np
    print("Using NumPy (CPU fallback)") 
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Base directory: {BASE_DIR}")
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "Data/"))
print(f"Data directory: {DATA_DIR}")
TX_SCRIPT = os.path.join(BASE_DIR, "TX.py")
print(f"TX script path: {TX_SCRIPT}")
RX_SCRIPT = os.path.join(BASE_DIR, "RX.py")
print(f"RX script path: {RX_SCRIPT}")
ML_SCRIPT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "RL-EnvConfig", "inference.py"))
print(f"ML script path: {ML_SCRIPT}")
TEMP_LOGGER_SCRIPT = os.path.abspath(os.path.join(BASE_DIR, "..", "SystemTesting", "temperature_logger.py"))
print(f"Temperature logger script path: {TEMP_LOGGER_SCRIPT}")

CSV_DIR = os.path.join(DATA_DIR, "csv")
if not os.path.exists(CSV_DIR):
    os.makedirs(CSV_DIR)
    print(f"Created CSV directory: {CSV_DIR}")
CSV_FILE_PATH = os.path.join(CSV_DIR, "signal.csv")
if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, 'w') as f:
        pass  # Create an empty CSV file if it doesn't exist
    print(f"Created CSV file: {CSV_FILE_PATH}")

RUNTIME_SECONDS = 10  # duration to run TX/RX per cycle

# TODO
# - Fix: "sink :warning: Soapy sink error: TIMEOUT"
# - Integrate Chelsea's comments from previous pr
# - Integrate AM scripts, address throttle block error and rerun on WSL
# - Integrate timed operation for flight



def install_requirements():
    try:
        subprocess.check_call(["conda", "env", "update", "--name", "conda_env", "--file", "environment.yml"])
    except subprocess.CalledProcessError as e:
        print("Failed to install some packages. Continuing anyway..\nError: {e}")


def run_script(script_path):
    try:
        if platform.system() == "Windows":
            return subprocess.Popen(["python", script_path])
        else:
            return subprocess.Popen(["python3", script_path], preexec_fn=os.setsid)
    except Exception as e:
        print(f"Error launching script {script_path}: {e}")
        return None



def terminate_process(proc):
    if proc is None:
        return
    try:    
        if platform.system() == "Windows":
            proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        print("Process already terminated.")
    except Exception as e:
        print(f"Error terminating process: {e}")


def save_to_csv(rx_file_path, tx_file_path, csv_file_path):
    if not os.path.exists(rx_file_path) or not os.path.exists(tx_file_path):
        print("rxdata.dat or txdata.dat not found. Please ensure the files exist in the correct directory.")
        return

    try:
        rx_data = np.fromfile(open(rx_file_path), dtype=np.complex64)
        tx_data = np.fromfile(open(tx_file_path), dtype=np.complex64)
    except Exception as e:
        print(f"Error reading binary data: {e}")
        return

    tx_data_last = tx_data[-500000:] if len(tx_data) >= 500000 else tx_data
    rx_data_last = rx_data[-500000:] if len(rx_data) >= 500000 else rx_data

    write_header = not os.path.exists(csv_file_path)

    try:
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
    except Exception as e:
        print(f"Error saving to CSV: {e}")



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