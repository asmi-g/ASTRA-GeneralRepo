import subprocess
import threading
import time
import sys
import os
import signal
import platform
import numpy as np
import csv

from TX import TX as TXBlock
from RX import RX as RXBlock

DATA_DIR = "Data/"
TX_SCRIPT = "TX.py"
RX_SCRIPT = "RX.py"
TEMP_SCRIPT = "temp.py"
ML_SCRIPT = "../../RL-EnvConfig/inference.py"
CSV_FILE_PATH = os.path.join(DATA_DIR, "signal.csv")
RUNTIME_SECONDS = 10  # duration to run TX/RX per cycle

# TO DO
# - Fix: "sink :warning: Soapy sink error: TIMEOUT"
# - Integrate Chelsea's comments from previous pr
# - Integrate AM scripts, address throttle block error and rerun on WSL
# - Integrate timed operation for flight

ml_proc = None
terminate_ml = False

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
    if proc is None:
        return

    if proc.poll() is not None:
        # Process already exited
        return

    if platform.system() == "Windows":
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # Send SIGTERM to process group
        except ProcessLookupError:
            # Process already exited
            return

        # Wait a bit for graceful shutdown
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # If still alive, force kill process group
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

def cleanup_and_exit(signum=None, frame=None):
    print("Cleanup: Received termination signal, exiting gracefully...")
    global terminate_ml
    terminate_ml = True 
    # Terminate subprocesses safely
    # Uncomment ml_proc if using
    terminate_process(ml_proc)
    terminate_process(therm_proc)

    sys.exit(0)


def create_empty_csv():
    # So that ML model doesn't throw an error trying to read a file before its created
    # Ensure directory exists
    os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
    # Write header only (overwrite if exists)
    with open(CSV_FILE_PATH, 'w') as f:
        f.write("Index,TX Magnitude,RX Magnitude\n")

def save_to_csv(rx_file_path, tx_file_path, csv_file_path):
    rx_data = np.fromfile(open(rx_file_path), dtype=np.complex64)
    tx_data = np.fromfile(open(tx_file_path), dtype=np.complex64)

    tx_data_last = tx_data[-500000:] if len(tx_data) >= 500000 else tx_data
    rx_data_last = rx_data[-500000:] if len(rx_data) >= 500000 else rx_data

    write_header = not os.path.exists(csv_file_path)

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        #if write_header:
        #    writer.writerow(["Index", "TX Magnitude", "RX Magnitude"])

        for i in range(len(tx_data_last)):
            tx = tx_data_last[i]
            rx = rx_data_last[i] if i < len(rx_data_last) else 0
            writer.writerow([
                i,
                np.abs(tx),
                np.abs(rx)
            ])

def ml_watchdog():
    global ml_proc, terminate_ml
    while not terminate_ml:
        if ml_proc is None or ml_proc.poll() is not None:
            print("[ML] Starting ML inference script...")
            ml_proc = run_script(ML_SCRIPT)
        time.sleep(5)  # Check every 5 seconds

def run_tx_rx_cycle(duration=10):
    tx = TXBlock()
    rx = RXBlock()
    print("[INFO] Starting TX and RX...")
    tx.start()
    rx.start()

    time.sleep(duration)

    print("[INFO] Stopping TX and RX...")
    tx.stop()
    rx.stop()
    tx.wait()
    rx.wait()
    print("[INFO] TX and RX complete.")


def SDR_cycle():
    print("Launching TX and RX scripts...")
    run_tx_rx_cycle()

    print(f"Running for {RUNTIME_SECONDS} seconds...")
    time.sleep(RUNTIME_SECONDS)

    print("Saving to CSV...")
    save_to_csv(os.path.join(DATA_DIR, "rxdata.dat"),
                os.path.join(DATA_DIR, "txdata.dat"),
                CSV_FILE_PATH)
    print("Cycle complete.\n")


def main():
    install_requirements() 

    # Launch thermal sensing script once
    global therm_proc
    print("Launching temperature logging script...")
    therm_proc = run_script(TEMP_SCRIPT)
 
    create_empty_csv()

    # Launch ML inference script once
    global ml_proc
    print("Launching ML model...")
    ml_proc = run_script(ML_SCRIPT)
    # Start ML watchdog thread
    threading.Thread(target=ml_watchdog, daemon=True).start()

    signal.signal(signal.SIGINT, cleanup_and_exit)   # Ctrl+C
    signal.signal(signal.SIGTERM, cleanup_and_exit)  # kill command

    # SDR capture loop
    try:
        while True:
            SDR_cycle()
            time.sleep(2)  # Optional delay between cycles
    except KeyboardInterrupt:
        print("Terminating Model Operation...")
        cleanup_and_exit()


if __name__ == "__main__":
    main()
