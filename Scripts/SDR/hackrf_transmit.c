#include <stdio.h>
#include <stdlib.h>
#include <hackrf.h>
#include <unistd.h>

#define SAMPLE_RATE 2000000  // 2 MHz
#define FREQUENCY 100100000  // 100.1 MHz
#define TX_GAIN 40  // Transmission Gain

// Sample transmission function
int transmit_callback(hackrf_transfer* transfer) {
    static uint8_t data = 0;
    for(int i = 0; i < transfer->buffer_length; i++) {
        transfer->buffer[i] = data++;
    }
    return 0;
}

int main() {
    hackrf_device* device;
    int status = hackrf_init();
    if(status != HACKRF_SUCCESS) {
        printf("HackRF initialization failed!\n");
        return -1;
    }

    status = hackrf_open(&device);
    if(status != HACKRF_SUCCESS) {
        printf("Failed to open HackRF device!\n");
        return -1;
    }

    hackrf_set_sample_rate(device, SAMPLE_RATE);
    hackrf_set_freq(device, FREQUENCY);
    hackrf_set_txvga_gain(device, TX_GAIN);
    hackrf_start_tx(device, transmit_callback, NULL);

    printf("Transmitting on %.1f MHz...\n", FREQUENCY / 1e6);
    sleep(10);  // Transmit for 10 seconds

    hackrf_stop_tx(device);
    hackrf_close(device);
    hackrf_exit();
    printf("Transmission stopped.\n");

    return 0;
}