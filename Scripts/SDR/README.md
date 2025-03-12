## Installing HackRF on macOS

Before running this project, install HackRF using Homebrew:

```bash
brew install hackrf
```

Whenever you connect a HackRF device, test it with the following command in the terminal:
```bash
hackrf_info
```

Then try running your compiled program.
Compile: 
```bash
clang -o Scripts/SDR/hackrf_transmit Scripts/SDR/hackrf_transmit.c -I/opt/homebrew/include/libhackrf -L/opt/homebrew/lib -lhackrf
```
Run: 
```bash
./Scripts/SDR/hackrf_transmit
```

## Compiling and Running HackRF C Programs

This project contains C code that interacts with HackRF SDR devices. You'll need to compile it before running.

### 1️⃣ Installing HackRF

HackRF needs to be installed system-wide using Homebrew (macOS), Chocolatey (Windows), or APT (Linux).

- **macOS**: 
  ```bash
  brew install hackrf
  ```
- **Windows**: 
  ```bash
  choco install hackrf
  ```
    If using Windows, also install MinGW-w64 for gcc:
    ```bash
    choco install hackrf
    ```