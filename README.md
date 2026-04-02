# DE10-Nano BitNet Inference SoC

A complete FPGA SoC design that runs BitNet LLM neural network inference on the **Terasic DE10-Nano** (Cyclone V 5CSEBA6U23I7) — using zero DSP blocks. Built on Intel's Golden Hardware Reference Design (GHRD), extended with a custom BitNet b1.58 ternary-weight accelerator written in Chisel.

## What This Does

The ARM Cortex-A9 on the Cyclone V SoC runs Linux and handles quantization, normalization, and control. The FPGA fabric contains a 128-PE accelerator that streams ternary weights from DDR3 and computes matrix-vector products using only LUT logic. Together they run real neural network inference:

## SoC Architecture

```
DE10-Nano (Cyclone V SoC)
|
+-- HPS (ARM Cortex-A9 dual-core, Linux)
|   +-- DDR3 1GB (shared: Linux + model weights)
|   +-- h2f_lw_axi_master --> BitNet slave (control/status, activations, results)
|   +-- f2sdram bridge     <-- BitNet master (256-bit DDR3 weight streaming)
|
+-- FPGA Fabric (100 MHz via PLL)
|   +-- BitNetAccelerator (Chisel-generated, 128 PEs, 0 DSP)
|   |   +-- Avalon-MM Slave   - HPS configures dims, DDR3 addresses
|   |   +-- Avalon-MM Master  - burst-reads 256-bit packed weights from DDR3
|   |   +-- DDR3-mode activation/result transfer (DMA via f2sdram)
|   |   +-- 128 Processing Elements (ternary multiply = pass/negate/zero)
|   |   +-- 7-level pipelined adder tree
|   |   +-- Double-buffered weight prefetch (hides DDR3 latency)
|   |   +-- Pipelined DDR3 sub-bursts (overlapped fetch/compute)
|   +-- custom_leds (8-bit LED controller)
|   +-- pio64_in / pio64_out (64-bit parallel I/O)
|
+-- Platform Designer (soc_system.qsys)
    +-- Interconnect, clock crossings, reset, SDRAM controller
```

## Repository Structure

```
ghrd_bitnet/
+-- DE10_NANO_SoC_GHRD.v          # FPGA top-level (PLL, soc_system instantiation)
+-- DE10_NANO_SoC_GHRD.qsf        # Pin assignments, device settings, HDL source list
+-- DE10_NANO_SOC_GHRD.sdc        # Timing constraints
+-- soc_system.qsys               # Platform Designer system definition
+-- bitnet_accel_hw.tcl            # BitNet accelerator Platform Designer component
+-- bitnet/                        # BitNet accelerator submodule (Chisel RTL + tests)
|   +-- chisel/src/main/scala/     #   Chisel source (13 modules)
|   +-- chisel/src/test/scala/     #   Test suites (8 files)
|   +-- chisel/generated/          #   Generated SystemVerilog for Quartus
|
+-- reference-projects/
|   +-- bitmamba.c/                    # BitMamba C11 inference engine (255M + 1B)
|       +-- src/                       #   Model implementation (pure C11, FPGA-aware)
|       +-- scripts/                   #   Weight export and packing tools
|   +-- bitmamba.cpp-main/             # BitMamba C++ inference engine (reference)
|       +-- src/                       #   Model implementation
|       +-- scripts/                   #   Weight export tools 
|
+-- software/
|   +-- mnist/                     # MNIST inference demo
|   |   +-- mnist_inference.c      #   3-layer MLP inference on FPGA
|   |   +-- generated/             #   Pre-exported weights and test data
|   +-- bitmamba_fpga/             # BitMamba FPGA driver library
|   |   +-- bitnet_fpga.h          #   FPGA driver (init, matmul, DDR3-mode DMA)
|   |   +-- test_fpga_driver.c     #   Driver smoke tests
|   +-- bitnet_test/               # Hardware verification tests
|   |   +-- bitnet_test_common.h   #   Shared mmap, register access, weight packing
|   +-- spl_bsp/                   # Preloader BSP
+-- docs/                          # Documentation
|   +-- partner_guide.md           #   Detailed architecture guide (English)
|   +-- partner_guide_zh.md        #   Detailed architecture guide (Chinese)
|   +-- fpga-build-workflow.md     #   FPGA build flow walkthrough
+-- ip/                            # Custom IP cores (LEDs, PIO64)
+-- hps_isw_handoff/               # HPS hardware-software handoff
+-- output_files/                  # Quartus compilation output (.sof, .rbf)
```

## Quick Start

### Prerequisites

- **Quartus Prime Lite 18.1** — `C:\intelFPGA_lite\18.1\quartus\bin64\` on PATH
- **Java 11 + sbt** — for Chisel RTL generation
- **ARM cross-compiler** — `arm-linux-gnueabihf-gcc` for HPS software
- **DE10-Nano** with SD card running Linux

#### Install Java 11 (Eclipse Temurin)

Windows (winget):
```bash
winget install EclipseAdoptium.Temurin.11.JDK
# Default install path: C:\Program Files\Eclipse Adoptium\jdk-11.0.29.7-hotspot
# Set JAVA_HOME before running sbt:
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-11.0.29.7-hotspot
```

Ubuntu/Debian:
```bash
sudo apt install -y temurin-11-jdk
# Or via Adoptium APT repo:
# wget -qO- https://packages.adoptium.net/artifactory/api/gpg/key/public | sudo tee /etc/apt/trusted.gpg.d/adoptium.asc
# echo "deb https://packages.adoptium.net/artifactory/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/adoptium.list
# sudo apt update && sudo apt install -y temurin-11-jdk
export JAVA_HOME=/usr/lib/jvm/temurin-11-jdk-amd64
```

#### Install sbt

Windows (winget):
```bash
winget install sbt.sbt
```

Ubuntu/Debian:
```bash
echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | sudo tee /etc/apt/sources.list.d/sbt.list
curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo apt-key add
sudo apt update && sudo apt install -y sbt
```

#### Install ARM Cross-Compiler

Windows — install [Linaro GCC](https://releases.linaro.org/components/toolchain/binaries/latest-7/arm-linux-gnueabihf/) and add its `bin/` to PATH.

Ubuntu/Debian:
```bash
sudo apt install -y gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf
```

### Build FPGA

```bash
# Generate SystemVerilog from Chisel (if modifying RTL)
cd bitnet/chisel
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-11.0.29.7-hotspot
sbt "runMain bitnet.BitNetAccelMain"

# Full Quartus compile
cd ../..
make sof               # QSys generate + synthesis + place & route
make rbf               # Convert to SD card boot format
make program_fpga      # Program via JTAG (live)
```

### Build HPS Software

```bash
# BitMamba 1B (cross-compile on host)
cd bitmamba.c
make CC=arm-linux-gnueabihf-gcc FPGA=1

# BitMamba FPGA driver test
cd software/bitmamba_fpga
make CC=arm-linux-gnueabihf-gcc
```

### Run on DE10-Nano

```bash
# BitMamba 1B with FPGA acceleration (on the board, as root)
sudo ./bitmamba_arm_fpga bitmamba_1b.fpga.bin -i "prompt text"

# BitMamba 255M (on the board, as root)
sudo ./bitmamba model.bin -i "prompt text"
```

## BitNet Accelerator

The accelerator is the core of this project. Key specs:

| Parameter | Value |
|-----------|-------|
| Processing Elements | 128 (ternary multiply via LUT) |
| Avalon Master | 256-bit, burst DDR3 reads |
| Avalon Slave | 15-bit address, 32-bit data |
| Max dimensions | M=1024, K=4096 |
| Output | Raw 32-bit accumulator (ARM dequantizes) |
| Adder tree | 7-level, fully pipelined (7 cycles) |
| Weight prefetch | Double-buffered with pipelined DDR3 sub-bursts |
| Activation/result | DDR3-mode DMA via f2sdram (M-tile pipelined overlap) |
| Clock | 100 MHz (PLL from 50 MHz) |
| DSP blocks | **0** |

The accelerator outputs raw accumulator values instead of requantized INT8. This preserves full precision for ARM-side dequantization, which is critical for accurate 1B model inference. DDR3-mode transfers activations and results through the f2sdram bridge, enabling pipelined M-tile dequantization overlap for higher throughput.

For register map, weight packing format, and detailed architecture, see [`bitnet/README.md`](bitnet/README.md) and [`docs/partner_guide.md`](docs/partner_guide.md).

## Build Targets

```bash
make sof              # Full Quartus compile -> output_files/*.sof
make rbf              # Convert .sof to .rbf for SD card boot
make program_fpga     # Program FPGA via JTAG
make qsys_edit        # Open Platform Designer GUI
make quartus_edit     # Open Quartus GUI
make preloader        # Build SPL BSP
make uboot            # Build U-Boot
make dts              # Generate device tree source from QSys
make dtb              # Compile device tree blob
make clean            # Remove stamp files (triggers rebuild)
make scrub_clean      # Deep clean to barebones state
```

## Chisel RTL

The accelerator is written in Chisel 3 and generates SystemVerilog for Quartus.

```bash
cd bitnet/chisel
sbt compile                          # Compile Chisel sources
sbt test                             # Run all 8 test suites
sbt "testOnly bitnet.<TestName>"     # Run single test suite
sbt "runMain bitnet.BitNetAccelMain" # Generate SystemVerilog
```

Output goes to `bitnet/chisel/generated/BitNetAccelerator.sv`.

## License

GHRD base design by Terasic/Intel. BitMamba.cpp under MIT license. BitNet accelerator — see repository for details.
