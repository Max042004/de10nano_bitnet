# DE10-Nano BitNet Inference SoC

A complete FPGA SoC design that runs **BitMamba 255M** and **MNIST** neural network inference on the **Terasic DE10-Nano** (Cyclone V 5CSEBA6U23I7) — using zero DSP blocks. Built on Intel's Golden Hardware Reference Design (GHRD), extended with a custom BitNet b1.58 ternary-weight accelerator written in Chisel.

## What This Does

The ARM Cortex-A9 on the Cyclone V SoC runs Linux and handles quantization, normalization, and control. The FPGA fabric contains a 128-PE accelerator that streams ternary weights from DDR3 and computes matrix-vector products using only LUT logic. Together they run real neural network inference:

- **BitMamba 255M** — 255M-parameter Mamba2 language model, full transformer-free inference
- **MNIST digit recognition** — 3-layer BitNet MLP (784 -> 256 -> 128 -> 10), live PGM image input

## SoC Architecture

```
DE10-Nano (Cyclone V SoC)
│
├── HPS (ARM Cortex-A9 dual-core, Linux)
│   ├── DDR3 1GB (shared: Linux + model weights)
│   ├── h2f_lw_axi_master ──► BitNet slave (control/status, activations, results)
│   └── f2sdram bridge     ◄── BitNet master (256-bit DDR3 weight streaming)
│
├── FPGA Fabric (100 MHz via PLL)
│   ├── BitNetAccelerator (Chisel-generated, 128 PEs, 0 DSP)
│   │   ├── Avalon-MM Slave   — HPS writes activations, reads raw 32-bit results
│   │   ├── Avalon-MM Master  — burst-reads 256-bit packed weights from DDR3
│   │   ├── 128 Processing Elements (ternary multiply = pass/negate/zero)
│   │   ├── 7-level pipelined adder tree
│   │   └── Double-buffered weight prefetch (hides DDR3 latency)
│   ├── custom_leds (8-bit LED controller)
│   └── pio64_in / pio64_out (64-bit parallel I/O)
│
└── Platform Designer (soc_system.qsys)
    └── Interconnect, clock crossings, reset, SDRAM controller
```

## Repository Structure

```
ghrd_bitnet/
├── DE10_NANO_SoC_GHRD.v          # FPGA top-level (PLL, soc_system instantiation)
├── DE10_NANO_SoC_GHRD.qsf        # Pin assignments, device settings, HDL source list
├── DE10_NANO_SOC_GHRD.sdc        # Timing constraints
├── soc_system.qsys               # Platform Designer system definition
├── bitnet_accel_hw.tcl            # BitNet accelerator Platform Designer component
├── bitnet/                        # BitNet accelerator submodule (Chisel RTL + tests)
│   ├── chisel/src/main/scala/     #   Chisel source (13 modules)
│   ├── chisel/src/test/scala/     #   Test suites (8 files)
│   └── chisel/generated/          #   Generated SystemVerilog for Quartus
├── bitmamba.cpp-main/             # BitMamba 255M C++ inference engine
│   ├── src/                       #   Model implementation
│   ├── scripts/                   #   Weight export tools
│   └── build-arm/                 #   ARM cross-compiled build
├── software/
│   ├── mnist/                     # MNIST inference demo
│   │   ├── mnist_inference.c      #   3-layer MLP inference on FPGA
│   │   └── generated/             #   Pre-exported weights and test data
│   ├── bitmamba_fpga/             # BitMamba FPGA driver
│   │   ├── bitnet_fpga.h          #   FPGA driver (init, matmul, float path)
│   │   └── test_fpga_driver.c     #   Driver smoke tests
│   ├── bitnet_test/               # Hardware verification tests
│   │   └── bitnet_test_common.h   #   Shared mmap, register access, weight packing
│   └── spl_bsp/                   # Preloader BSP
├── ip/                            # Custom IP cores (LEDs, PIO64)
├── hps_isw_handoff/               # HPS hardware-software handoff
└── output_files/                  # Quartus compilation output (.sof, .rbf)
```

## Quick Start

### Prerequisites

- **Quartus Prime Lite 18.1** — `C:\intelFPGA_lite\18.1\quartus\bin64\` on PATH
- **sbt + Java 11** — for Chisel RTL (set `JAVA_HOME` to Eclipse Temurin 11)
- **ARM cross-compiler** — `arm-linux-gnueabihf-gcc` for HPS software
- **DE10-Nano** with SD card running Linux

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
# MNIST demo (cross-compile)
cd software/mnist
make CC=arm-linux-gnueabihf-gcc

# BitMamba FPGA driver test
cd software/bitmamba_fpga
make CC=arm-linux-gnueabihf-gcc
```

### Run on DE10-Nano

```bash
# MNIST inference (on the board, as root)
sudo ./mnist_inference image.pgm              # Single image
sudo ./mnist_inference --dir ./test_images/   # Directory of images
sudo ./mnist_inference --benchmark            # Embedded test set

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
| Max dimensions | M=1024, K=2048 |
| Output | Raw 32-bit accumulator (ARM dequantizes) |
| Adder tree | 7-level, fully pipelined (7 cycles) |
| Clock | 100 MHz (PLL from 50 MHz) |
| DSP blocks | **0** |

The accelerator outputs raw accumulator values instead of requantized INT8. This preserves full precision for ARM-side dequantization, which is critical for accurate 255M model inference.

For register map, weight packing format, and detailed architecture, see [`bitnet/README.md`](bitnet/README.md).

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
