# DE10-Nano BitNet Inference SoC

A complete FPGA SoC design that runs **BitNet b1.58** LLM inference on the **Terasic DE10-Nano** (Cyclone V 5CSEBA6U23I7) — using **zero DSP blocks**. Built on Intel's Golden Hardware Reference Design (GHRD), extended with a custom **T-MAC (table-based MAC)** accelerator written in Chisel.

Primary target model: **microsoft/bitnet-b1.58-2B-4T**. Secondary: **BitMamba 1B**.

## What This Does

The ARM Cortex-A9 runs Linux and handles tokenization, RMSNorm, RoPE, attention, sampling, and (TurboQuant-quantized) KV cache management. The FPGA fabric contains a **T-MAC accelerator** that streams ternary weights from DDR3 as `nibble + sign` arrays, walks them through 32 parallel LUT-lookup engines, reduces via a 6-level pipelined adder tree, and writes the requantized result back. The main GEMV loop contains **no multipliers, no DSP blocks** — just BRAM, MUXes, and adders.

## SoC Architecture

```
DE10-Nano (Cyclone V SoC)
|
+-- HPS (ARM Cortex-A9 dual-core, Linux)
|   +-- DDR3 1GB (shared: Linux + ternary weights + KV cache)
|   +-- h2f_lw_axi_master --> TMacAccelerator slave (control/status, activations, results)
|   +-- f2sdram bridge     <-- TMacAccelerator master (128-bit DDR3 weight streaming)
|
+-- FPGA Fabric (100 MHz via PLL)
|   +-- TMacAccelerator (Chisel-generated, 32 engines, 0 DSP)
|   |   +-- Avalon-MM Slave   - HPS configures dims, DDR3 addresses, supplies N3 = K/3
|   |   +-- Avalon-MM Master  - burst-reads 128-bit nibble + sign streams from DDR3
|   |   +-- LutBuilder         - 3-stage pipelined LUT construction (read -> compute -> write)
|   |   +-- LutBram (banks)    - 16-entry x 16-bit LUTs feeding the compute core
|   |   +-- 32 T-MAC engines   - 16:1 LUT MUX + 2's-complement sign correction
|   |   +-- 6-level adder tree (fully pipelined)
|   |   +-- Row accumulator + requantize (shift + clamp)
|   |   +-- Double-buffered weight prefetch (nibBuf A/B + signBuf A/B)
|   +-- custom_leds (8-bit LED controller)
|   +-- pio64_in / pio64_out (64-bit parallel I/O)
|
+-- Platform Designer (soc_system.qsys)
    +-- Interconnect, clock crossings, reset, SDRAM controller (100 MHz)
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
sbt "runMain bitnet.TMacAccelMain"

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

## T-MAC Accelerator

The accelerator is the core of this project. It replaces traditional per-element multiply-accumulate with **table-based lookup**: for every 3 activations `(a0, a1, a2)`, the 16 distinct sums of the form `±a0 ± a1 ± a2` are pre-computed once into a LUT BRAM, and the weight matrix is then walked as a stream of 4-bit nibble indices + 1-bit signs. The main loop has no multipliers.

Key specs:

| Parameter | Value |
|-----------|-------|
| Compute model | T-MAC (table-based MAC), group size = 3 |
| Engines | 32 parallel LUT lookups per cycle |
| LUT entries / group | 16 × INT16 (4-bit nibble index) |
| Avalon Master | 128-bit, burst DDR3 reads (nibble + sign streams) |
| Avalon Slave | 15-bit address, 32-bit data |
| Max dimensions | M = 1024, **K = 4096** (sized for BitMamba 1B `out_proj`) |
| Adder tree | 6-level, fully pipelined |
| Pipeline depth | 10 stages |
| Weight prefetch | Double-buffered (`nibBuf A/B` + `signBuf A/B`) with pipelined DDR3 sub-bursts |
| Clock | **100 MHz** (PLL from 50 MHz) |
| **DSP blocks** | **0** |

### 100 MHz Timing Closure

Closing 100 MHz on Cyclone V required several micro-architectural rewrites driven by TimeQuest worst-path reports:

| Optimization | Result |
|--------------|--------|
| **HPS supplies `REG_DIM_N3 = K/3` (offset `0x1C`)** | Removed a 17-level combinational divider for the non-power-of-2 `K/3`. Setup slack −19.846 ns → −4.744 ns; Fmax 34 MHz → 67 MHz. |
| **WeightStreamer per-row stride accumulator** | Eliminated `rowIdx × tilesPerRow` and `rowIdx × signBeats` variable×variable LUT multipliers. |
| **LutBuilder 3-stage pipeline (`sRead → sCompute → sWrite`)** | Broke the BRAM → 16 INT16 adders → BRAM single-cycle critical path. |
| **TMacComputeCore split LUT MUX / sign correction** | The 16:1 MUX on a 256-bit LUT word + conditional negate now span two pipeline stages. |
| **QSys `clk_0.clockFrequency` 50 → 100 MHz** | The PLL was producing 100 MHz but the QSys clock declaration mismatched STA constraints. |

For the full register map, weight format, T-MAC algorithm details, and Chisel module map, see [`bitnet/README.md`](bitnet/README.md) and [`docs/partner_guide.md`](docs/partner_guide.md).

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
sbt "runMain bitnet.TMacAccelMain"   # Generate SystemVerilog
```

Output goes to `bitnet/chisel/generated/TMacAccelerator.sv`. All 82 ScalaTest cases pass on the current `turbo` branch.

## License

GHRD base design by Terasic/Intel. BitMamba.cpp under MIT license. BitNet accelerator — see repository for details.
