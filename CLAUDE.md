# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## debug regulations
Writes discoverd bug in a bug list. Only fixing one bug at a time, remember testing before commit. If testing failed, recover 
modification and try to fix another bug which is more likely the root cause.
Don't modify generated SystemVerilog BitNet hardware, always modify Chisel BitNet hardware and using sbt to generates SystemVerilog.

## Project Overview

DE10-Nano Golden Hardware Reference Design (GHRD) extended with a DSP-free BitNet b1.58 inference accelerator. The accelerator uses ternary weights {-1, 0, +1} to replace multiplication with LUT logic, enabling neural network inference on Cyclone V without DSP blocks.

**Target:** Terasic DE10-Nano — Cyclone V SoC (5CSEBA6U23I7), Quartus Prime Lite 18.1

## Build Commands

### FPGA (from project root)

Quartus toolchain location: `C:\intelFPGA_lite\18.1\quartus\bin64\` (e.g. `quartus_sh.exe`). Ensure this is on PATH before running make targets.

```bash
make sof              # Generate QSys + full Quartus compile → output_files/*.sof
make rbf              # Convert .sof to .rbf for SD card boot
make program_fpga     # Program FPGA via JTAG
make qsys_edit        # Open Platform Designer GUI
make quartus_edit     # Open Quartus GUI
make clean            # Remove stamp files (triggers rebuild)
make scrub_clean      # Deep clean to barebones state
make help             # List all targets
```

### Chisel RTL (from bitnet/chisel/)

Requires sbt and Java 11. Set JAVA_HOME to Eclipse Temurin 11 before running sbt:

```bash
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-11.0.29.7-hotspot
sbt compile                          # Compile Chisel sources
sbt test                             # Run all tests (8 test suites)
sbt "testOnly bitnet.<TestName>"     # Run single test suite
sbt "runMain bitnet.BitNetAccelMain" # Generate SystemVerilog → generated/BitNetAccelerator.sv
```

### HPS Software/Boot (from project root)

```bash
make preloader        # Build SPL BSP
make uboot            # Build U-Boot
make dts              # Generate device tree source from QSys
make dtb              # Compile device tree blob
```

## Architecture

### SoC System (soc_system.qsys)

```
HPS (ARM Cortex-A9 dual-core)
├── h2f_lw_axi_master ──► BitNet slave (control/status, activation writes, result reads)
├── f2sdram            ◄── BitNet master (128-bit DDR3 weight streaming)
└── DDR3 controller (shared memory for weights + Linux)

FPGA Fabric
├── BitNetAccelerator (generated from Chisel)
├── custom_leds (8-bit LED controller, Avalon-MM slave)
├── pio64_in / pio64_out (64-bit parallel I/O)
└── debounce, edge_detect, altsource_probe (utility IP)
```

### BitNet Accelerator Pipeline

HPS writes INT8 activations and configures dimensions via Avalon-MM slave, then pulses START. The accelerator streams 2-bit packed weights from DDR3 (64 weights per 128-bit beat), decodes to enable/sign pairs, runs through 64 parallel processing elements (ternary multiply = pass/negate/zero), reduces via a 6-level pipelined adder tree (3 pipeline stages), accumulates across K-dimension tiles, and requantizes (shift+clamp) to INT8 output.

See `bitnet/README.md` for register map, weight packing format, and HPS software examples. See `bitnet/CLAUDE.md` for Chisel module details.

## Key Files

| File | Purpose |
|------|---------|
| `DE10_NANO_SoC_GHRD.v` | FPGA top-level module (instantiates soc_system) |
| `DE10_NANO_SoC_GHRD.qsf` | Pin assignments, device settings, HDL source list |
| `DE10_NANO_SOC_GHRD.sdc` | Timing constraints (3× 50 MHz clocks, JTAG) |
| `soc_system.qsys` | Platform Designer system definition |
| `Makefile` | Full build flow orchestration |
| `ip/` | Custom IP cores (custom_leds, pio64, debounce, edge_detect) |
| `bitnet/chisel/src/main/scala/bitnet/` | Chisel RTL source (13 modules) |
| `bitnet/chisel/src/test/scala/bitnet/` | Chisel test suites (8 files) |
| `bitnet/chisel/generated/` | Generated SystemVerilog for Quartus import |
| `hps_isw_handoff/` | HPS hardware-software handoff (SDRAM calibration) |
| `software/spl_bsp/` | Preloader BSP generated code |

## Platform Designer Integration

To add the BitNet accelerator to the QSys system:
1. Generate SV: `sbt "runMain bitnet.BitNetAccelMain"` in `bitnet/chisel/`
2. Import `bitnet/chisel/generated/BitNetAccelerator.sv` into Quartus project
3. Create Platform Designer component with Avalon-MM slave (12-bit addr, 32-bit data, read latency 1) and Avalon-MM master (32-bit addr, 128-bit data, variable latency with readdatavalid)
4. Connect slave to `h2f_lw_axi_master`, master to `f2sdram` bridge
5. Assign slave base address in lightweight bridge space (default 0xFF200000)

## Custom IP TCL Definitions

Custom IP for Platform Designer is defined via `*_hw.tcl` files in the project root:
- `custom_leds_hw.tcl` — 8-bit LED controller
- `pio64_in_hw.tcl` / `pio64_out_hw.tcl` — 64-bit parallel I/O

Implementation files live in `ip/<name>/`. Follow the same pattern (TCL component definition + SV/V implementation in ip/) when creating new Platform Designer components.
