# FPGA Build & Deploy Workflow

After modifying Chisel BitNet hardware, follow these steps to rebuild and deploy to DE10-Nano.

## Prerequisites

- Windows host with Quartus Prime Lite 18.1
- WSL Ubuntu-22.04 with `arm-linux-gnueabihf-gcc` and `sshpass`
- DE10-Nano at `root@192.168.1.108` (passwd: root)
- Java 11 (Eclipse Temurin) for sbt

## 1. Compile & Test Chisel

```bash
cd bitnet/chisel
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-11.0.29.7-hotspot
sbt test                             # Run all 8 test suites (72 tests)
sbt "runMain bitnet.BitNetAccelMain" # Generate SystemVerilog → generated/BitNetAccelerator.sv
```

## 2. Regenerate QSys

If port widths or interfaces changed in `bitnet_accel_hw.tcl`, QSys must be regenerated.

```bash
cd C:\intelFPGA_lite\18.1\ghrd_bitnet
set PATH=C:\intelFPGA_lite\18.1\quartus\bin64;C:\intelFPGA_lite\18.1\quartus\sopc_builder\bin;%PATH%
set QUARTUS_ROOTDIR=C:\intelFPGA_lite\18.1\quartus

qsys-generate soc_system.qsys --synthesis=VERILOG --output-directory=soc_system --family="Cyclone V" --part=5CSEBA6U23I7
```

Skip this step if only internal Chisel logic changed (no port width or TCL changes).

## 3. Quartus Compile

```bash
quartus_sh --flow compile DE10_NANO_SoC_GHRD
```

Takes ~5-7 minutes. Check results:

```bash
# Resource usage
grep -E "Logic utilization|Total registers|block memory|DSP" output_files/DE10_NANO_SoC_GHRD.flow.rpt

# Timing (Fmax)
grep -A5 "Slow 1100mV 100C Model Fmax Summary" output_files/DE10_NANO_SoC_GHRD.sta.rpt | grep MHz
```

## 4. Program FPGA via JTAG (using Linux FPGA manager not works, so via JTAG is necessary)

Program the .sof directly to DE10-Nano over USB-Blaster JTAG (no RBF conversion needed):

```bash
quartus_pgm -m jtag -o "p;output_files/DE10_NANO_SoC_GHRD.sof@2"
```

Note: `@2` targets the FPGA device in the JTAG chain (device index 2 on DE10-Nano).

## 5. Cross-Compile Software (WSL)

```bash
wsl -d Ubuntu-22.04 -- bash -c "cd /mnt/c/intelFPGA_lite/18.1/ghrd_bitnet/bitmamba.c && make clean && make arm-fpga"
```

Output: `bitmamba_arm_fpga` (ARM Cortex-A9 binary with FPGA offload)

## 6. Deploy Binary to DE10-Nano

```bash
wsl -d Ubuntu-22.04 -- bash -c '
  sshpass -p "root" scp -o StrictHostKeyChecking=no \
    /mnt/c/intelFPGA_lite/18.1/ghrd_bitnet/bitmamba.c/bitmamba_arm_fpga \
    root@192.168.1.108:/root/bitmamba.c/
'
```

## 7. Test

```bash
# Quick FPGA driver test and checks token/s
cd /root/bitmamba.c
./bitmamba_arm_fpga --fpga bitmamba_1b.fpga.bin bitmamba_1b.bin "The History of AI" tokenizer 0.7 1.1 0.05 0.9 40 20


# Full benchmark (compare FPGA vs CPU-only)
./bench_fpga_vs_cpu.sh
```

## Files Modified Per Change

| What changed | Files to update |
|---|---|
| Chisel logic only (no port changes) | Chisel sources → `sbt test` → `sbt runMain` → step 3-8 |
| Port widths or interfaces | Above + `bitnet_accel_hw.tcl` → step 2-8 |
| Register map (offsets) | Above + `bitnet_fpga.h`, `bitnet_test_common.h` → step 5-8 |
| FPGA driver constants | `bitnet_fpga.h` → step 5-8 only (no FPGA rebuild) |
