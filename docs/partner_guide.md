# BitNet FPGA Accelerator on DE10-Nano ??Partner Guide

## Table of Contents

1. [Background Concepts](#1-background-concepts)
   - [What is Chisel?](#11-what-is-chisel)
   - [What is a BitNet / Ternary AI Accelerator?](#12-what-is-a-bitnet--ternary-ai-accelerator)
   - [What is DE10-Nano?](#13-what-is-de10-nano)
   - [What is BitMamba?](#14-what-is-bitmamba)
2. [System Overview](#2-system-overview)
3. [Hardware Architecture](#3-hardware-architecture)
   - [SoC Block Diagram](#31-soc-block-diagram)
   - [BitNet Accelerator Pipeline](#32-bitnet-accelerator-pipeline)
   - [Processing Element](#33-processing-element)
   - [Tile Array and Adder Tree](#34-tile-array-and-adder-tree)
   - [Weight Encoding (2-bit Ternary)](#35-weight-encoding-2-bit-ternary)
   - [Weight Streamer and Double Buffering](#36-weight-streamer-and-double-buffering)
   - [Activation Buffer](#37-activation-buffer)
   - [FSM Controller](#38-fsm-controller)
4. [Software Architecture](#4-software-architecture)
   - [Memory Map](#41-memory-map)
   - [Register Map](#42-register-map)
   - [HPS Driver API](#43-hps-driver-api)
   - [Computation Flow](#44-computation-flow)
5. [Build Flow](#5-build-flow)
6. [Key Files](#6-key-files)

---

## 1. Background Concepts

### 1.1 What is Chisel?

If you are familiar with Verilog, you can think of **Chisel** (Constructive Hardware In a Scala Embedded Language) as a hardware construction language that compiles down to Verilog/SystemVerilog. It is developed at UC Berkeley.

Key differences from Verilog:

| Aspect | Verilog | Chisel |
|--------|---------|--------|
| Language | Standalone HDL | Embedded in Scala (a JVM language) |
| Parameterization | `generate`, `parameter` | Full Scala: loops, generics, case classes |
| Type safety | Weak | Strong (bit-width errors caught at compile time) |
| Testing | External testbench (ModelSim, VCS) | Built-in `ChiselTest` (runs in JVM, no simulator license) |
| Output | N/A (is the target) | Generates synthesizable SystemVerilog |

**Why Chisel here?** The accelerator is highly parameterized (PE count, tile count, encoding scheme, buffer depths). Chisel lets us change one config object and regenerate the entire design ??something that would require massive `generate` blocks in Verilog. The generated SystemVerilog (`BitNetAccelerator.sv`) is what Quartus actually synthesizes; you never need to touch Chisel to work with the FPGA bitstream.

### 1.2 What is a BitNet / Ternary AI Accelerator?

Standard neural networks use FP32 or FP16 weights, requiring DSP multipliers for every multiply-accumulate (MAC). **BitNet b1.58** is a quantization scheme from Microsoft Research where every weight is constrained to **{-1, 0, +1}** ??only three possible values (1.58 bits of information).

This transforms multiplication into simple logic:

```
weight = +1  ?? output = +activation
weight =  0  ?? output = 0
weight = -1  ?? output = -activation
```

No multiplier needed ??just a MUX. This is why the accelerator uses **zero DSP blocks**. Each Processing Element (PE) is roughly 8 ALMs of pure LUT logic on Cyclone V.

The core operation is still matrix-vector multiplication: `y = W ? x`, where `W` is an M?K ternary weight matrix and `x` is a K-element INT8 activation vector. The accelerator computes this one row at a time, 128 weights in parallel per cycle.

### 1.3 What is DE10-Nano?

The **Terasic DE10-Nano** is a low-cost (~$110) development board built around the **Intel (Altera) Cyclone V SoC** (part number 5CSEBA6U23I7). It is Intel's equivalent of Xilinx's Zynq ??a dual-core ARM Cortex-A9 hard processor system (HPS) tightly coupled with FPGA fabric on one die.

Comparison to Xilinx Zynq ecosystem:

| Concept (Xilinx) | Equivalent (Intel/Altera) |
|-------------------|--------------------------|
| Vivado | Quartus Prime |
| Block Design / IPI | Platform Designer (QSys) |
| AXI interconnect | Avalon-MM / Avalon-ST |
| PS (Processing System) | HPS (Hard Processor System) |
| PL (Programmable Logic) | FPGA Fabric |
| .bit bitstream | .sof / .rbf bitstream |
| Zynq-7020 | Cyclone V SoC (5CSEBA6) |

Key Cyclone V 5CSEBA6 resources:

- **Logic Elements:** 41,910 ALMs (~110K LE equivalent)
- **Memory:** 5,570 Kbit M10K blocks
- **DSP:** 112 variable-precision DSP blocks (unused by this design)
- **HPS:** Dual ARM Cortex-A9 @ 800 MHz, 1 GB DDR3 (shared with FPGA)
- **Bridges:** HPS-to-FPGA (h2f), lightweight HPS-to-FPGA (h2f_lw), FPGA-to-SDRAM (f2sdram)

### 1.4 What is BitMamba?

**Mamba** is a class of language models based on Selective State Space Models (SSMs) rather than Transformers. Unlike Transformers which need O(n^2) attention, Mamba processes sequences in O(n) time, making it more efficient for long-context inference.

**BitMamba** combines Mamba architecture with BitNet b1.58 quantization ??a 255M-parameter language model where all linear layers use ternary weights. This makes it ideal for FPGA inference: the model is small enough to fit in DDR3, and every matrix multiplication can be done with our DSP-free accelerator.

The FPGA handles the computationally intensive ternary matrix-vector products, while the ARM cores running Linux handle everything else (state space computation, normalization, tokenization, sampling).

---

## 2. System Overview

```
?Œâ??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€??
??                       DE10-Nano Board                          ??
??                                                                ??
?? ?Œâ??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€??   ?Œâ??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€????
?? ??   HPS (ARM x2)      ??   ??       FPGA Fabric             ????
?? ??                     ??   ??                               ????
?? ?? Linux + BitMamba    ??   ?? ?Œâ??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?? ????
?? ?? inference software  ??   ?? ?? BitNet Accelerator      ?? ????
?? ??                     ??   ?? ?? 128 PEs, 0 DSPs         ?? ????
?? ?? ARM does:           ??   ?? ??                         ?? ????
?? ?? - tokenization      ??   ?? ?? FPGA does:              ?? ????
?? ?? - SSM computation   ??   ?? ?? - ternary matmul W?x   ?? ????
?? ?? - normalization     ??   ?? ?? - weight streaming      ?? ????
?? ?? - float dequant     ??   ?? ?? - accumulation          ?? ????
?? ?? - sampling          ??   ?? ??                         ?? ????
?? ??        ??           ??   ?? ?”â??€?€?€?€?¬â??€?€?€?€?€?€?€?€?€?¬â??€?€?€?€?€?€?€?€?? ????
?? ??        ??           ??   ??       ??         ??            ????
?? ?”â??€?€?€?€?€?€?€?€?¼â??€?€?€?€?€?€?€?€?€?€?€??   ?”â??€?€?€?€?€?€?€?¼â??€?€?€?€?€?€?€?€?€?¼â??€?€?€?€?€?€?€?€?€?€?€?€????
??           ??                         ??         ??              ??
??    ?Œâ??€?€?€?€?€?´â??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?´â??€?? ?Œâ??€?€?´â??€?€?€?€?€?€?€??     ??
??    ?? h2f_lw bridge (control path)      ?? ?? f2sdram   ??     ??
??    ?? Avalon-MM slave: regs + act + res ?? ?? (weights) ??     ??
??    ?”â??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?? ?”â??€?€?¬â??€?€?€?€?€?€?€??     ??
??                                                 ??              ??
??    ?Œâ??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?´â??€?€?€?€?€?€??     ??
??    ??             DDR3 1 GB (shared)                     ??     ??
??    ?? [Linux memory ........] [Weight matrices ........] ??     ??
??    ?”â??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€??     ??
?”â??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€??
```

Two data paths connect HPS and FPGA:

1. **Lightweight bridge (h2f_lw):** HPS writes activations and control registers, reads results. Low bandwidth, register-level access. Maps to physical address 0xFF200000.
2. **FPGA-to-SDRAM (f2sdram):** Accelerator reads packed weights directly from DDR3 via 256-bit burst reads. High bandwidth, DMA-like.

---

## 3. Hardware Architecture

### 3.1 SoC Block Diagram

```
                    Platform Designer (QSys) System
?Œâ??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€??
??                                                              ??
??  HPS (hps_0)                                                 ??
??  ?œâ??€ h2f_lw_axi_master ?€?€??bitnet_accel_0.avs_slave         ??
??  ??                         (regs, activations, results)     ??
??  ?œâ??€ f2sdram ?„â??€?€?€?€?€?€?€?€?€?€?€?€ bitnet_accel_0.avm_master        ??
??  ??                         (256-bit DDR3 weight reads)      ??
??  ?”â??€ DDR3 controller                                         ??
??                                                              ??
??  Other FPGA peripherals:                                     ??
??  ?œâ??€ custom_leds (8-bit LED, Avalon-MM slave)                ??
??  ?œâ??€ pio64_in / pio64_out (64-bit parallel I/O)              ??
??  ?”â??€ debounce, edge_detect, altsource_probe                  ??
??                                                              ??
??  Clocking: PLL 50 MHz ??100 MHz system clock                ??
?”â??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€??
```

### 3.2 BitNet Accelerator Pipeline

The accelerator computes one row of `y = W ? x` per iteration. For each row:

```
DDR3 Weights ?€?€??Weight     ?€?€??Weight Decoder  ?€?€??128 PEs    ?€?€??Tile      ?€?€??Accum  ?€?€??Result
(256-bit bursts)  Streamer      (2-bit ternary)      (ternary       AdderTree     ulator     Buffer
                  (double-buf)  (32 groups ??        multiply)      (4 tiles      (K-dim     (32-bit
                                 128 en/sign)                        ??1 sum)      tiles)     raw)
                                      ??
Activation  ?€?€??Activation ?€?€?€?€?€?€?€?€?€?€?€??
(INT8, from     Buffer
 HPS writes)   (128 banks)
```

**Latency per tile:** 1 (PE register) + 5 (intra-tile tree) + 2 (inter-tile tree) = **8 clock cycles**

**Tiles per row:** ceil(K / 128). For K=4096, that is 32 tiles.

### 3.3 Processing Element

Each PE implements a DSP-free ternary multiplier ??the core trick that makes this design possible on Cyclone V without using any DSP blocks.

```
              ?Œâ??€?€?€?€?€?€?€?€?€?€?€?€?€?€??
  act[7:0] ?€?€?ºâ?               ??
  (INT8)      ?? enable=0 ??0 ?‚â??€??result[8:0]
  enable   ?€?€?ºâ?  sign=0   ??+act ??  (signed 9-bit)
  sign     ?€?€?ºâ?  sign=1   ??-act ??
              ?”â??€?€?€?€?€?€?€?€?€?€?€?€?€?€??
```

In Verilog terms, the logic is:

```verilog
assign result = ~enable ? 9'sd0 :
                ~sign   ? {1'b0, act} :    // +act (zero-extend)
                          -{1'b0, act};    // -act (negate)
```

Each PE uses approximately 8 ALMs. With 128 PEs, the PE array costs ~1,000 ALMs total.

### 3.4 Tile Array and Adder Tree

The 128 PEs are organized into **4 tiles of 32 PEs** each (4?32=128).

```
Tile 0 (32 PEs) ?€?€??AdderTree(32??) ?€?€??
Tile 1 (32 PEs) ?€?€??AdderTree(32??) ?€?€??
Tile 2 (32 PEs) ?€?€??AdderTree(32??) ?€?€?¼â??€??AdderTree(4??) ?€?€??Accumulator
Tile 3 (32 PEs) ?€?€??AdderTree(32??) ?€?€??

Intra-tile: 5 pipeline stages (log2(32) = 5 levels)
Inter-tile: 2 pipeline stages (ceil(log2(4)) = 2 levels)
```

Each adder tree level is registered (1 pipeline stage per level), giving deterministic latency. Bit widths grow by 1 bit per level to prevent overflow.

### 3.5 Weight Encoding (2-bit Ternary)

Ternary weights have 3 possible values, so 5 weights require log2(3^5) = 7.92 bits. We pack 5 ternary weights into 8 bits using **base-3 encoding**:

```
encoded_byte = t0 + t1?3 + t2?9 + t3?27 + t4?81

where each ti ??{0, 1, 2} maps to:
  0 ??weight = -1  (enable=1, sign=1)
  1 ??weight =  0  (enable=0)
  2 ??weight = +1  (enable=1, sign=0)
```

Valid range: 0??42 (3^5 - 1). Values 243??55 are invalid and disable all 5 PEs.

Each 256-bit DDR3 beat carries **32 groups ? 8 bits = 256 bits**, decoding to **128 weight slots**.

The decoder is implemented as a 256-entry ROM (purely combinational, ~20 ALMs per instance, 32 instances total).

### 3.6 Weight Streamer and Double Buffering

The weight streamer is an Avalon-MM master that reads packed weights from DDR3 in bursts.

```
                    ?Œâ??€?€?€?€?€?€?€?€??
  DDR3 (f2sdram) ?€?€?ºâ?  FIFO A ?‚â??€??
    256-bit bursts  ?œâ??€?€?€?€?€?€?€?€?? ?œâ??€??swap ?€?€??Compute Core
                    ?? FIFO B ?‚â??€??
                    ?”â??€?€?€?€?€?€?€?€??

State machine:
  sIdle ??sRead (issue burst) ??sFill (receive data) ??done
```

**Double buffering:** While the compute core consumes weights from FIFO A for the current row, the streamer prefetches the next row's weights into FIFO B. On row completion, the FIFOs swap roles. This hides DDR3 latency.

Address computation: `addr = WEIGHT_BASE + rowIdx ? tilesPerRow ? 32`

### 3.7 Activation Buffer

INT8 activations are written by HPS via the Avalon-MM slave and stored in **128 independent BRAM banks** (one per PE). Bank-interleaved addressing:

```
activation[i] ??bank[i % 128], address[i / 128]
```

This allows all 128 PEs to read their activation simultaneously in **2 cycles** (1 cycle issue + 1 cycle BRAM latency). The same activation buffer persists across all M rows (activations are reused).

### 3.8 FSM Controller

The top-level FSM orchestrates the full matrix-vector product:

```
            ?Œâ??€?€?€?€?€?€?€??
            ??sIdle  ?‚â??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€??
            ?”â??€?€?¬â??€?€?€?? (await START pulse from HPS)     ??
                ??                                        ??
            ?Œâ??€?€?¼â??€?€?€?€?€??                                 ??
            ?‚sStartRow ?? (begin prefetch for row 0)      ??
            ?”â??€?€?¬â??€?€?€?€?€??                                 ??
                ??                                        ??
            ?Œâ??€?€?¼â??€?€?€?€?€??                                 ??
            ?‚sWaitFill ?? (wait for DDR3 burst complete)  ??
            ?”â??€?€?¬â??€?€?€?€?€??                                 ??
                ??                                        ??
     ?Œâ??€?€?€?€?€?€?€?€?€?¼â??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€??     ??
     ?? sSwapAndGo ??sStartPrefetch ??sLoadTile   ??     ??
     ?? ??sWaitTile ??sConsumeWeight               ??     ??
     ?? (loop: tile-by-tile weight consumption)    ??     ??
     ?? ??sWaitPipeline (flush adder pipeline)     ??     ??
     ?”â??€?€?€?€?€?€?€?€?€?¬â??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€??     ??
                ??                                        ??
            ?Œâ??€?€?¼â??€?€?€?€?€??                                 ??
            ??sRowNext ?? (next row or done?)              ??
            ?”â??€?€?¬â??€?¬â??€?€??                                 ??
                ?? ??                                     ??
          done? ?? ??more rows                            ??
                ?? ?”â??€?€?€?€?€??sSwapAndGo (loop)             ??
            ?Œâ??€?€?¼â??€??                                     ??
            ?‚sDone ?‚â??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€??
            ?”â??€?€?€?€?€?? (set DONE flag, return to idle)
```

---

## 4. Software Architecture

### 4.1 Memory Map

From the HPS (ARM) perspective:

```
Physical Address Range          Usage
?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
0x0000_0000 ??0x2FFF_FFFF      Linux DDR3 (768 MB)
0x3000_0000 ??0x3FFF_FFFF      Weight matrices in DDR3 (256 MB)
0xFF20_0000 ??0xFF3F_FFFF      Lightweight HPS-to-FPGA bridge (2 MB)
  ?”â??€ +0x0000                   BitNet accelerator registers
      +0x0080                   Activation write window
      +0x4000                   Result read window
```

### 4.2 Register Map

Avalon-MM slave: 15-bit byte address, 32-bit data, read latency = 1.

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| `0x00` | CTRL | W | Bit 0: START pulse (auto-clears, also clears DONE) |
| `0x04` | STATUS | R | Bit 0: BUSY, Bit 1: DONE |
| `0x08` | WEIGHT_BASE | R/W | DDR3 byte address of weight matrix |
| `0x0C` | DIM_M | R/W | Number of output rows (max 1024) |
| `0x10` | DIM_K | R/W | Reduction dimension (max 4096) |
| `0x14` | SHIFT_AMT | R/W | Requantization right-shift amount (0??1) |
| `0x18` | PERF_CYCLES | R | Clock cycles for last computation |
| `0x1C` | NUM_PES | R | Hardware PE count (128) |
| `0x20` | WEIGHTS_PER_BEAT | R | Weights decoded per 256-bit beat (128) |
| `0x24` | ENCODING_MODE | R | 0 = 2-bit ternary encoding |
| `0x0080`?“`0x207C` | ACT_DATA | W | Activation buffer (act[i] at offset 0x80 + i?4) |
| `0x4000`+ | RES_DATA | R | Result buffer (res[i] at offset 0x4000 + i*4, 32-bit signed) |

### 4.3 HPS Driver API

The C driver (`software/bitmamba_fpga/bitnet_fpga.h`) provides:

```c
// Initialize: mmap lightweight bridge + DDR3 region
int fpga_init(uint32_t ddr3_base, uint32_t ddr3_span);

// Load packed weight file into DDR3
int fpga_load_weights(const char *fpga_bin_path);

// Full matrix-vector multiply:
//   writes INT8 activations ??configures dims ??START ??poll DONE ??read results
void fpga_bitlinear(const int8_t *act, int32_t *result, int M, int K,
                    uint32_t weight_addr, int shift_amt);

// Float-to-float wrapper (ARM handles quant/dequant):
//   float input ??quantize to INT8 ??FPGA matmul ??dequant to float
void bitlinear_forward_fpga(const float *input, float *output,
                            int M, int K, uint32_t weight_addr,
                            float input_scale, float weight_scale);

// Cleanup
void fpga_cleanup(void);
```

### 4.4 Computation Flow

A single BitLinear layer inference proceeds as:

```
ARM (HPS)                          FPGA
?€?€?€?€?€?€?€?€?€                          ?€?€?€?€
1. Quantize float?’INT8
2. Write act[0..K-1] to ACT_DATA ?€?€??
3. Write WEIGHT_BASE, DIM_M, DIM_K ?€??
4. Write CTRL.START=1 ?€?€?€?€?€?€?€?€?€?€?€?€?€?€??
                                    5. Stream weights from DDR3 (row 0)
                                    6. For each row:
                                       a. Decode weights (2-bit ternary)
                                       b. 128 PEs: ternary multiply
                                       c. Adder tree reduction (8 cycles)
                                       d. Accumulate across K tiles
                                       e. Store 32-bit raw result
                                       f. Prefetch next row weights
                                    7. Set DONE flag
5. Poll STATUS.DONE ?„â??€?€?€?€?€?€?€?€?€?€?€?€?€?€
6. Read RES_DATA[0..M-1] ?„â??€?€?€?€?€?€?€?€?€
7. Dequantize INT32?’float (on ARM)
```

For models with M > 1024 (FPGA maximum), the HPS driver splits the operation into multiple FPGA invocations, each handling up to 1024 rows.

---

## 5. Build Flow

### Chisel ??SystemVerilog

```bash
cd bitnet/chisel/
# Set Java 11 (required by sbt/Chisel)
export JAVA_HOME="/c/Program Files/Eclipse Adoptium/jdk-11.0.29.7-hotspot"

sbt "runMain bitnet.BitNetAccelMain"
# Output: generated/BitNetAccelerator.sv
```

### FPGA Synthesis (Quartus 18.1)

```bash
# Ensure Quartus is on PATH
export PATH="/c/intelFPGA_lite/18.1/quartus/bin64:$PATH"

make sof           # QSys generate ??synthesis ??P&R ??output_files/*.sof
make rbf           # Convert .sof ??.rbf (SD card boot image)
make program_fpga  # Program via JTAG (live, volatile)
```

### HPS Software

```bash
make preloader     # Build SPL BSP
make uboot         # Build U-Boot
make dtb           # Compile device tree blob
```

Cross-compile C drivers with `arm-linux-gnueabihf-gcc` targeting the DE10-Nano's Linux.

---

## 6. Key Files

| Path | Description |
|------|-------------|
| `DE10_NANO_SoC_GHRD.v` | FPGA top-level (PLL, soc_system instantiation) |
| `DE10_NANO_SoC_GHRD.qsf` | Pin assignments, device config, source file list |
| `soc_system.qsys` | Platform Designer system (HPS + FPGA peripherals) |
| `bitnet_accel_hw.tcl` | BitNet accelerator QSys component definition |
| `bitnet/chisel/src/main/scala/bitnet/` | Chisel RTL source (13 modules) |
| `bitnet/chisel/generated/BitNetAccelerator.sv` | Generated SystemVerilog (Quartus input) |
| `software/bitmamba_fpga/bitnet_fpga.h` | HPS-side FPGA driver |
| `software/bitmamba_fpga/test_fpga_driver.c` | Driver test program |
| `Makefile` | Full build orchestration |



