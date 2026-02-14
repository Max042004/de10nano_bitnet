/*
 * bitnet_fpga.h — FPGA BitNet accelerator driver for BitMamba inference
 *
 * Provides:
 *   - fpga_init() / fpga_cleanup(): memory-mapped I/O setup
 *   - fpga_bitlinear(): INT8 activation -> FPGA matmul -> INT8 result
 *   - bitlinear_forward_fpga(): full float->float BitLinear with FPGA offload
 *
 * The accelerator has maxDimK=2048 and maxDimM=1024. For M > 1024, this
 * driver tiles over M in software (multiple FPGA invocations with the same
 * activations persisting in BRAM).
 */

#ifndef BITNET_FPGA_H
#define BITNET_FPGA_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

/* --- Memory map constants --- */
#define LW_BRIDGE_BASE  0xFF200000
#define LW_BRIDGE_SPAN  0x00200000   /* 2 MB */
#define BITNET_OFFSET   0x0

/* --- Register offsets (byte-addressed) --- */
#define REG_CTRL        0x00
#define REG_STATUS      0x04
#define REG_WEIGHT_BASE 0x08
#define REG_DIM_M       0x0C
#define REG_DIM_K       0x10
#define REG_SHIFT_AMT   0x14
#define REG_PERF_CYCLES 0x18
#define REG_ACT_BASE    0x80
#define REG_RES_BASE    0x4000

/* --- Hardware parameters --- */
#define FPGA_NUM_PES    128
#define FPGA_MAX_DIM_K  2048
#define FPGA_MAX_DIM_M  1024
#define FPGA_BYTES_PER_BEAT 32  /* 256-bit = 32 bytes */

/* --- Global state --- */
static int          fpga_devmem_fd = -1;
static volatile uint32_t *fpga_lw_bridge = NULL;
static volatile uint32_t *fpga_bitnet    = NULL;
static volatile uint32_t *fpga_ddr3      = NULL;
static uint32_t     fpga_ddr3_phys_base  = 0;
static uint32_t     fpga_ddr3_span       = 0;

/* --- Low-level register access --- */

static inline void fpga_reg_write(uint32_t offset, uint32_t val)
{
	fpga_bitnet[offset / 4] = val;
}

static inline uint32_t fpga_reg_read(uint32_t offset)
{
	return fpga_bitnet[offset / 4];
}

/* --- Init / Cleanup --- */

/*
 * fpga_init: map lightweight bridge and DDR3 weight region.
 * ddr3_base: physical address of FPGA weight region (e.g., 0x30000000)
 * ddr3_span: size of weight region in bytes
 * Returns 0 on success, -1 on failure.
 */
static int fpga_init(uint32_t ddr3_base, uint32_t ddr3_span)
{
	fpga_devmem_fd = open("/dev/mem", O_RDWR | O_SYNC);
	if (fpga_devmem_fd < 0) {
		perror("fpga_init: open /dev/mem");
		return -1;
	}

	fpga_lw_bridge = (volatile uint32_t *)mmap(NULL, LW_BRIDGE_SPAN,
		PROT_READ | PROT_WRITE, MAP_SHARED, fpga_devmem_fd, LW_BRIDGE_BASE);
	if (fpga_lw_bridge == MAP_FAILED) {
		perror("fpga_init: mmap lw_bridge");
		close(fpga_devmem_fd);
		return -1;
	}

	fpga_ddr3 = (volatile uint32_t *)mmap(NULL, ddr3_span,
		PROT_READ | PROT_WRITE, MAP_SHARED, fpga_devmem_fd, ddr3_base);
	if (fpga_ddr3 == MAP_FAILED) {
		perror("fpga_init: mmap ddr3");
		munmap((void *)fpga_lw_bridge, LW_BRIDGE_SPAN);
		close(fpga_devmem_fd);
		return -1;
	}

	fpga_bitnet = (volatile uint32_t *)((uint8_t *)fpga_lw_bridge + BITNET_OFFSET);
	fpga_ddr3_phys_base = ddr3_base;
	fpga_ddr3_span = ddr3_span;
	return 0;
}

static void fpga_cleanup(void)
{
	if (fpga_ddr3 && fpga_ddr3 != MAP_FAILED)
		munmap((void *)fpga_ddr3, fpga_ddr3_span);
	if (fpga_lw_bridge && fpga_lw_bridge != MAP_FAILED)
		munmap((void *)fpga_lw_bridge, LW_BRIDGE_SPAN);
	if (fpga_devmem_fd >= 0)
		close(fpga_devmem_fd);
}

/* --- Wait for DONE --- */

static int fpga_wait_done(int timeout_us)
{
	while (timeout_us > 0) {
		uint32_t st = fpga_reg_read(REG_STATUS);
		if (st & 0x2)
			return 0;
		usleep(10);
		timeout_us -= 10;
	}
	return -1;
}

/* --- Load FPGA weights into DDR3 --- */

/*
 * Load pre-converted FPGA weight binary from file into DDR3 region.
 * Returns 0 on success, -1 on failure.
 */
static int fpga_load_weights(const char *fpga_bin_path)
{
	FILE *f = fopen(fpga_bin_path, "rb");
	if (!f) {
		perror("fpga_load_weights: fopen");
		return -1;
	}

	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	fseek(f, 0, SEEK_SET);

	if ((uint32_t)size > fpga_ddr3_span) {
		fprintf(stderr, "fpga_load_weights: file %ld bytes exceeds DDR3 span %u\n",
			size, fpga_ddr3_span);
		fclose(f);
		return -1;
	}

	/* Read into DDR3 via mmap */
	size_t read = fread((void *)fpga_ddr3, 1, size, f);
	fclose(f);

	if ((long)read != size) {
		fprintf(stderr, "fpga_load_weights: short read %zu / %ld\n", read, size);
		return -1;
	}

	printf("Loaded %ld bytes of FPGA weights into DDR3 @ 0x%08X\n",
		size, fpga_ddr3_phys_base);
	return 0;
}

/* --- Core FPGA BitLinear --- */

/*
 * fpga_bitlinear: run ternary matrix-vector multiply on FPGA.
 *
 * Returns raw 32-bit accumulator values (no requantization) for full precision.
 *
 * activations: INT8 input vector, length K
 * K:           input dimension (must be <= FPGA_MAX_DIM_K, multiple of NUM_PES)
 * weight_base: DDR3 physical address of weight matrix
 * M:           output dimension (can exceed FPGA_MAX_DIM_M; software tiles)
 * stride:      bytes per weight row in DDR3 (tiles_per_row * BYTES_PER_BEAT)
 * results:     output INT32 raw accumulator vector, length M
 */
static void fpga_bitlinear(const int8_t *activations, int K,
                           uint32_t weight_base, int M,
                           int stride,
                           int32_t *results)
{
	int i;

	/* Write activations to FPGA buffer (persists across M-tiles) */
	for (i = 0; i < K; i++)
		fpga_reg_write(REG_ACT_BASE + i * 4, (uint32_t)(uint8_t)activations[i]);

	/* Set K (shift unused — FPGA outputs raw accumulator) */
	fpga_reg_write(REG_DIM_K, (uint32_t)K);
	fpga_reg_write(REG_SHIFT_AMT, 0);

	/* Tile over M dimension */
	int rows_done = 0;
	while (rows_done < M) {
		int tile_m = M - rows_done;
		if (tile_m > FPGA_MAX_DIM_M)
			tile_m = FPGA_MAX_DIM_M;

		/* Weight base for this tile: advance by rows_done * stride */
		uint32_t tile_weight_base = weight_base + (uint32_t)rows_done * stride;

		fpga_reg_write(REG_WEIGHT_BASE, tile_weight_base);
		fpga_reg_write(REG_DIM_M, (uint32_t)tile_m);

		/* Pulse START */
		fpga_reg_write(REG_CTRL, 0x1);

		/* Wait for completion (1 second timeout per tile) */
		if (fpga_wait_done(1000000) < 0) {
			fprintf(stderr, "fpga_bitlinear: timeout at M-tile offset %d\n", rows_done);
			memset(&results[rows_done], 0, tile_m * sizeof(int32_t));
			rows_done += tile_m;
			continue;
		}

		/* Read raw 32-bit accumulator results */
		for (i = 0; i < tile_m; i++)
			results[rows_done + i] = (int32_t)fpga_reg_read(REG_RES_BASE + i * 4);

		rows_done += tile_m;
	}
}

/* --- ARM-side quantization helpers --- */

/*
 * rms_norm_int8: RMS-normalize float vector with learned weights, then
 * quantize to INT8. Returns the quantization scale factor.
 *
 * x:           input float vector, length size
 * norm_weight: per-element normalization weights, length size
 * size:        vector length
 * out:         output INT8 vector, length size
 * Returns:     scale_x = 127.0 / max_abs(normalized)
 */
static float rms_norm_int8(const float *x, const float *norm_weight,
                           int size, int8_t *out)
{
	int i;

	/* RMS normalization */
	float sum_sq = 0.0f;
	for (i = 0; i < size; i++)
		sum_sq += x[i] * x[i];
	float rms = 1.0f / sqrtf(sum_sq / size + 1e-6f);

	/* Normalize + find max_abs */
	float max_abs = 0.0f;
	/* Use a temp buffer on stack for small sizes, heap for large */
	float *normalized = (float *)malloc(size * sizeof(float));
	for (i = 0; i < size; i++) {
		normalized[i] = x[i] * rms * norm_weight[i];
		float a = fabsf(normalized[i]);
		if (a > max_abs) max_abs = a;
	}

	/* Quantize to INT8 */
	float scale_x = 127.0f / (max_abs + 1e-5f);
	for (i = 0; i < size; i++) {
		float val = normalized[i] * scale_x;
		if (val > 127.0f) val = 127.0f;
		if (val < -128.0f) val = -128.0f;
		out[i] = (int8_t)roundf(val);
	}

	free(normalized);
	return scale_x;
}

/*
 * dequantize_results: convert raw accumulator FPGA output to float.
 * out[i] = raw_accum[i] / (scale_x * weight_scale)
 */
static void dequantize_results(const int32_t *fpga_out, int size,
                               float scale_x, float weight_scale,
                               float *out)
{
	float inv_scale = 1.0f / (scale_x * weight_scale);
	int i;
	for (i = 0; i < size; i++)
		out[i] = (float)fpga_out[i] * inv_scale;
}

/*
 * bitlinear_forward_fpga: full BitLinear layer using FPGA.
 *
 * FPGA returns raw accumulator values (no shift/clamp), preserving full
 * precision.  Dequantization to float happens on ARM:
 *   out[i] = raw_accum[i] / (scale_x * weight_scale)
 *
 * x:           input float vector, length K
 * K:           input dimension
 * M:           output dimension
 * norm_weight: RMS norm weights, length K
 * weight_base: DDR3 physical address of FPGA-encoded weights
 * weight_scale: quantization scale from export (1/mean_abs)
 * stride:      bytes per weight row in DDR3
 * out:         output float vector, length M
 */
static void bitlinear_forward_fpga(const float *x, int K, int M,
                                   const float *norm_weight,
                                   uint32_t weight_base,
                                   float weight_scale,
                                   int stride,
                                   float *out)
{
	/* a. RMS normalize + quantize to INT8 */
	int8_t *x_quant = (int8_t *)malloc(K);
	float scale_x = rms_norm_int8(x, norm_weight, K, x_quant);

	/* b. FPGA matmul (returns raw 32-bit accumulators) */
	int32_t *raw_results = (int32_t *)malloc(M * sizeof(int32_t));
	fpga_bitlinear(x_quant, K, weight_base, M, stride, raw_results);

	/* c. Dequantize to float (full precision, no shift loss) */
	dequantize_results(raw_results, M, scale_x, weight_scale, out);

	free(x_quant);
	free(raw_results);
}

#endif /* BITNET_FPGA_H */
