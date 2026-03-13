/*
 * bitnet_fpga.h -- FPGA BitNet accelerator driver for BitMamba inference
 *
 * Provides:
 *   - fpga_init() / fpga_cleanup(): memory-mapped I/O setup
 *   - fpga_load_weights(): load pre-packed FPGA weights into DDR3
 *   - fpga_bitlinear(): INT8 activation -> FPGA matmul -> INT32 accumulator
 *   - bitlinear_forward_fpga(): full float->float BitLinear with FPGA offload
 */

#ifndef BITNET_FPGA_H
#define BITNET_FPGA_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

/* --- Memory map constants --- */
#define LW_BRIDGE_BASE  0xFF200000
#define LW_BRIDGE_SPAN  0x00200000   /* 2 MB */
#define BITNET_OFFSET   0x0000

/* --- Register offsets (byte-addressed) --- */
#define REG_CTRL             0x00
#define REG_STATUS           0x04
#define REG_WEIGHT_BASE      0x08
#define REG_DIM_M            0x0C
#define REG_DIM_K            0x10
#define REG_SHIFT_AMT        0x14
#define REG_PERF_CYCLES      0x18
#define REG_NUM_PES          0x1C
#define REG_WEIGHTS_PER_BEAT 0x20
#define REG_ENCODING_MODE    0x24
#define REG_ACT_BASE         0x80
#define REG_RES_BASE         0x8000

/* --- Expected hardware contract (DE10-Nano 128-PE ternary target) --- */
#define FPGA_NUM_PES         128
#define FPGA_MAX_DIM_K       4096
#define FPGA_MAX_DIM_M       1024
#define FPGA_BYTES_PER_BEAT  32   /* 256-bit = 32 bytes */
#define FPGA_ENCODING_MODE   0    /* 0 = 2-bit ternary */

/* --- Global state --- */
static int          fpga_devmem_fd = -1;
static volatile uint32_t *fpga_lw_bridge = NULL;
static volatile uint32_t *fpga_bitnet    = NULL;
static volatile uint32_t *fpga_ddr3      = NULL;
static uint32_t     fpga_lw_phys_base    = LW_BRIDGE_BASE;
static uint32_t     fpga_lw_span         = LW_BRIDGE_SPAN;
static uint32_t     fpga_bitnet_offset   = BITNET_OFFSET;
static uint32_t     fpga_ddr3_phys_base  = 0;
static uint32_t     fpga_ddr3_span       = 0;

/* Discovered at init from hardware capability registers */
static uint32_t     fpga_hw_num_pes = 0;
static uint32_t     fpga_hw_weights_per_beat = 0;
static uint32_t     fpga_hw_encoding_mode = 0;
static int          fpga_weight_addr_mode_abs = 0;
static int          fpga_status_debug = 0;
static int          fpga_wait_timeout_us = 1000000;
static int          fpga_strict_caps = 0;

/* Forward declaration for debug helper. */
static inline uint32_t fpga_reg_read(uint32_t offset);

static int fpga_parse_u32_env(const char *name, uint32_t *out)
{
	const char *s = getenv(name);
	char *end = NULL;
	unsigned long long v;

	if (!s || !*s)
		return 0;

	errno = 0;
	v = strtoull(s, &end, 0);
	if (errno == ERANGE || end == s || (end && *end != '\0') || v > 0xFFFFFFFFULL) {
		fprintf(stderr, "fpga_init: invalid %s='%s'\n", name, s);
		return -1;
	}
	*out = (uint32_t)v;
	return 1;
}

static int fpga_parse_bool_env(const char *name, int fallback)
{
	const char *s = getenv(name);
	if (!s || !*s)
		return fallback;
	if (!strcmp(s, "1") || !strcasecmp(s, "true") || !strcasecmp(s, "yes"))
		return 1;
	if (!strcmp(s, "0") || !strcasecmp(s, "false") || !strcasecmp(s, "no"))
		return 0;
	fprintf(stderr, "fpga_init: invalid %s='%s' (use 0/1)\n", name, s);
	return fallback;
}

static void fpga_log_status_sample(const char *tag)
{
	uint32_t st = fpga_reg_read(REG_STATUS);
	uint32_t perf = fpga_reg_read(REG_PERF_CYCLES);
	fprintf(stderr,
		"[FPGA] %s: STATUS=0x%08X (BUSY=%u DONE=%u) PERF=%u\n",
		tag, st, st & 0x1, (st >> 1) & 0x1, perf);
}

/* --- Low-level register access --- */

static inline void fpga_reg_write(uint32_t offset, uint32_t val)
{
	fpga_bitnet[offset / 4] = val;
}

static inline uint32_t fpga_reg_read(uint32_t offset)
{
	return fpga_bitnet[offset / 4];
}

/* --- Capability probe --- */

static int fpga_probe_capabilities(int verbose)
{
	fpga_hw_num_pes = fpga_reg_read(REG_NUM_PES);
	fpga_hw_weights_per_beat = fpga_reg_read(REG_WEIGHTS_PER_BEAT);
	fpga_hw_encoding_mode = fpga_reg_read(REG_ENCODING_MODE);

	/* Legacy 128-PE revisions may not implement diagnostic capability regs.
	 * Treat all-zero readback as "unknown capability", but continue. */
	if (fpga_hw_num_pes == 0 &&
	    fpga_hw_weights_per_beat == 0 &&
	    fpga_hw_encoding_mode == 0) {
		if (verbose) {
			fprintf(stderr,
				"[FPGA] capability regs unavailable at LW base 0x%08X + 0x%X; using legacy defaults (NUM_PES=%d ENCODING_MODE=%d)\n",
				fpga_lw_phys_base, fpga_bitnet_offset,
				FPGA_NUM_PES, FPGA_ENCODING_MODE);
		}
		return 0;
	}

	if (fpga_hw_num_pes != FPGA_NUM_PES ||
	    fpga_hw_weights_per_beat != FPGA_NUM_PES ||
	    fpga_hw_encoding_mode != FPGA_ENCODING_MODE) {
		if (verbose) {
			fprintf(stderr,
				"fpga_init: hardware contract mismatch at LW base 0x%08X + 0x%X (NUM_PES=%u, WEIGHTS_PER_BEAT=%u, ENCODING_MODE=%u), expected (%d, %d, %d)\n",
				fpga_lw_phys_base,
				fpga_bitnet_offset,
				fpga_hw_num_pes,
				fpga_hw_weights_per_beat,
				fpga_hw_encoding_mode,
				FPGA_NUM_PES,
				FPGA_NUM_PES,
				FPGA_ENCODING_MODE);
		}
		/* Legacy revisions can expose unrelated values at diagnostic offsets.
		 * Keep strict mode opt-in so known-good older bitstreams still run. */
		if (fpga_strict_caps)
			return -1;
		if (verbose) {
			fprintf(stderr,
				"[FPGA] capability mismatch ignored (BM_FPGA_STRICT_CAPS=0); continuing with legacy register map assumptions.\n");
		}
		return 0;
	}

	return 0;
}

/* --- Init / Cleanup --- */
static void fpga_cleanup(void);

/*
 * fpga_init: map lightweight bridge and DDR3 weight region.
 * ddr3_base: physical address of FPGA weight region (e.g., 0x30000000)
 * ddr3_span: size of weight region in bytes
 * Returns 0 on success, -1 on failure.
 */
static int fpga_init(uint32_t ddr3_base, uint32_t ddr3_span)
{
	uint32_t lw_base = LW_BRIDGE_BASE;
	uint32_t lw_span = LW_BRIDGE_SPAN;
	uint32_t forced_offset = BITNET_OFFSET;
	int has_forced_offset = 0;
	int env_rc;

	env_rc = fpga_parse_u32_env("BM_FPGA_LW_BASE", &lw_base);
	if (env_rc < 0)
		return -1;
	env_rc = fpga_parse_u32_env("BM_FPGA_LW_SPAN", &lw_span);
	if (env_rc < 0)
		return -1;
	env_rc = fpga_parse_u32_env("BM_FPGA_BITNET_OFFSET", &forced_offset);
	if (env_rc < 0)
		return -1;
	if (env_rc > 0)
		has_forced_offset = 1;

	fpga_devmem_fd = open("/dev/mem", O_RDWR | O_SYNC);
	if (fpga_devmem_fd < 0) {
		perror("fpga_init: open /dev/mem");
		return -1;
	}

	fpga_lw_bridge = (volatile uint32_t *)mmap(NULL, lw_span,
		PROT_READ | PROT_WRITE, MAP_SHARED, fpga_devmem_fd, lw_base);
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

	fpga_lw_phys_base = lw_base;
	fpga_lw_span = lw_span;
	fpga_ddr3_phys_base = ddr3_base;
	fpga_ddr3_span = ddr3_span;
	/* Legacy 128-PE ternary contract uses absolute DDR addresses in REG_WEIGHT_BASE.
	 * Keep env override for alternate address maps, but default to absolute. */
	fpga_weight_addr_mode_abs = fpga_parse_bool_env("BM_FPGA_WEIGHT_ADDR_ABS", 1);
	fpga_status_debug = fpga_parse_bool_env("BM_FPGA_DEBUG_STATUS", 0);
	fpga_strict_caps = fpga_parse_bool_env("BM_FPGA_STRICT_CAPS", 0);
	{
		uint32_t timeout_env = 0;
		int trc = fpga_parse_u32_env("BM_FPGA_TIMEOUT_US", &timeout_env);
		if (trc < 0) {
			fpga_cleanup();
			return -1;
		}
		if (trc > 0)
			fpga_wait_timeout_us = (int)timeout_env;
	}

	if (has_forced_offset) {
		if (forced_offset + REG_ENCODING_MODE + 4 > fpga_lw_span) {
			fprintf(stderr,
				"fpga_init: BM_FPGA_BITNET_OFFSET=0x%X is outside LW span 0x%X\n",
				forced_offset, fpga_lw_span);
			fpga_cleanup();
			return -1;
		}
		fpga_bitnet_offset = forced_offset;
		fpga_bitnet = (volatile uint32_t *)((uint8_t *)fpga_lw_bridge + fpga_bitnet_offset);
		if (fpga_probe_capabilities(1) < 0) {
			fpga_cleanup();
			return -1;
		}
	} else {
		/* Keep legacy behavior by default: fixed slave offset at BITNET_OFFSET.
		 * If your Qsys map changed, set BM_FPGA_BITNET_OFFSET explicitly. */
		fpga_bitnet_offset = BITNET_OFFSET;
		if (fpga_bitnet_offset + REG_ENCODING_MODE + 4 > fpga_lw_span) {
			fprintf(stderr,
				"fpga_init: bitnet offset 0x%X is outside LW span 0x%X\n",
				fpga_bitnet_offset, fpga_lw_span);
			fpga_cleanup();
			return -1;
		}
		fpga_bitnet = (volatile uint32_t *)((uint8_t *)fpga_lw_bridge + fpga_bitnet_offset);
	}

	fprintf(stderr,
		"[FPGA] MMIO bound: LW 0x%08X span 0x%X, bitnet offset 0x%X\n",
		fpga_lw_phys_base, fpga_lw_span, fpga_bitnet_offset);

	fprintf(stderr,
		"[FPGA] Weight address mode: %s (BM_FPGA_WEIGHT_ADDR_ABS=%d)\n",
		fpga_weight_addr_mode_abs ? "absolute" : "offset-from-ddr-base",
		fpga_weight_addr_mode_abs);
	fprintf(stderr, "[FPGA] Capability strict mode: %d (BM_FPGA_STRICT_CAPS)\n",
		fpga_strict_caps);
	fprintf(stderr, "[FPGA] Wait timeout: %d us (BM_FPGA_TIMEOUT_US)\n",
		fpga_wait_timeout_us);

	if (fpga_probe_capabilities(1) < 0) {
		fpga_cleanup();
		return -1;
	}

	return 0;
}

static void fpga_cleanup(void)
{
	if (fpga_ddr3 && fpga_ddr3 != MAP_FAILED)
		munmap((void *)fpga_ddr3, fpga_ddr3_span);
	if (fpga_lw_bridge && fpga_lw_bridge != MAP_FAILED)
		munmap((void *)fpga_lw_bridge, fpga_lw_span);
	if (fpga_devmem_fd >= 0)
		close(fpga_devmem_fd);
	fpga_devmem_fd = -1;
	fpga_lw_bridge = NULL;
	fpga_bitnet = NULL;
	fpga_ddr3 = NULL;
	fpga_lw_phys_base = LW_BRIDGE_BASE;
	fpga_lw_span = LW_BRIDGE_SPAN;
	fpga_bitnet_offset = BITNET_OFFSET;
	fpga_ddr3_phys_base = 0;
	fpga_ddr3_span = 0;
	fpga_hw_num_pes = 0;
	fpga_hw_weights_per_beat = 0;
	fpga_hw_encoding_mode = 0;
	fpga_wait_timeout_us = 1000000;
	fpga_strict_caps = 0;
}

/* --- Wait for DONE --- */

static int fpga_wait_done(int timeout_us)
{
	int saw_busy = 0;
	while (timeout_us > 0) {
		uint32_t st = fpga_reg_read(REG_STATUS);
		if (st & 0x2)
			return 0;
		if (st & 0x1) {
			saw_busy = 1;
		} else if (saw_busy) {
			/* Legacy fallback: some RTL variants don't latch DONE but BUSY drops at completion. */
			return 0;
		}
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
 */
static void fpga_bitlinear(const int8_t *activations, int K,
                           uint32_t weight_base, int M,
                           int stride,
                           int32_t *results)
{
	int i;
	static int addr_mode_logged = 0;

	if (K <= 0 || K > FPGA_MAX_DIM_K) {
		fprintf(stderr, "fpga_bitlinear: invalid K=%d (max %d)\n", K, FPGA_MAX_DIM_K);
		if (M > 0)
			memset(results, 0, (size_t)M * sizeof(int32_t));
		return;
	}

	for (i = 0; i < K; i++)
		fpga_reg_write(REG_ACT_BASE + i * 4, (uint32_t)(uint8_t)activations[i]);

	fpga_reg_write(REG_DIM_K, (uint32_t)K);
	fpga_reg_write(REG_SHIFT_AMT, 0);

	int rows_done = 0;
	while (rows_done < M) {
		int tile_m = M - rows_done;
		if (tile_m > FPGA_MAX_DIM_M)
			tile_m = FPGA_MAX_DIM_M;

		uint32_t tile_weight_base = weight_base + (uint32_t)rows_done * (uint32_t)stride;
		uint32_t tile_weight_hw_addr = tile_weight_base;
		if (!fpga_weight_addr_mode_abs) {
			if (tile_weight_base < fpga_ddr3_phys_base) {
				fprintf(stderr,
					"fpga_bitlinear: weight base underflow (tile=0x%08X ddr_base=0x%08X)\n",
					tile_weight_base, fpga_ddr3_phys_base);
				memset(&results[rows_done], 0, (size_t)tile_m * sizeof(int32_t));
				rows_done += tile_m;
				continue;
			}
			tile_weight_hw_addr = tile_weight_base - fpga_ddr3_phys_base;
		}
		if (!addr_mode_logged) {
			fprintf(stderr,
				"[FPGA] Weight addr sample: user=0x%08X hw=0x%08X ddr_base=0x%08X\n",
				tile_weight_base, tile_weight_hw_addr, fpga_ddr3_phys_base);
			addr_mode_logged = 1;
		}

		fpga_reg_write(REG_WEIGHT_BASE, tile_weight_hw_addr);
		fpga_reg_write(REG_DIM_M, (uint32_t)tile_m);
		/* Match the previously working bitmamba.cpp driver behavior. */
		fpga_reg_write(REG_CTRL, 0x1);
		if (fpga_status_debug && rows_done == 0)
			fpga_log_status_sample("after START");

		if (fpga_wait_done(fpga_wait_timeout_us) < 0) {
			fprintf(stderr, "fpga_bitlinear: timeout at M-tile offset %d\n", rows_done);
			fpga_log_status_sample("timeout");
			fprintf(stderr,
				"[FPGA] reg readback: WEIGHT_BASE=0x%08X DIM_M=%u DIM_K=%u\n",
				fpga_reg_read(REG_WEIGHT_BASE),
				fpga_reg_read(REG_DIM_M),
				fpga_reg_read(REG_DIM_K));
			fprintf(stderr,
				"[FPGA] timeout context: DIM_M=%d DIM_K=%d WEIGHT_BASE=0x%08X\n",
				tile_m, K, tile_weight_hw_addr);
			memset(&results[rows_done], 0, (size_t)tile_m * sizeof(int32_t));
			rows_done += tile_m;
			continue;
		}

		for (i = 0; i < tile_m; i++)
			results[rows_done + i] = (int32_t)fpga_reg_read(REG_RES_BASE + i * 4);

		rows_done += tile_m;
	}
}

/* --- ARM-side quantization helpers --- */

static float rms_norm_int8(const float *x, const float *norm_weight,
                           int size, int8_t *out)
{
	int i;

	float sum_sq = 0.0f;
	for (i = 0; i < size; i++)
		sum_sq += x[i] * x[i];
	float rms = 1.0f / sqrtf(sum_sq / size + 1e-6f);

	float max_abs = 0.0f;
	float *normalized = (float *)malloc((size_t)size * sizeof(float));
	for (i = 0; i < size; i++) {
		normalized[i] = x[i] * rms * norm_weight[i];
		float a = fabsf(normalized[i]);
		if (a > max_abs) max_abs = a;
	}

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

static void dequantize_results(const int32_t *fpga_out, int size,
                               float scale_x, float weight_scale,
                               float *out)
{
	float inv_scale = 1.0f / (scale_x * weight_scale);
	int i;
	for (i = 0; i < size; i++)
		out[i] = (float)fpga_out[i] * inv_scale;
}

static void bitlinear_forward_fpga(const float *x, int K, int M,
                                   const float *norm_weight,
                                   uint32_t weight_base,
                                   float weight_scale,
                                   int stride,
                                   float *out)
{
	int8_t *x_quant = (int8_t *)malloc((size_t)K);
	float scale_x = rms_norm_int8(x, norm_weight, K, x_quant);

	int32_t *raw_results = (int32_t *)malloc((size_t)M * sizeof(int32_t));
	fpga_bitlinear(x_quant, K, weight_base, M, stride, raw_results);

	dequantize_results(raw_results, M, scale_x, weight_scale, out);

	free(x_quant);
	free(raw_results);
}

#endif /* BITNET_FPGA_H */
