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
#include <sched.h>
#include <time.h>

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
#define REG_ACT_DDR3_BASE    0x28
#define REG_RES_DDR3_BASE    0x2C
#define REG_ACT_BASE         0x80
#define REG_RES_BASE         0x8000

#define CTRL_START           0x01
#define CTRL_DDR3_MODE       0x02

/* --- Expected hardware contract (DE10-Nano 128-PE ternary target) --- */
#define FPGA_NUM_PES         128
#define FPGA_MAX_DIM_K       4096
#define FPGA_MAX_DIM_M       1024
#define FPGA_BYTES_PER_BEAT  32   /* tile size: 128 PEs * 2-bit = 256 bits = 32 bytes */
#define FPGA_ENCODING_MODE   0    /* 0 = 2-bit ternary */

#if defined(__arm__) || defined(__aarch64__)
#define FPGA_CPU_RELAX() __asm__ volatile("yield")
#elif defined(__x86_64__) || defined(__i386__)
#define FPGA_CPU_RELAX() __asm__ volatile("pause")
#else
#define FPGA_CPU_RELAX() ((void)0)
#endif

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

/* DDR3 buffer offsets for activation/result transfer (set in fpga_init) */
static uint32_t     fpga_act_ddr3_offset = 0;
static uint32_t     fpga_res_ddr3_offset = 0;

/* Discovered at init from hardware capability registers */
static uint32_t     fpga_hw_num_pes = 0;
static uint32_t     fpga_hw_weights_per_beat = 0;
static uint32_t     fpga_hw_encoding_mode = 0;
static int          fpga_weight_addr_mode_abs = 0;
static int          fpga_status_debug = 0;
static int          fpga_wait_timeout_us = 1000000;
static int          fpga_strict_caps = 0;
static int8_t      *fpga_x_quant_buf = NULL;
static size_t       fpga_x_quant_cap = 0;
static int32_t     *fpga_raw_results_buf = NULL;
static size_t       fpga_raw_results_cap = 0;

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

	/* Allocate DDR3 buffers for activation/result transfer after weight region.
	 * Activations: up to FPGA_MAX_DIM_K bytes (4096).
	 * Results: up to FPGA_MAX_DIM_M * 4 bytes (4096).
	 * Both 4KB-aligned for burst efficiency. */
	{
		uint32_t weight_end = ddr3_span;  /* conservative: assume weights fill the span */
		fpga_act_ddr3_offset = (weight_end - 16384) & ~4095U;  /* 4KB aligned, near end */
		fpga_res_ddr3_offset = fpga_act_ddr3_offset + 8192;
		fprintf(stderr,
			"[FPGA] DDR3 act buffer: offset 0x%08X, res buffer: offset 0x%08X\n",
			fpga_act_ddr3_offset, fpga_res_ddr3_offset);
	}

	return 0;
}

static void fpga_cleanup(void)
{
	if (fpga_x_quant_buf)
		free(fpga_x_quant_buf);
	if (fpga_raw_results_buf)
		free(fpga_raw_results_buf);
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
	fpga_act_ddr3_offset = 0;
	fpga_res_ddr3_offset = 0;
	fpga_x_quant_buf = NULL;
	fpga_x_quant_cap = 0;
	fpga_raw_results_buf = NULL;
	fpga_raw_results_cap = 0;
}

/* --- Wait for DONE --- */

static int fpga_wait_done(int timeout_us)
{
	struct timespec start, now;
	int saw_busy = 0;
	unsigned int spins = 0;

	clock_gettime(CLOCK_MONOTONIC, &start);
	for (;;) {
		uint32_t st = fpga_reg_read(REG_STATUS);
		if (st & 0x2)
			return 0;
		if (st & 0x1) {
			saw_busy = 1;
		} else if (saw_busy) {
			/* Legacy fallback: some RTL variants don't latch DONE but BUSY drops at completion. */
			return 0;
		}

		spins++;
		if ((spins & 0xFF) == 0) {
			clock_gettime(CLOCK_MONOTONIC, &now);
			long elapsed_us =
				(long)(now.tv_sec - start.tv_sec) * 1000000L +
				(long)(now.tv_nsec - start.tv_nsec) / 1000L;
			if (elapsed_us >= timeout_us)
				return -1;

			/* Stay in a tight MMIO poll for a short window, then yield. */
			if (spins >= 4096)
				sched_yield();
		} else {
			FPGA_CPU_RELAX();
		}
	}
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
	static int addr_mode_logged = 0;

	if (K <= 0 || K > FPGA_MAX_DIM_K) {
		fprintf(stderr, "fpga_bitlinear: invalid K=%d (max %d)\n", K, FPGA_MAX_DIM_K);
		if (M > 0)
			memset(results, 0, (size_t)M * sizeof(int32_t));
		return;
	}

	/* DDR3 mode: write activations to DDR3 via memcpy instead of LW bridge */
	uint32_t act_phys = fpga_ddr3_phys_base + fpga_act_ddr3_offset;
	uint32_t res_phys = fpga_ddr3_phys_base + fpga_res_ddr3_offset;
	memcpy((void *)(fpga_ddr3 + fpga_act_ddr3_offset / 4), activations, (size_t)K);
	fpga_reg_write(REG_ACT_DDR3_BASE, act_phys);
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
		fpga_reg_write(REG_RES_DDR3_BASE, res_phys);
		/* START with DDR3_MODE flag */
		fpga_reg_write(REG_CTRL, CTRL_START | CTRL_DDR3_MODE);
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

		/* Read results from DDR3 via memcpy instead of LW bridge */
		memcpy(&results[rows_done],
		       (void *)(fpga_ddr3 + fpga_res_ddr3_offset / 4),
		       (size_t)tile_m * sizeof(int32_t));

		rows_done += tile_m;
	}
}

/* --- ARM-side quantization helpers --- */

static int fpga_reserve_scratch(size_t k_bytes, size_t m_elems)
{
	if (k_bytes > fpga_x_quant_cap) {
		int8_t *new_x = (int8_t *)realloc(fpga_x_quant_buf, k_bytes);
		if (!new_x)
			return -1;
		fpga_x_quant_buf = new_x;
		fpga_x_quant_cap = k_bytes;
	}

	if (m_elems > fpga_raw_results_cap) {
		int32_t *new_res = (int32_t *)realloc(
			fpga_raw_results_buf, m_elems * sizeof(int32_t));
		if (!new_res)
			return -1;
		fpga_raw_results_buf = new_res;
		fpga_raw_results_cap = m_elems;
	}

	return 0;
}

static inline int8_t fpga_round_clamped_i8(float val)
{
	if (val > 127.0f)
		val = 127.0f;
	if (val < -128.0f)
		val = -128.0f;
	return (int8_t)((val >= 0.0f) ? (val + 0.5f) : (val - 0.5f));
}

static float rms_norm_int8(const float *x, const float *norm_weight,
                           int size, int8_t *out)
{
	int i;

	float sum_sq = 0.0f;
	for (i = 0; i < size; i++)
		sum_sq += x[i] * x[i];
	float rms = 1.0f / sqrtf(sum_sq / size + 1e-6f);

	float max_abs = 0.0f;
	for (i = 0; i < size; i++) {
		float normalized = x[i] * rms * norm_weight[i];
		float a = fabsf(normalized);
		if (a > max_abs) max_abs = a;
	}

	float scale_x = 127.0f / (max_abs + 1e-5f);
	for (i = 0; i < size; i++) {
		float normalized = x[i] * rms * norm_weight[i];
		out[i] = fpga_round_clamped_i8(normalized * scale_x);
	}

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

/** Fused BitLinear forward: quant → FPGA matmul → dequant with M-tile pipelining.
 *
 * Overlaps dequantization of M-tile N with FPGA computation of M-tile N+1.
 * For in_proj (M=8224, 9 tiles), this hides ~8 dequant passes behind FPGA time.
 */
static void bitlinear_forward_fpga(const float *x, int K, int M,
                                   const float *norm_weight,
                                   uint32_t weight_base,
                                   float weight_scale,
                                   int stride,
                                   float *out)
{
	int8_t *x_quant;
	int32_t *raw_results;
	int use_scratch = fpga_reserve_scratch((size_t)K, (size_t)M) == 0;
	if (use_scratch) {
		x_quant = fpga_x_quant_buf;
		raw_results = fpga_raw_results_buf;
	} else {
		x_quant = (int8_t *)malloc((size_t)K);
		raw_results = (int32_t *)malloc((size_t)M * sizeof(int32_t));
		if (!x_quant || !raw_results) {
			free(x_quant);
			free(raw_results);
			memset(out, 0, (size_t)M * sizeof(float));
			return;
		}
	}

	/* ARM: quantize activations */
	float scale_x = rms_norm_int8(x, norm_weight, K, x_quant);
	float inv_scale = 1.0f / (scale_x * weight_scale);

	/* Write activations to DDR3 once (shared across all M-tiles) */
	uint32_t act_phys = fpga_ddr3_phys_base + fpga_act_ddr3_offset;
	uint32_t res_phys = fpga_ddr3_phys_base + fpga_res_ddr3_offset;
	memcpy((void *)(fpga_ddr3 + fpga_act_ddr3_offset / 4), x_quant, (size_t)K);
	fpga_reg_write(REG_ACT_DDR3_BASE, act_phys);
	fpga_reg_write(REG_DIM_K, (uint32_t)K);
	fpga_reg_write(REG_SHIFT_AMT, 0);

	static int addr_mode_logged = 0;
	int rows_done = 0;
	int prev_tile_m = 0;    /* previous tile's M (for pipelined dequant) */
	int prev_tile_start = 0;

	while (rows_done < M) {
		int tile_m = M - rows_done;
		if (tile_m > FPGA_MAX_DIM_M)
			tile_m = FPGA_MAX_DIM_M;

		uint32_t tile_weight_base = weight_base + (uint32_t)rows_done * (uint32_t)stride;
		uint32_t tile_weight_hw_addr = tile_weight_base;
		if (!fpga_weight_addr_mode_abs) {
			if (tile_weight_base < fpga_ddr3_phys_base) {
				memset(&out[rows_done], 0, (size_t)tile_m * sizeof(float));
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

		/* Start FPGA on this M-tile */
		fpga_reg_write(REG_WEIGHT_BASE, tile_weight_hw_addr);
		fpga_reg_write(REG_DIM_M, (uint32_t)tile_m);
		fpga_reg_write(REG_RES_DDR3_BASE, res_phys);
		fpga_reg_write(REG_CTRL, CTRL_START | CTRL_DDR3_MODE);

		/* Pipeline: dequantize PREVIOUS tile while FPGA computes THIS tile */
		if (prev_tile_m > 0) {
			int i;
			for (i = 0; i < prev_tile_m; i++)
				out[prev_tile_start + i] = (float)raw_results[prev_tile_start + i] * inv_scale;
		}

		/* Wait for current tile */
		if (fpga_wait_done(fpga_wait_timeout_us) < 0) {
			fprintf(stderr, "fpga_bitlinear: timeout at M-tile offset %d\n", rows_done);
			memset(&out[rows_done], 0, (size_t)tile_m * sizeof(float));
			rows_done += tile_m;
			continue;
		}

		/* Read results from DDR3 */
		memcpy(&raw_results[rows_done],
		       (void *)(fpga_ddr3 + fpga_res_ddr3_offset / 4),
		       (size_t)tile_m * sizeof(int32_t));

		prev_tile_start = rows_done;
		prev_tile_m = tile_m;
		rows_done += tile_m;
	}

	/* Dequantize last tile (no next FPGA tile to overlap with) */
	if (prev_tile_m > 0) {
		int i;
		for (i = 0; i < prev_tile_m; i++)
			out[prev_tile_start + i] = (float)raw_results[prev_tile_start + i] * inv_scale;
	}

	if (!use_scratch) {
		free(x_quant);
		free(raw_results);
	}
}

#endif /* BITNET_FPGA_H */
