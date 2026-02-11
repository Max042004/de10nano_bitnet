/*
 * bitnet_test_common.h — Shared definitions and helpers for BitNet HPS tests
 *
 * Extracted from test_bitnet.c with additions for multi-row, multi-tile,
 * and reference-model computation.
 */

#ifndef BITNET_TEST_COMMON_H
#define BITNET_TEST_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

/* --- Memory map constants --- */
#define LW_BRIDGE_BASE  0xFF200000   /* HPS-to-FPGA lightweight bridge */
#define LW_BRIDGE_SPAN  0x00200000   /* 2 MB */

#define DDR3_BASE       0x30000000   /* DDR3 region for weights */
#define DDR3_SPAN       0x00100000   /* 1 MB for weight data (enough for 1024x1024) */

/* BitNet base offset within lightweight bridge (match Platform Designer) */
#define BITNET_OFFSET   0x0

/* --- BitNet accelerator register offsets (byte-addressed) --- */
#define REG_CTRL        0x00   /* W:  bit 0 = START (pulse) */
#define REG_STATUS      0x04   /* R:  bit 0 = BUSY, bit 1 = DONE */
#define REG_WEIGHT_BASE 0x08   /* RW: DDR3 byte address of weights */
#define REG_DIM_M       0x0C   /* RW: number of output rows */
#define REG_DIM_K       0x10   /* RW: input vector length */
#define REG_SHIFT_AMT   0x14   /* RW: requantization shift (0-31) */
#define REG_PERF_CYCLES 0x18   /* R:  cycle count of last run */
#define REG_ACT_BASE    0x80   /* W:  activation[i] at 0x80 + i*4 */
#define REG_RES_BASE    0x2000 /* R:  result[i]     at 0x2000 + i*4 */

/* Weight encoding: 2 bits per weight, 64 weights per 128-bit word */
/*   00 = 0    01 = +1    10 = -1    11 = reserved                 */

#define NUM_PES 64

/* --- Test framework macros --- */

static int g_tests_passed = 0;
static int g_tests_failed = 0;
static int g_tests_total  = 0;

#define TEST_PASS(name) do { \
	g_tests_passed++; g_tests_total++; \
	printf("  PASS: %s\n", (name)); \
} while (0)

#define TEST_FAIL(name, fmt, ...) do { \
	g_tests_failed++; g_tests_total++; \
	printf("  FAIL: %s — " fmt "\n", (name), ##__VA_ARGS__); \
} while (0)

#define ASSERT_EQ(name, actual, expected) do { \
	if ((actual) == (expected)) { \
		TEST_PASS(name); \
	} else { \
		TEST_FAIL(name, "got %d, expected %d", (int)(actual), (int)(expected)); \
	} \
} while (0)

#define ASSERT_NEQ(name, actual, not_expected) do { \
	if ((actual) != (not_expected)) { \
		TEST_PASS(name); \
	} else { \
		TEST_FAIL(name, "got %d, should not equal %d", (int)(actual), (int)(not_expected)); \
	} \
} while (0)

#define ASSERT_GT(name, actual, threshold) do { \
	if ((actual) > (threshold)) { \
		TEST_PASS(name); \
	} else { \
		TEST_FAIL(name, "got %d, expected > %d", (int)(actual), (int)(threshold)); \
	} \
} while (0)

/* --- Global state for memory-mapped pointers --- */

static int          g_devmem_fd = -1;
static volatile uint32_t *g_lw_bridge = NULL;
static volatile uint32_t *g_bitnet    = NULL;
static volatile uint32_t *g_ddr3      = NULL;

static int mmap_init(void)
{
	g_devmem_fd = open("/dev/mem", O_RDWR | O_SYNC);
	if (g_devmem_fd < 0) {
		perror("open /dev/mem");
		return -1;
	}

	g_lw_bridge = (volatile uint32_t *)mmap(NULL, LW_BRIDGE_SPAN,
		PROT_READ | PROT_WRITE, MAP_SHARED, g_devmem_fd, LW_BRIDGE_BASE);
	if (g_lw_bridge == MAP_FAILED) {
		perror("mmap lw_bridge");
		close(g_devmem_fd);
		return -1;
	}

	g_ddr3 = (volatile uint32_t *)mmap(NULL, DDR3_SPAN,
		PROT_READ | PROT_WRITE, MAP_SHARED, g_devmem_fd, DDR3_BASE);
	if (g_ddr3 == MAP_FAILED) {
		perror("mmap ddr3");
		munmap((void *)g_lw_bridge, LW_BRIDGE_SPAN);
		close(g_devmem_fd);
		return -1;
	}

	g_bitnet = (volatile uint32_t *)((uint8_t *)g_lw_bridge + BITNET_OFFSET);
	return 0;
}

static void mmap_cleanup(void)
{
	if (g_ddr3 && g_ddr3 != MAP_FAILED)
		munmap((void *)g_ddr3, DDR3_SPAN);
	if (g_lw_bridge && g_lw_bridge != MAP_FAILED)
		munmap((void *)g_lw_bridge, LW_BRIDGE_SPAN);
	if (g_devmem_fd >= 0)
		close(g_devmem_fd);
}

/* --- Register access helpers --- */

static inline void reg_write(volatile uint32_t *base, uint32_t offset, uint32_t val)
{
	base[offset / 4] = val;
}

static inline uint32_t reg_read(volatile uint32_t *base, uint32_t offset)
{
	return base[offset / 4];
}

/* Wait for accelerator DONE (STATUS bit 1), returns 0 on success, -1 on timeout */
static int wait_done(volatile uint32_t *base, int timeout_us)
{
	while (timeout_us > 0) {
		uint32_t st = reg_read(base, REG_STATUS);
		if (st & 0x2)
			return 0;
		usleep(10);
		timeout_us -= 10;
	}
	return -1;
}

/* --- Weight packing --- */

/* Pack 64 ternary weights into a 128-bit (4x uint32_t) DDR3 word.
 * weights[i] = -1, 0, or +1.  Encoding: 00=0, 01=+1, 10=-1 */
static void pack_weights(const int8_t weights[64], uint32_t out[4])
{
	int i;
	memset(out, 0, 16);
	for (i = 0; i < 64; i++) {
		uint32_t enc;
		if (weights[i] == 1)       enc = 0x1;
		else if (weights[i] == -1) enc = 0x2;
		else                       enc = 0x0;
		out[i / 16] |= enc << ((i % 16) * 2);
	}
}

/* Write an M x K weight matrix to DDR3 with proper tile layout.
 * wmat[row * K + col] is the weight at (row, col).
 * Memory layout: base + row * tilesPerRow * 16 + tile * 16
 * Each tile covers 64 consecutive columns. */
static void write_weight_matrix(const int8_t *wmat, int M, int K)
{
	int row, tile;
	int tiles_per_row = (K + NUM_PES - 1) / NUM_PES;

	for (row = 0; row < M; row++) {
		for (tile = 0; tile < tiles_per_row; tile++) {
			int8_t chunk[64];
			uint32_t packed[4];
			int col_start = tile * NUM_PES;
			int i;

			for (i = 0; i < 64; i++) {
				int col = col_start + i;
				if (col < K)
					chunk[i] = wmat[row * K + col];
				else
					chunk[i] = 0;  /* pad with zeros */
			}

			pack_weights(chunk, packed);

			/* Write 128-bit word to DDR3 */
			uint32_t word_offset = (row * tiles_per_row + tile) * 4; /* in uint32 units */
			for (i = 0; i < 4; i++)
				g_ddr3[word_offset + i] = packed[i];
		}
	}
}

/* Write K activations to the activation register space */
static void write_activations(const int8_t *acts, int K)
{
	int i;
	for (i = 0; i < K; i++)
		reg_write(g_bitnet, REG_ACT_BASE + i * 4, (uint32_t)(uint8_t)acts[i]);
}

/* Read M results from the result buffer */
static void read_results(int8_t *results, int M)
{
	int i;
	for (i = 0; i < M; i++)
		results[i] = (int8_t)(reg_read(g_bitnet, REG_RES_BASE + i * 4) & 0xFF);
}

/* --- Reference model --- */

/* Compute expected result for a single row: dot product → arithmetic right shift → clamp to [-128, +127] */
static int8_t compute_expected_row(const int8_t *weights, const int8_t *acts, int K, int shift)
{
	int32_t acc = 0;
	int i;
	for (i = 0; i < K; i++)
		acc += (int32_t)weights[i] * (int32_t)acts[i];

	/* Arithmetic right shift */
	acc = acc >> shift;

	/* Clamp to INT8 */
	if (acc > 127)  acc = 127;
	if (acc < -128) acc = -128;

	return (int8_t)acc;
}

/* Compute expected results for M rows */
static void compute_expected(const int8_t *wmat, const int8_t *acts,
                             int M, int K, int shift, int8_t *expected)
{
	int row;
	for (row = 0; row < M; row++)
		expected[row] = compute_expected_row(&wmat[row * K], acts, K, shift);
}

/* --- High-level test runner --- */

/* Configure, load, execute, and read results. Returns 0 on success, -1 on timeout. */
static int run_test(const int8_t *wmat, const int8_t *acts,
                    int M, int K, int shift, int8_t *results)
{
	/* Write weights to DDR3 */
	write_weight_matrix(wmat, M, K);

	/* Configure dimensions */
	reg_write(g_bitnet, REG_WEIGHT_BASE, DDR3_BASE);
	reg_write(g_bitnet, REG_DIM_M,       (uint32_t)M);
	reg_write(g_bitnet, REG_DIM_K,       (uint32_t)K);
	reg_write(g_bitnet, REG_SHIFT_AMT,   (uint32_t)shift);

	/* Write activations */
	write_activations(acts, K);

	/* Start computation */
	reg_write(g_bitnet, REG_CTRL, 0x1);

	/* Wait for completion */
	if (wait_done(g_bitnet, 500000) < 0)
		return -1;

	/* Read results */
	read_results(results, M);
	return 0;
}

#endif /* BITNET_TEST_COMMON_H */
