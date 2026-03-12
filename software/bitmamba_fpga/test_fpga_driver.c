/*
 * test_fpga_driver.c -- Smoke test for the FPGA BitLinear driver.
 *
 * Tests:
 *   1. fpga_init / fpga_cleanup
 *   2. fpga_bitlinear with known packed weights
 *   3. M-tiling behavior
 *   4. bitlinear_forward_fpga end-to-end float->float path
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bitnet_fpga.h"

#define DDR3_BASE 0x30000000
#define DDR3_SPAN 0x00100000  /* 1 MB for test */

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_EQ(name, actual, expected) do { \
	if ((actual) == (expected)) { \
		printf("  PASS: %s\n", (name)); \
		tests_passed++; \
	} else { \
		printf("  FAIL: %s -- got %d, expected %d\n", (name), (int)(actual), (int)(expected)); \
		tests_failed++; \
	} \
} while (0)

/* Pack 128 ternary weights into one 256-bit beat (8 x uint32_t). */
static void pack_weights_128(const int8_t weights[128], uint32_t out[8])
{
	int i;
	memset(out, 0, 32);
	for (i = 0; i < 128; i++) {
		uint32_t enc;
		if (weights[i] == 1)       enc = 0x1;
		else if (weights[i] == -1) enc = 0x2;
		else                       enc = 0x0;
		out[i / 16] |= enc << ((i % 16) * 2);
	}
}

static void write_test_weights(int M, int K, int8_t fill_weight)
{
	int tiles_per_row = (K + FPGA_NUM_PES - 1) / FPGA_NUM_PES;
	int row, tile, i;

	for (row = 0; row < M; row++) {
		for (tile = 0; tile < tiles_per_row; tile++) {
			int8_t chunk[128];
			uint32_t packed[8];

			for (i = 0; i < 128; i++) {
				int col = tile * FPGA_NUM_PES + i;
				chunk[i] = (col < K) ? fill_weight : 0;
			}

			pack_weights_128(chunk, packed);

			uint32_t word_offset = (row * tiles_per_row + tile) * 8;
			for (i = 0; i < 8; i++)
				fpga_ddr3[word_offset + i] = packed[i];
		}
	}
}

static void test_basic(void)
{
	printf("\n--- Test 1: Basic M=4, K=128, all +1 ---\n");

	int M = 4, K = 128;
	int8_t acts[128];
	int32_t results[4];
	int i;

	for (i = 0; i < K; i++) acts[i] = 2;
	write_test_weights(M, K, 1);

	fpga_bitlinear(acts, K, DDR3_BASE, M,
	               (K / FPGA_NUM_PES) * FPGA_BYTES_PER_BEAT, results);

	for (i = 0; i < M; i++) {
		char name[64];
		snprintf(name, sizeof(name), "Row %d = 256", i);
		ASSERT_EQ(name, results[i], 256);
	}
}

static void test_max_k(void)
{
	printf("\n--- Test 2: K=2048, M=1, all +1, act=1 ---\n");

	int M = 1, K = 2048;
	int8_t *acts = (int8_t *)malloc((size_t)K);
	int32_t results[1];
	int i;

	for (i = 0; i < K; i++) acts[i] = 1;
	write_test_weights(M, K, 1);

	fpga_bitlinear(acts, K, DDR3_BASE, M,
	               (K / FPGA_NUM_PES) * FPGA_BYTES_PER_BEAT, results);

	ASSERT_EQ("K=2048 raw accumulator", results[0], 2048);
	free(acts);
}

static void test_float_path(void)
{
	printf("\n--- Test 3: bitlinear_forward_fpga float path ---\n");

	int K = 128, M = 4;
	float x[128];
	float norm_w[128];
	float out[4];
	int i;

	for (i = 0; i < K; i++) {
		x[i] = 1.0f;
		norm_w[i] = 1.0f;
	}

	write_test_weights(M, K, 1);

	bitlinear_forward_fpga(x, K, M, norm_w, DDR3_BASE,
	                       1.0f,
	                       (K / FPGA_NUM_PES) * FPGA_BYTES_PER_BEAT,
	                       out);

	printf("  Float output: [%.4f, %.4f, %.4f, %.4f]\n",
		out[0], out[1], out[2], out[3]);

	int all_positive = 1;
	for (i = 0; i < M; i++) {
		if (out[i] <= 0.0f) all_positive = 0;
	}
	if (all_positive) {
		printf("  PASS: All outputs positive\n");
		tests_passed++;
	} else {
		printf("  FAIL: Expected all positive outputs\n");
		tests_failed++;
	}
}

int main(void)
{
	printf("=== BitNet FPGA Driver Test ===\n");

	if (fpga_init(DDR3_BASE, DDR3_SPAN) < 0) {
		fprintf(stderr, "Failed to initialize FPGA. Run as root.\n");
		return 1;
	}

	test_basic();
	test_max_k();
	test_float_path();

	fpga_cleanup();

	printf("\n=== Results: %d passed, %d failed ===\n",
		tests_passed, tests_failed);
	return tests_failed > 0 ? 1 : 0;
}
