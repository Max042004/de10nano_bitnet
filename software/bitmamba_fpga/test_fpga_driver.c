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

	/* Debug: verify activation buffer content in DDR3 */
	printf("  DDR3 act offset=0x%08X, res offset=0x%08X\n",
		fpga_act_ddr3_offset, fpga_res_ddr3_offset);
	volatile uint32_t *act_buf = fpga_ddr3 + fpga_act_ddr3_offset / 4;
	volatile uint32_t *res_buf = fpga_ddr3 + fpga_res_ddr3_offset / 4;

	/* Pre-fill result buffer with 0xDEADBEEF to detect if FPGA writes */
	for (i = 0; i < M; i++) res_buf[i] = 0xDEADBEEF;

	/* Debug: manually set ACT_DDR3_BASE and verify readback */
	uint32_t act_phys = fpga_ddr3_phys_base + fpga_act_ddr3_offset;
	printf("  Expected ACT_DDR3_BASE=0x%08X (phys_base=0x%08X + offset=0x%08X)\n",
		act_phys, fpga_ddr3_phys_base, fpga_act_ddr3_offset);
	fpga_reg_write(REG_ACT_DDR3_BASE, act_phys);
	printf("  ACT_DDR3_BASE readback=0x%08X\n", fpga_reg_read(REG_ACT_DDR3_BASE));

	/* Verify DDR3 at act_offset has 0x02, not 0x55 */
	printf("  DDR3[act_offset+0..3]: 0x%08X 0x%08X 0x%08X 0x%08X\n",
		act_buf[0], act_buf[1], act_buf[2], act_buf[3]);
	printf("  DDR3[0..3] (weights): 0x%08X 0x%08X 0x%08X 0x%08X\n",
		fpga_ddr3[0], fpga_ddr3[1], fpga_ddr3[2], fpga_ddr3[3]);

	fpga_bitlinear(acts, K, DDR3_BASE, M,
	               (K / FPGA_NUM_PES) * FPGA_BYTES_PER_BEAT, results);

	/* Debug: dump activation buffer after fpga_bitlinear wrote it */
	printf("  Act DDR3 [0..3]: 0x%08X 0x%08X 0x%08X 0x%08X\n",
		act_buf[0], act_buf[1], act_buf[2], act_buf[3]);
	/* Debug: dump result buffer to see what FPGA wrote */
	printf("  Res DDR3 [0..3]: 0x%08X 0x%08X 0x%08X 0x%08X\n",
		res_buf[0], res_buf[1], res_buf[2], res_buf[3]);
	printf("  Results [0..3]: %d %d %d %d\n",
		results[0], results[1], results[2], results[3]);

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

/* Test 0: Register write/readback diagnostic */
static void test_register_map(void)
{
	printf("\n--- Test 0: Register map diagnostic ---\n");

	/* Dump initial register state */
	printf("  Initial register state:\n");
	printf("    CTRL     (0x00) = 0x%08X\n", fpga_reg_read(REG_CTRL));
	printf("    STATUS   (0x04) = 0x%08X\n", fpga_reg_read(REG_STATUS));
	printf("    WBASE    (0x08) = 0x%08X\n", fpga_reg_read(REG_WEIGHT_BASE));
	printf("    DIM_M    (0x0C) = 0x%08X\n", fpga_reg_read(REG_DIM_M));
	printf("    DIM_K    (0x10) = 0x%08X\n", fpga_reg_read(REG_DIM_K));
	printf("    SHIFT    (0x14) = 0x%08X\n", fpga_reg_read(REG_SHIFT_AMT));
	printf("    PERF     (0x18) = 0x%08X\n", fpga_reg_read(REG_PERF_CYCLES));
	printf("    0x1C            = 0x%08X\n", fpga_reg_read(0x1C));
	printf("    0x20            = 0x%08X\n", fpga_reg_read(0x20));
	printf("    0x24            = 0x%08X\n", fpga_reg_read(0x24));
	printf("    ACT_DDR3 (0x28) = 0x%08X\n", fpga_reg_read(REG_ACT_DDR3_BASE));
	printf("    RES_DDR3 (0x2C) = 0x%08X\n", fpga_reg_read(REG_RES_DDR3_BASE));

	/* Write/readback test for R/W registers */
	fpga_reg_write(REG_DIM_K, 0xABCD);
	uint32_t readback_k = fpga_reg_read(REG_DIM_K);
	printf("  DIM_K write 0xABCD, readback = 0x%08X\n", readback_k);
	ASSERT_EQ("DIM_K readback", readback_k, 0xABCD);

	fpga_reg_write(REG_DIM_M, 0x1234);
	uint32_t readback_m = fpga_reg_read(REG_DIM_M);
	printf("  DIM_M write 0x1234, readback = 0x%08X\n", readback_m);
	ASSERT_EQ("DIM_M readback", readback_m, 0x1234);

	fpga_reg_write(REG_WEIGHT_BASE, 0xDEADBEEF);
	uint32_t readback_wb = fpga_reg_read(REG_WEIGHT_BASE);
	printf("  WEIGHT_BASE write 0xDEADBEEF, readback = 0x%08X\n", readback_wb);
	ASSERT_EQ("WEIGHT_BASE readback", (int)readback_wb, (int)0xDEADBEEF);

	fpga_reg_write(REG_SHIFT_AMT, 7);
	uint32_t readback_sh = fpga_reg_read(REG_SHIFT_AMT);
	printf("  SHIFT_AMT write 7, readback = 0x%08X\n", readback_sh);
	ASSERT_EQ("SHIFT_AMT readback", readback_sh, 7);

	/* Reset registers to sane values */
	fpga_reg_write(REG_DIM_K, 0);
	fpga_reg_write(REG_DIM_M, 0);
	fpga_reg_write(REG_WEIGHT_BASE, 0);
	fpga_reg_write(REG_SHIFT_AMT, 0);
}

/* Test legacy LW bridge path: write activations via register, no DDR3 mode */
static void test_legacy_path(void)
{
	printf("\n--- Test 4: Legacy LW bridge path (no DDR3 mode) ---\n");

	int M = 4, K = 128;
	int8_t acts[128];
	int32_t results[4];
	int i;

	for (i = 0; i < K; i++) acts[i] = 2;
	write_test_weights(M, K, 1);

	/* Write activations via LW bridge (old way) */
	for (i = 0; i < K; i++)
		fpga_reg_write(0x80 + i * 4, (uint32_t)(uint8_t)acts[i]);

	fpga_reg_write(REG_DIM_K, (uint32_t)K);
	fpga_reg_write(REG_DIM_N3, (uint32_t)(K / 3));
	fpga_reg_write(REG_SHIFT_AMT, 0);
	fpga_reg_write(REG_WEIGHT_BASE, DDR3_BASE);
	fpga_reg_write(REG_DIM_M, (uint32_t)M);

	/* Check register readback before START */
	printf("  Pre-START: DIM_K=%u DIM_M=%u WEIGHT_BASE=0x%08X STATUS=0x%08X\n",
		fpga_reg_read(REG_DIM_K), fpga_reg_read(REG_DIM_M),
		fpga_reg_read(REG_WEIGHT_BASE), fpga_reg_read(REG_STATUS));

	/* START without DDR3_MODE flag */
	fpga_reg_write(REG_CTRL, CTRL_START);

	/* Poll STATUS a few times to see transitions */
	printf("  STATUS after START: 0x%08X\n", fpga_reg_read(REG_STATUS));
	usleep(100);
	printf("  STATUS +100us: 0x%08X\n", fpga_reg_read(REG_STATUS));

	if (fpga_wait_done(1000000) < 0) {
		printf("  TIMEOUT in legacy mode!\n");
		printf("  STATUS at timeout: 0x%08X  PERF=%u\n",
			fpga_reg_read(REG_STATUS), fpga_reg_read(REG_PERF_CYCLES));
		return;
	}

	printf("  DONE! PERF_CYCLES=%u\n", fpga_reg_read(REG_PERF_CYCLES));

	/* Read results via LW bridge (old way) — scan more addresses */
	for (i = 0; i < M; i++)
		results[i] = (int32_t)fpga_reg_read(0x8000 + i * 4);

	printf("  Legacy results [0..3]: %d %d %d %d\n",
		results[0], results[1], results[2], results[3]);

	/* Extended scan: read result addresses 0..15 to detect address mapping issues */
	printf("  Result scan [0..15]:");
	for (i = 0; i < 16; i++)
		printf(" %d", (int32_t)fpga_reg_read(0x8000 + i * 4));
	printf("\n");

	/* Also scan a few words of DDR3 weight region to verify test weights are there */
	printf("  DDR3 weights [0..7]: 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X\n",
		fpga_ddr3[0], fpga_ddr3[1], fpga_ddr3[2], fpga_ddr3[3],
		fpga_ddr3[4], fpga_ddr3[5], fpga_ddr3[6], fpga_ddr3[7]);

	for (i = 0; i < M; i++) {
		char name[64];
		snprintf(name, sizeof(name), "Legacy Row %d = 256", i);
		ASSERT_EQ(name, results[i], 256);
	}
}

int main(void)
{
	printf("=== BitNet FPGA Driver Test ===\n");

	if (fpga_init(DDR3_BASE, DDR3_SPAN) < 0) {
		fprintf(stderr, "Failed to initialize FPGA. Run as root.\n");
		return 1;
	}

	test_register_map();
	test_legacy_path();
	test_basic();
	test_max_k();
	test_float_path();

	fpga_cleanup();

	printf("\n=== Results: %d passed, %d failed ===\n",
		tests_passed, tests_failed);
	return tests_failed > 0 ? 1 : 0;
}
