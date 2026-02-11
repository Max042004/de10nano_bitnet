/*
 * test_bitnet_comprehensive.c — Comprehensive HPS test suite for BitNet accelerator
 *
 * Covers: multi-row, multi-tile, boundary cases, negative values, shift amounts,
 * clamping, register readback, performance counters, and back-to-back computations.
 *
 * Cross-compile:
 *   arm-linux-gnueabihf-gcc -O2 -o test_bitnet_comprehensive test_bitnet_comprehensive.c
 *
 * Run (requires root for /dev/mem):
 *   sudo ./test_bitnet_comprehensive         # run all tests
 *   sudo ./test_bitnet_comprehensive A        # run only category A tests
 *   sudo ./test_bitnet_comprehensive A1       # run single test
 */

#include "bitnet_test_common.h"

/* ================================================================== */
/*  Category W: Weight type basics (M=1, K=64, single tile)           */
/* ================================================================== */

static void test_W1_all_plus_one(void)
{
	const int M = 1, K = 64, shift = 0;
	int8_t wmat[64], acts[64], results[1];
	int i;

	printf("  W1: All +1 weights, act=1 => 64\n");

	for (i = 0; i < 64; i++) { wmat[i] = 1; acts[i] = 1; }

	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("W1", "timeout"); return;
	}
	ASSERT_EQ("W1", results[0], 64);
}

static void test_W2_all_zero(void)
{
	const int M = 1, K = 64, shift = 0;
	int8_t wmat[64], acts[64], results[1];
	int i;

	printf("  W2: All zero weights, act=100 => 0\n");

	for (i = 0; i < 64; i++) { wmat[i] = 0; acts[i] = 100; }

	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("W2", "timeout"); return;
	}
	ASSERT_EQ("W2", results[0], 0);
}

static void test_W3_all_minus_one(void)
{
	const int M = 1, K = 64, shift = 1;
	int8_t wmat[64], acts[64], results[1];
	int i;

	printf("  W3: All -1 weights, act=2, shift=1 => -64\n");

	for (i = 0; i < 64; i++) { wmat[i] = -1; acts[i] = 2; }

	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("W3", "timeout"); return;
	}
	ASSERT_EQ("W3", results[0], -64);
}

static void test_W4_mixed_cancel(void)
{
	const int M = 1, K = 64, shift = 0;
	int8_t wmat[64], acts[64], results[1];
	int i;

	printf("  W4: Half +1, half -1 weights cancel => 0\n");

	for (i = 0; i < 32; i++) wmat[i] = 1;
	for (i = 32; i < 64; i++) wmat[i] = -1;
	for (i = 0; i < 64; i++) acts[i] = 1;

	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("W4", "timeout"); return;
	}
	ASSERT_EQ("W4", results[0], 0);
}

static void test_W5_positive_clamp(void)
{
	const int M = 1, K = 64, shift = 0;
	int8_t wmat[64], acts[64], results[1];
	int i;

	printf("  W5: All +1, act=4, 64*4=256 => clamp 127\n");

	for (i = 0; i < 64; i++) { wmat[i] = 1; acts[i] = 4; }

	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("W5", "timeout"); return;
	}
	ASSERT_EQ("W5", results[0], 127);
}

/* ================================================================== */
/*  Category A: Multi-row (M > 1, K = 64)                            */
/* ================================================================== */

static void test_A1_multi_row_2(void)
{
	/* M=2, K=64: two rows with different weight patterns */
	const int M = 2, K = 64, shift = 0;
	int8_t wmat[2 * 64], acts[64], results[2], expected[2];
	int i;

	printf("  A1: M=2, K=64, different rows\n");

	/* Row 0: all +1, Row 1: all -1 */
	for (i = 0; i < 64; i++) { wmat[i] = 1; wmat[64 + i] = -1; }
	for (i = 0; i < 64; i++) acts[i] = 1;

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("A1 row0", "timeout");
		TEST_FAIL("A1 row1", "timeout");
		return;
	}
	ASSERT_EQ("A1 row0", results[0], expected[0]);  /* 64 */
	ASSERT_EQ("A1 row1", results[1], expected[1]);  /* -64 */
}

static void test_A2_multi_row_4(void)
{
	const int M = 4, K = 64, shift = 0;
	int8_t wmat[4 * 64], acts[64], results[4], expected[4];
	int i, row;

	printf("  A2: M=4, K=64, varying patterns\n");

	/* Row 0: all +1, Row 1: all 0, Row 2: all -1, Row 3: alternating */
	for (i = 0; i < 64; i++) {
		wmat[0 * 64 + i] = 1;
		wmat[1 * 64 + i] = 0;
		wmat[2 * 64 + i] = -1;
		wmat[3 * 64 + i] = (i % 2 == 0) ? 1 : -1;
	}
	for (i = 0; i < 64; i++) acts[i] = 2;

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		for (row = 0; row < M; row++)
			TEST_FAIL("A2 row", "timeout");
		return;
	}
	for (row = 0; row < M; row++) {
		char name[32];
		snprintf(name, sizeof(name), "A2 row%d", row);
		ASSERT_EQ(name, results[row], expected[row]);
	}
}

static void test_A3_multi_row_8(void)
{
	const int M = 8, K = 64, shift = 2;
	int8_t wmat[8 * 64], acts[64], results[8], expected[8];
	int i, row;

	printf("  A3: M=8, K=64, shift=2\n");

	/* Each row has a different number of +1 weights */
	for (row = 0; row < M; row++)
		for (i = 0; i < 64; i++)
			wmat[row * 64 + i] = (i < (row + 1) * 8) ? 1 : 0;
	for (i = 0; i < 64; i++) acts[i] = 4;

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		for (row = 0; row < M; row++)
			TEST_FAIL("A3 row", "timeout");
		return;
	}
	for (row = 0; row < M; row++) {
		char name[32];
		snprintf(name, sizeof(name), "A3 row%d", row);
		ASSERT_EQ(name, results[row], expected[row]);
	}
}

/* ================================================================== */
/*  Category B: Multi-tile (K > 64, M = 1)                           */
/* ================================================================== */

static void test_B1_two_tiles(void)
{
	const int M = 1, K = 128, shift = 1;
	int8_t wmat[128], acts[128], results[1], expected[1];
	int i;

	printf("  B1: M=1, K=128 (2 tiles), shift=1\n");

	for (i = 0; i < 128; i++) wmat[i] = 1;
	for (i = 0; i < 128; i++) acts[i] = 1;

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("B1", "timeout"); return;
	}
	ASSERT_EQ("B1", results[0], expected[0]);  /* 128 >> 1 = 64 */
}

static void test_B2_three_tiles(void)
{
	const int M = 1, K = 192, shift = 2;
	int8_t wmat[192], acts[192], results[1], expected[1];
	int i;

	printf("  B2: M=1, K=192 (3 tiles), shift=2\n");

	for (i = 0; i < 192; i++) wmat[i] = 1;
	for (i = 0; i < 192; i++) acts[i] = 1;

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("B2", "timeout"); return;
	}
	ASSERT_EQ("B2", results[0], expected[0]);  /* 192 >> 2 = 48 */
}

static void test_B3_four_tiles(void)
{
	const int M = 1, K = 256, shift = 2;
	int8_t wmat[256], acts[256], results[1], expected[1];
	int i;

	printf("  B3: M=1, K=256 (4 tiles), shift=2\n");

	for (i = 0; i < 256; i++) wmat[i] = (i % 3 == 0) ? 1 : (i % 3 == 1) ? -1 : 0;
	for (i = 0; i < 256; i++) acts[i] = 3;

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("B3", "timeout"); return;
	}
	ASSERT_EQ("B3", results[0], expected[0]);
}

static void test_B4_tile_accumulation(void)
{
	/* Verify that tiles actually accumulate: tile0 positive, tile1 negative */
	const int M = 1, K = 128, shift = 0;
	int8_t wmat[128], acts[128], results[1], expected[1];
	int i;

	printf("  B4: M=1, K=128, tile0=+1, tile1=-1, should cancel\n");

	for (i = 0; i < 64; i++)  wmat[i] = 1;
	for (i = 64; i < 128; i++) wmat[i] = -1;
	for (i = 0; i < 128; i++) acts[i] = 1;

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("B4", "timeout"); return;
	}
	ASSERT_EQ("B4", results[0], expected[0]);  /* 64 - 64 = 0 */
}

static void test_B5_partial_last_tile(void)
{
	/* K=128 but only first 96 weights are +1, last 32 are 0 */
	const int M = 1, K = 128, shift = 1;
	int8_t wmat[128], acts[128], results[1], expected[1];
	int i;

	printf("  B5: M=1, K=128, only 96 active weights\n");

	for (i = 0; i < 96; i++)  wmat[i] = 1;
	for (i = 96; i < 128; i++) wmat[i] = 0;
	for (i = 0; i < 128; i++) acts[i] = 1;

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("B5", "timeout"); return;
	}
	ASSERT_EQ("B5", results[0], expected[0]);  /* 96 >> 1 = 48 */
}

/* ================================================================== */
/*  Category C: Multi-row + Multi-tile                                */
/* ================================================================== */

static void test_C1_2x128(void)
{
	const int M = 2, K = 128, shift = 1;
	int8_t wmat[2 * 128], acts[128], results[2], expected[2];
	int i, row;

	printf("  C1: M=2, K=128\n");

	/* Row 0: all +1, Row 1: first half +1, second half -1 */
	for (i = 0; i < 128; i++) { wmat[i] = 1; wmat[128 + i] = (i < 64) ? 1 : -1; }
	for (i = 0; i < 128; i++) acts[i] = 1;

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		for (row = 0; row < M; row++) TEST_FAIL("C1", "timeout");
		return;
	}
	ASSERT_EQ("C1 row0", results[0], expected[0]);  /* 128>>1 = 64 */
	ASSERT_EQ("C1 row1", results[1], expected[1]);  /* 0>>1 = 0 */
}

static void test_C2_4x192(void)
{
	const int M = 4, K = 192, shift = 2;
	int8_t *wmat, acts[192], results[4], expected[4];
	int i, row;

	printf("  C2: M=4, K=192\n");

	wmat = (int8_t *)calloc(M * K, sizeof(int8_t));
	if (!wmat) { TEST_FAIL("C2", "alloc failed"); return; }

	for (row = 0; row < M; row++)
		for (i = 0; i < K; i++)
			wmat[row * K + i] = ((i + row) % 3 == 0) ? 1 : ((i + row) % 3 == 1) ? -1 : 0;
	for (i = 0; i < K; i++) acts[i] = 2;

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		for (row = 0; row < M; row++) TEST_FAIL("C2", "timeout");
		free(wmat); return;
	}
	for (row = 0; row < M; row++) {
		char name[32];
		snprintf(name, sizeof(name), "C2 row%d", row);
		ASSERT_EQ(name, results[row], expected[row]);
	}
	free(wmat);
}

/* ================================================================== */
/*  Category D: Boundary dimensions                                   */
/* ================================================================== */

static void test_D1_min_dims(void)
{
	/* Minimum supported: M=1, K=64 */
	const int M = 1, K = 64, shift = 0;
	int8_t wmat[64], acts[64], results[1], expected[1];
	int i;

	printf("  D1: Minimum dimensions M=1, K=64\n");

	for (i = 0; i < 64; i++) { wmat[i] = 1; acts[i] = 1; }

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("D1", "timeout"); return;
	}
	ASSERT_EQ("D1", results[0], expected[0]);
}

static void test_D2_K_not_mult_64(void)
{
	/* K=96 — not a multiple of 64, requires 2 tiles with padding */
	const int M = 1, K = 96, shift = 0;
	int8_t wmat[96], acts[96], results[1], expected[1];
	int i;

	printf("  D2: K=96 (not multiple of 64)\n");

	for (i = 0; i < 96; i++) { wmat[i] = 1; acts[i] = 1; }

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("D2", "timeout"); return;
	}
	ASSERT_EQ("D2", results[0], expected[0]);  /* 96, clamped to 96 */
}

static void test_D3_M16(void)
{
	const int M = 16, K = 64, shift = 0;
	int8_t wmat[16 * 64], acts[64], results[16], expected[16];
	int i, row;

	printf("  D3: M=16, K=64\n");

	for (row = 0; row < M; row++)
		for (i = 0; i < 64; i++)
			wmat[row * 64 + i] = (i < row * 4) ? 1 : 0;
	for (i = 0; i < 64; i++) acts[i] = 1;

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		for (row = 0; row < M; row++) TEST_FAIL("D3", "timeout");
		return;
	}
	for (row = 0; row < M; row++) {
		char name[32];
		snprintf(name, sizeof(name), "D3 row%d", row);
		ASSERT_EQ(name, results[row], expected[row]);
	}
}

static void test_D4_large_K(void)
{
	/* K=512 — 8 tiles, tests deeper accumulation */
	const int M = 1, K = 512, shift = 3;
	int8_t *wmat, *acts;
	int8_t results[1], expected[1];
	int i;

	printf("  D4: M=1, K=512 (8 tiles), shift=3\n");

	wmat = (int8_t *)calloc(K, sizeof(int8_t));
	acts = (int8_t *)calloc(K, sizeof(int8_t));
	if (!wmat || !acts) { TEST_FAIL("D4", "alloc failed"); free(wmat); free(acts); return; }

	/* Alternating +1/-1 with a bias toward +1 */
	for (i = 0; i < K; i++) { wmat[i] = (i % 4 == 0) ? 1 : 0; acts[i] = 4; }

	compute_expected(wmat, acts, 1, K, shift, expected);
	if (run_test(wmat, acts, 1, K, shift, results) < 0) {
		TEST_FAIL("D4", "timeout"); free(wmat); free(acts); return;
	}
	ASSERT_EQ("D4", results[0], expected[0]);
	free(wmat);
	free(acts);
}

/* ================================================================== */
/*  Category E: Weight type coverage                                  */
/* ================================================================== */

static void test_E1_all_weight_types(void)
{
	/* All three weight types in one computation */
	const int M = 1, K = 192, shift = 1;
	int8_t wmat[192], acts[192], results[1], expected[1];
	int i;

	printf("  E1: All weight types (-1, 0, +1) in single computation\n");

	/* 64 x +1, 64 x 0, 64 x -1 */
	for (i = 0; i < 64; i++)   wmat[i] = 1;
	for (i = 64; i < 128; i++) wmat[i] = 0;
	for (i = 128; i < 192; i++) wmat[i] = -1;
	for (i = 0; i < 192; i++) acts[i] = 3;

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("E1", "timeout"); return;
	}
	ASSERT_EQ("E1", results[0], expected[0]);  /* (64*3 + 0 - 64*3)>>1 = 0 */
}

/* ================================================================== */
/*  Category F: Negative activations                                  */
/* ================================================================== */

static void test_F1_negative_acts(void)
{
	const int M = 1, K = 64, shift = 0;
	int8_t wmat[64], acts[64], results[1], expected[1];
	int i;

	printf("  F1: Negative activations, all +1 weights\n");

	for (i = 0; i < 64; i++) { wmat[i] = 1; acts[i] = -2; }

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("F1", "timeout"); return;
	}
	ASSERT_EQ("F1", results[0], expected[0]);  /* -128 */
}

static void test_F2_double_negation(void)
{
	const int M = 1, K = 64, shift = 1;
	int8_t wmat[64], acts[64], results[1], expected[1];
	int i;

	printf("  F2: Double negation (neg acts * neg weights = positive)\n");

	for (i = 0; i < 64; i++) { wmat[i] = -1; acts[i] = -2; }

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("F2", "timeout"); return;
	}
	ASSERT_EQ("F2", results[0], expected[0]);  /* 64*2 = 128, >>1 = 64 */
}

static void test_F3_mixed_signs(void)
{
	const int M = 1, K = 64, shift = 0;
	int8_t wmat[64], acts[64], results[1], expected[1];
	int i;

	printf("  F3: Mixed positive and negative activations\n");

	for (i = 0; i < 64; i++) {
		wmat[i] = 1;
		acts[i] = (i < 32) ? 3 : -3;
	}

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("F3", "timeout"); return;
	}
	ASSERT_EQ("F3", results[0], expected[0]);  /* 32*3 - 32*3 = 0 */
}

static void test_F4_neg_acts_neg_weights(void)
{
	const int M = 1, K = 64, shift = 0;
	int8_t wmat[64], acts[64], results[1], expected[1];
	int i;

	printf("  F4: Half neg acts with +1 weights, half neg acts with -1 weights\n");

	for (i = 0; i < 32; i++) { wmat[i] = 1;  acts[i] = -1; }
	for (i = 32; i < 64; i++) { wmat[i] = -1; acts[i] = -1; }

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("F4", "timeout"); return;
	}
	ASSERT_EQ("F4", results[0], expected[0]);  /* -32 + 32 = 0 */
}

/* ================================================================== */
/*  Category G: Shift amount sweep                                    */
/* ================================================================== */

static void test_G1_shift_sweep(void)
{
	const int M = 1, K = 64;
	int8_t wmat[64], acts[64];
	int i, shift;

	printf("  G1: Shift sweep 0-9 with fixed accumulator=64\n");

	for (i = 0; i < 64; i++) { wmat[i] = 1; acts[i] = 1; }

	/* accumulator = 64 for each shift value */
	for (shift = 0; shift <= 9; shift++) {
		int8_t results[1], expected[1];
		char name[32];

		compute_expected(wmat, acts, M, K, shift, expected);
		if (run_test(wmat, acts, M, K, shift, results) < 0) {
			snprintf(name, sizeof(name), "G1 shift=%d", shift);
			TEST_FAIL(name, "timeout");
			continue;
		}
		snprintf(name, sizeof(name), "G1 shift=%d", shift);
		ASSERT_EQ(name, results[0], expected[0]);
	}
}

/* ================================================================== */
/*  Category H: Clamp behavior                                        */
/* ================================================================== */

static void test_H1_negative_overflow(void)
{
	const int M = 1, K = 64, shift = 0;
	int8_t wmat[64], acts[64], results[1], expected[1];
	int i;

	printf("  H1: Negative overflow (all -1 weights, act=4 => -256 -> clamp -128)\n");

	for (i = 0; i < 64; i++) { wmat[i] = -1; acts[i] = 4; }

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("H1", "timeout"); return;
	}
	ASSERT_EQ("H1", results[0], expected[0]);  /* -128 */
}

static void test_H2_exact_pos_127(void)
{
	/* Construct accumulator = exactly 127 (no clamp needed) */
	/* 127 = 63*2 + 1*1.  Use 63 weights=+1 with act=2, plus 1 weight=+1 with act=1.
	 * But all activations are shared... use K=128: 63 +1 weights (act=2) + 1 +1 (act forced)
	 * Simpler: K=64, shift=0: need sum=127. Use acts[i]=2 for 63 of them and acts[63]=1.
	 * wmat all +1: sum = 63*2 + 1 = 127 */
	const int M = 1, K = 64, shift = 0;
	int8_t wmat[64], acts[64], results[1], expected[1];
	int i;

	printf("  H2: Exact +127 (no clamping)\n");

	for (i = 0; i < 64; i++) wmat[i] = 1;
	for (i = 0; i < 63; i++) acts[i] = 2;
	acts[63] = 1;

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("H2", "timeout"); return;
	}
	ASSERT_EQ("H2", results[0], expected[0]);  /* 127 */
}

static void test_H3_exact_neg_128(void)
{
	/* Construct accumulator = exactly -128 */
	/* K=64, all -1 weights, act=2: sum = -128, shift=0 => -128 exactly */
	const int M = 1, K = 64, shift = 0;
	int8_t wmat[64], acts[64], results[1], expected[1];
	int i;

	printf("  H3: Exact -128 (no clamping)\n");

	for (i = 0; i < 64; i++) { wmat[i] = -1; acts[i] = 2; }

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("H3", "timeout"); return;
	}
	ASSERT_EQ("H3", results[0], expected[0]);  /* -128 */
}

static void test_H4_just_over_127(void)
{
	/* Accumulator = 128, should clamp to 127 */
	const int M = 1, K = 64, shift = 0;
	int8_t wmat[64], acts[64], results[1], expected[1];
	int i;

	printf("  H4: Just over +127 (128 -> clamp to 127)\n");

	for (i = 0; i < 64; i++) wmat[i] = 1;
	for (i = 0; i < 64; i++) acts[i] = 2;  /* sum = 128 */

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("H4", "timeout"); return;
	}
	ASSERT_EQ("H4", results[0], expected[0]);  /* 127 (clamped) */
}

static void test_H5_just_under_neg_128(void)
{
	/* Accumulator = -192, should clamp to -128 */
	const int M = 1, K = 64, shift = 0;
	int8_t wmat[64], acts[64], results[1], expected[1];
	int i;

	printf("  H5: Just under -128 (-192 -> clamp to -128)\n");

	for (i = 0; i < 64; i++) { wmat[i] = -1; acts[i] = 3; }  /* sum = -192 */

	compute_expected(wmat, acts, M, K, shift, expected);
	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("H5", "timeout"); return;
	}
	ASSERT_EQ("H5", results[0], expected[0]);  /* -128 (clamped) */
}

/* ================================================================== */
/*  Category I: Register read-back                                    */
/* ================================================================== */

static void test_I1_register_readback(void)
{
	printf("  I1: Register read-back (write config, read, verify)\n");

	reg_write(g_bitnet, REG_WEIGHT_BASE, 0x30001000);
	reg_write(g_bitnet, REG_DIM_M,       42);
	reg_write(g_bitnet, REG_DIM_K,       256);
	reg_write(g_bitnet, REG_SHIFT_AMT,   7);

	ASSERT_EQ("I1 WEIGHT_BASE", reg_read(g_bitnet, REG_WEIGHT_BASE), 0x30001000);
	ASSERT_EQ("I1 DIM_M",       reg_read(g_bitnet, REG_DIM_M),       42);
	ASSERT_EQ("I1 DIM_K",       reg_read(g_bitnet, REG_DIM_K),       256);
	ASSERT_EQ("I1 SHIFT_AMT",   reg_read(g_bitnet, REG_SHIFT_AMT),   7);

	/* Restore defaults for subsequent tests */
	reg_write(g_bitnet, REG_WEIGHT_BASE, DDR3_BASE);
}

/* ================================================================== */
/*  Category J: Performance counter                                   */
/* ================================================================== */

static void test_J1_perf_counter(void)
{
	int8_t wmat_small[64], wmat_large[256];
	int8_t acts_small[64], acts_large[256];
	int8_t results[1];
	uint32_t cycles_small, cycles_large;
	int i;

	printf("  J1: Performance counter non-zero and scales with K\n");

	/* Small computation: K=64 */
	for (i = 0; i < 64; i++)  { wmat_small[i] = 1; acts_small[i] = 1; }
	if (run_test(wmat_small, acts_small, 1, 64, 0, results) < 0) {
		TEST_FAIL("J1 small", "timeout"); return;
	}
	cycles_small = reg_read(g_bitnet, REG_PERF_CYCLES);

	/* Large computation: K=256 */
	for (i = 0; i < 256; i++) { wmat_large[i] = 1; acts_large[i] = 1; }
	if (run_test(wmat_large, acts_large, 1, 256, 2, results) < 0) {
		TEST_FAIL("J1 large", "timeout"); return;
	}
	cycles_large = reg_read(g_bitnet, REG_PERF_CYCLES);

	printf("    INFO: K=64 => %u cycles, K=256 => %u cycles\n", cycles_small, cycles_large);

	ASSERT_GT("J1 non-zero", (int)cycles_small, 0);
	ASSERT_GT("J1 scales",   (int)cycles_large, (int)cycles_small);
}

/* ================================================================== */
/*  Category K: Back-to-back computations                             */
/* ================================================================== */

static void test_K1_no_state_leak(void)
{
	/* Run two different computations back-to-back, verify no state leaks */
	const int M = 1, K = 64, shift = 0;
	int8_t wmat1[64], wmat2[64], acts1[64], acts2[64];
	int8_t results[1], expected[1];
	int i;

	printf("  K1: Back-to-back, no state leak\n");

	/* Run 1: all +1, act=1 => 64 */
	for (i = 0; i < 64; i++) { wmat1[i] = 1; acts1[i] = 1; }
	compute_expected(wmat1, acts1, M, K, shift, expected);
	if (run_test(wmat1, acts1, M, K, shift, results) < 0) {
		TEST_FAIL("K1 run1", "timeout"); return;
	}
	ASSERT_EQ("K1 run1", results[0], expected[0]);

	/* Run 2: all 0, act=100 => 0 */
	for (i = 0; i < 64; i++) { wmat2[i] = 0; acts2[i] = 100; }
	compute_expected(wmat2, acts2, M, K, shift, expected);
	if (run_test(wmat2, acts2, M, K, shift, results) < 0) {
		TEST_FAIL("K1 run2", "timeout"); return;
	}
	ASSERT_EQ("K1 run2", results[0], expected[0]);
}

static void test_K2_dimension_change(void)
{
	/* Change dimensions between runs */
	int8_t wmat1[64], wmat2[128], acts1[64], acts2[128];
	int8_t results[2], expected[2];
	int i;

	printf("  K2: Dimension change between runs\n");

	/* Run 1: M=1, K=64 */
	for (i = 0; i < 64; i++) { wmat1[i] = 1; acts1[i] = 1; }
	compute_expected(wmat1, acts1, 1, 64, 0, expected);
	if (run_test(wmat1, acts1, 1, 64, 0, results) < 0) {
		TEST_FAIL("K2 run1", "timeout"); return;
	}
	ASSERT_EQ("K2 run1", results[0], expected[0]);

	/* Run 2: M=2, K=128 */
	for (i = 0; i < 128; i++) wmat2[i] = 1;
	/* Row 1 is also all +1 (wmat2 treated as 2x128 but we only have 128 entries) */
	/* Need 2*128 weights */
	{
		int8_t *wmat2_full = (int8_t *)calloc(2 * 128, sizeof(int8_t));
		if (!wmat2_full) { TEST_FAIL("K2", "alloc failed"); return; }
		for (i = 0; i < 128; i++) { wmat2_full[i] = 1; wmat2_full[128 + i] = -1; }
		for (i = 0; i < 128; i++) acts2[i] = 1;

		compute_expected(wmat2_full, acts2, 2, 128, 1, expected);
		if (run_test(wmat2_full, acts2, 2, 128, 1, results) < 0) {
			TEST_FAIL("K2 run2", "timeout"); free(wmat2_full); return;
		}
		ASSERT_EQ("K2 run2 row0", results[0], expected[0]);  /* 128>>1 = 64 */
		ASSERT_EQ("K2 run2 row1", results[1], expected[1]);  /* -128>>1 = -64 */
		free(wmat2_full);
	}
}

/* ================================================================== */
/*  Category L: Known-answer vectors                                  */
/* ================================================================== */

static void test_L1_known_answer_1(void)
{
	/* Hand-computed: K=64, all +1 weights, act=3, shift=2
	 * acc = 64*3 = 192, 192>>2 = 48 */
	const int M = 1, K = 64, shift = 2;
	int8_t wmat[64], acts[64], results[1];
	int i;

	printf("  L1: Known answer: 64*3 >> 2 = 48\n");

	for (i = 0; i < 64; i++) { wmat[i] = 1; acts[i] = 3; }

	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("L1", "timeout"); return;
	}
	ASSERT_EQ("L1", results[0], 48);
}

static void test_L2_known_answer_2(void)
{
	/* Hand-computed: K=128, row0: first 64 = +1, rest = -1, act=2, shift=0
	 * acc = 64*2 - 64*2 = 0 */
	/* Row1: all +1, act=2, shift=0 => acc = 128*2 = 256, clamp to 127 */
	const int M = 2, K = 128, shift = 0;
	int8_t wmat[2 * 128], acts[128], results[2];
	int i;

	printf("  L2: Known answer: row0=0, row1=127 (clamped from 256)\n");

	for (i = 0; i < 64; i++)   wmat[i] = 1;
	for (i = 64; i < 128; i++) wmat[i] = -1;
	for (i = 0; i < 128; i++)  wmat[128 + i] = 1;
	for (i = 0; i < 128; i++)  acts[i] = 2;

	if (run_test(wmat, acts, M, K, shift, results) < 0) {
		TEST_FAIL("L2", "timeout"); return;
	}
	ASSERT_EQ("L2 row0", results[0], 0);
	ASSERT_EQ("L2 row1", results[1], 127);
}

/* ================================================================== */
/*  Test registry                                                     */
/* ================================================================== */

typedef struct {
	const char *name;
	void (*func)(void);
} test_entry_t;

static const test_entry_t test_registry[] = {
	/* Category W: Weight type basics */
	{ "W1", test_W1_all_plus_one },
	{ "W2", test_W2_all_zero },
	{ "W3", test_W3_all_minus_one },
	{ "W4", test_W4_mixed_cancel },
	{ "W5", test_W5_positive_clamp },

	/* Category A: Multi-row */
	{ "A1", test_A1_multi_row_2 },
	{ "A2", test_A2_multi_row_4 },
	{ "A3", test_A3_multi_row_8 },

	/* Category B: Multi-tile */
	{ "B1", test_B1_two_tiles },
	{ "B2", test_B2_three_tiles },
	{ "B3", test_B3_four_tiles },
	{ "B4", test_B4_tile_accumulation },
	{ "B5", test_B5_partial_last_tile },

	/* Category C: Multi-row + multi-tile */
	{ "C1", test_C1_2x128 },
	{ "C2", test_C2_4x192 },

	/* Category D: Boundary dimensions */
	{ "D1", test_D1_min_dims },
	{ "D2", test_D2_K_not_mult_64 },
	{ "D3", test_D3_M16 },
	{ "D4", test_D4_large_K },

	/* Category E: Weight type coverage */
	{ "E1", test_E1_all_weight_types },

	/* Category F: Negative activations */
	{ "F1", test_F1_negative_acts },
	{ "F2", test_F2_double_negation },
	{ "F3", test_F3_mixed_signs },
	{ "F4", test_F4_neg_acts_neg_weights },

	/* Category G: Shift amounts */
	{ "G1", test_G1_shift_sweep },

	/* Category H: Clamp behavior */
	{ "H1", test_H1_negative_overflow },
	{ "H2", test_H2_exact_pos_127 },
	{ "H3", test_H3_exact_neg_128 },
	{ "H4", test_H4_just_over_127 },
	{ "H5", test_H5_just_under_neg_128 },

	/* Category I: Register readback */
	{ "I1", test_I1_register_readback },

	/* Category J: Performance counter */
	{ "J1", test_J1_perf_counter },

	/* Category K: Back-to-back */
	{ "K1", test_K1_no_state_leak },
	{ "K2", test_K2_dimension_change },

	/* Category L: Known-answer vectors */
	{ "L1", test_L1_known_answer_1 },
	{ "L2", test_L2_known_answer_2 },
};

#define NUM_TESTS (sizeof(test_registry) / sizeof(test_registry[0]))

/* ================================================================== */
/*  Main                                                              */
/* ================================================================== */

int main(int argc, char *argv[])
{
	const char *filter = (argc > 1) ? argv[1] : NULL;
	size_t i;

	printf("=== BitNet Accelerator Comprehensive Test Suite ===\n");
	if (filter)
		printf("Filter: \"%s\"\n", filter);
	printf("\n");

	if (mmap_init() < 0)
		return 1;

	printf("STATUS reg = 0x%08X\n\n", reg_read(g_bitnet, REG_STATUS));

	for (i = 0; i < NUM_TESTS; i++) {
		/* Apply filter: test name must start with the filter string */
		if (filter && strncmp(test_registry[i].name, filter, strlen(filter)) != 0)
			continue;

		printf("[%s]\n", test_registry[i].name);
		test_registry[i].func();
		printf("\n");
	}

	printf("========================================\n");
	printf("  RESULTS: %d / %d passed, %d failed\n",
		g_tests_passed, g_tests_total, g_tests_failed);
	printf("========================================\n");

	mmap_cleanup();

	return (g_tests_failed == 0) ? 0 : 1;
}
