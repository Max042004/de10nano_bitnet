/*
 * test_bitnet.c — Minimal HPS test for BitNet accelerator on DE10-Nano
 *
 * Compile on DE10-Nano:
 *   gcc -O2 -o test_bitnet test_bitnet.c
 *
 * Run (requires root for /dev/mem):
 *   sudo ./test_bitnet
 */

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

#define DDR3_BASE       0x30000000   /* DDR3 region for weights (must not overlap Linux) */
#define DDR3_SPAN       0x00010000   /* 64 KB for weight data */

/* --- BitNet accelerator register offsets (byte-addressed) --- */
#define REG_CTRL        0x00   /* W:  bit 0 = START (pulse) */
#define REG_STATUS      0x04   /* R:  bit 0 = BUSY, bit 1 = DONE */
#define REG_WEIGHT_BASE 0x08   /* RW: DDR3 byte address of weights */
#define REG_DIM_M       0x0C   /* RW: number of output rows */
#define REG_DIM_K       0x10   /* RW: input vector length */
#define REG_SHIFT_AMT   0x14   /* RW: requantization shift (0-31) */
#define REG_PERF_CYCLES 0x18    /* R:  cycle count of last run */
#define REG_ACT_BASE    0x80    /* W:  activation[i] at 0x80 + i*4 (up to 0x47F) */
#define REG_RES_BASE    0x800   /* R:  result[i]     at 0x800 + i*4 */

/* --- Weight encoding (2 bits per weight, 64 weights per 128-bit word) --- */
/*  00 = 0    01 = +1    10 = -1    11 = reserved                         */

#define NUM_PES 64

/* No explicit cache flush needed: DDR3 is mmap'd with O_SYNC, which
 * creates a non-cacheable mapping on ARM.  CPU writes go directly to
 * the SDRAM controller, so the FPGA sees them immediately via f2sdram. */

/* Helper: write a 32-bit register */
static inline void reg_write(volatile uint32_t *base, uint32_t offset, uint32_t val)
{
	base[offset / 4] = val;
}

/* Helper: read a 32-bit register */
static inline uint32_t reg_read(volatile uint32_t *base, uint32_t offset)
{
	return base[offset / 4];
}

/* Wait for accelerator DONE (STATUS bit 1) */
static int wait_done(volatile uint32_t *base, int timeout_us)
{
	while (timeout_us > 0) {
		uint32_t st = reg_read(base, REG_STATUS);
		if (st & 0x2)
			return 0;  /* done */
		usleep(10);
		timeout_us -= 10;
	}
	return -1;  /* timeout */
}

/* Pack 64 ternary weights into a 128-bit DDR3 word.
 * weights[i] = -1, 0, or +1.  Stored as 2 bits: 00=0, 01=+1, 10=-1 */
static void pack_weights(const int8_t weights[64], uint32_t out[4])
{
	int i;
	memset(out, 0, 16);
	for (i = 0; i < 64; i++) {
		uint32_t enc;
		if (weights[i] == 1)       enc = 0x1;  /* 01 */
		else if (weights[i] == -1) enc = 0x2;  /* 10 */
		else                       enc = 0x0;  /* 00 */
		out[i / 16] |= enc << ((i % 16) * 2);
	}
}

/* ------------------------------------------------------------------ */
/*                          Test cases                                */
/* ------------------------------------------------------------------ */

static int test_identity(volatile uint32_t *bitnet, volatile uint32_t *ddr3)
{
	int i, pass = 1;
	int8_t wt[64];
	uint32_t packed[4];

	printf("TEST 1: All +1 weights, act=1, M=1, K=64, shift=0\n");

	/* Weights: all +1 */
	for (i = 0; i < 64; i++) wt[i] = 1;
	pack_weights(wt, packed);
	for (i = 0; i < 4; i++) ddr3[i] = packed[i];


	/* Configure */
	reg_write(bitnet, REG_WEIGHT_BASE, DDR3_BASE);
	reg_write(bitnet, REG_DIM_M,       1);
	reg_write(bitnet, REG_DIM_K,       64);
	reg_write(bitnet, REG_SHIFT_AMT,   0);

	/* Write activations: all = 1 */
	for (i = 0; i < 64; i++)
		reg_write(bitnet, REG_ACT_BASE + i * 4, 1);

	/* Start */
	reg_write(bitnet, REG_CTRL, 0x1);

	if (wait_done(bitnet, 100000) < 0) {
		printf("  FAIL: timeout\n");
		return 0;
	}

	/* Expected: sum(1*1, 64 times) = 64 */
	int8_t result = (int8_t)(reg_read(bitnet, REG_RES_BASE) & 0xFF);
	uint32_t cycles = reg_read(bitnet, REG_PERF_CYCLES);

	if (result == 64)
		printf("  PASS: result = %d (expected 64)\n", result);
	else {
		printf("  FAIL: result = %d (expected 64)\n", result);
		pass = 0;
	}
	printf("  INFO: %u cycles\n", cycles);
	return pass;
}

static int test_zero_weights(volatile uint32_t *bitnet, volatile uint32_t *ddr3)
{
	int i, pass = 1;
	int8_t wt[64];
	uint32_t packed[4];

	printf("TEST 2: Zero weights, act=100, M=1, K=64, shift=0\n");

	/* Weights: all 0 */
	for (i = 0; i < 64; i++) wt[i] = 0;
	pack_weights(wt, packed);
	for (i = 0; i < 4; i++) ddr3[i] = packed[i];


	reg_write(bitnet, REG_WEIGHT_BASE, DDR3_BASE);
	reg_write(bitnet, REG_DIM_M,       1);
	reg_write(bitnet, REG_DIM_K,       64);
	reg_write(bitnet, REG_SHIFT_AMT,   0);

	for (i = 0; i < 64; i++)
		reg_write(bitnet, REG_ACT_BASE + i * 4, 100);

	reg_write(bitnet, REG_CTRL, 0x1);

	if (wait_done(bitnet, 100000) < 0) {
		printf("  FAIL: timeout\n");
		return 0;
	}

	int8_t result = (int8_t)(reg_read(bitnet, REG_RES_BASE) & 0xFF);
	if (result == 0)
		printf("  PASS: result = %d (expected 0)\n", result);
	else {
		printf("  FAIL: result = %d (expected 0)\n", result);
		pass = 0;
	}
	return pass;
}

static int test_negate(volatile uint32_t *bitnet, volatile uint32_t *ddr3)
{
	int i, pass = 1;
	int8_t wt[64];
	uint32_t packed[4];

	printf("TEST 3: All -1 weights, act=2, M=1, K=64, shift=1\n");

	/* Weights: all -1 */
	for (i = 0; i < 64; i++) wt[i] = -1;
	pack_weights(wt, packed);
	for (i = 0; i < 4; i++) ddr3[i] = packed[i];


	reg_write(bitnet, REG_WEIGHT_BASE, DDR3_BASE);
	reg_write(bitnet, REG_DIM_M,       1);
	reg_write(bitnet, REG_DIM_K,       64);
	reg_write(bitnet, REG_SHIFT_AMT,   1);

	for (i = 0; i < 64; i++)
		reg_write(bitnet, REG_ACT_BASE + i * 4, 2);

	reg_write(bitnet, REG_CTRL, 0x1);

	if (wait_done(bitnet, 100000) < 0) {
		printf("  FAIL: timeout\n");
		return 0;
	}

	/* 64 * (-2) = -128, shift>>1 = -64 */
	int8_t result = (int8_t)(reg_read(bitnet, REG_RES_BASE) & 0xFF);
	if (result == -64)
		printf("  PASS: result = %d (expected -64)\n", result);
	else {
		printf("  FAIL: result = %d (expected -64)\n", result);
		pass = 0;
	}
	return pass;
}

static int test_mixed(volatile uint32_t *bitnet, volatile uint32_t *ddr3)
{
	int i, pass = 1;
	int8_t wt[64];
	uint32_t packed[4];

	printf("TEST 4: Mixed weights (+1/-1), act=1, M=1, K=64, shift=0\n");

	/* First 32 = +1, last 32 = -1 => sum = 0 */
	for (i = 0; i < 32; i++) wt[i] = 1;
	for (i = 32; i < 64; i++) wt[i] = -1;
	pack_weights(wt, packed);
	for (i = 0; i < 4; i++) ddr3[i] = packed[i];


	reg_write(bitnet, REG_WEIGHT_BASE, DDR3_BASE);
	reg_write(bitnet, REG_DIM_M,       1);
	reg_write(bitnet, REG_DIM_K,       64);
	reg_write(bitnet, REG_SHIFT_AMT,   0);

	for (i = 0; i < 64; i++)
		reg_write(bitnet, REG_ACT_BASE + i * 4, 1);

	reg_write(bitnet, REG_CTRL, 0x1);

	if (wait_done(bitnet, 100000) < 0) {
		printf("  FAIL: timeout\n");
		return 0;
	}

	int8_t result = (int8_t)(reg_read(bitnet, REG_RES_BASE) & 0xFF);
	if (result == 0)
		printf("  PASS: result = %d (expected 0)\n", result);
	else {
		printf("  FAIL: result = %d (expected 0)\n", result);
		pass = 0;
	}
	return pass;
}

static int test_clamp(volatile uint32_t *bitnet, volatile uint32_t *ddr3)
{
	int i, pass = 1;
	int8_t wt[64];
	uint32_t packed[4];

	printf("TEST 5: Positive clamp, all +1, act=4, shift=0 (64*4=256 -> clamp 127)\n");

	for (i = 0; i < 64; i++) wt[i] = 1;
	pack_weights(wt, packed);
	for (i = 0; i < 4; i++) ddr3[i] = packed[i];


	reg_write(bitnet, REG_WEIGHT_BASE, DDR3_BASE);
	reg_write(bitnet, REG_DIM_M,       1);
	reg_write(bitnet, REG_DIM_K,       64);
	reg_write(bitnet, REG_SHIFT_AMT,   0);

	for (i = 0; i < 64; i++)
		reg_write(bitnet, REG_ACT_BASE + i * 4, 4);

	reg_write(bitnet, REG_CTRL, 0x1);

	if (wait_done(bitnet, 100000) < 0) {
		printf("  FAIL: timeout\n");
		return 0;
	}

	int8_t result = (int8_t)(reg_read(bitnet, REG_RES_BASE) & 0xFF);
	if (result == 127)
		printf("  PASS: result = %d (expected 127, clamped)\n", result);
	else {
		printf("  FAIL: result = %d (expected 127)\n", result);
		pass = 0;
	}
	return pass;
}

/* ------------------------------------------------------------------ */
/*                             Main                                   */
/* ------------------------------------------------------------------ */

int main(void)
{
	int fd;
	volatile uint32_t *lw_bridge, *bitnet, *ddr3;
	int passed = 0, total = 5;

	printf("=== BitNet Accelerator HPS Test ===\n\n");

	fd = open("/dev/mem", O_RDWR | O_SYNC);
	if (fd < 0) {
		perror("open /dev/mem");
		return 1;
	}

	/* Map lightweight bridge (BitNet registers) */
	lw_bridge = (volatile uint32_t *)mmap(NULL, LW_BRIDGE_SPAN,
		PROT_READ | PROT_WRITE, MAP_SHARED, fd, LW_BRIDGE_BASE);
	if (lw_bridge == MAP_FAILED) {
		perror("mmap lw_bridge");
		close(fd);
		return 1;
	}

	/* Map DDR3 region for weights */
	ddr3 = (volatile uint32_t *)mmap(NULL, DDR3_SPAN,
		PROT_READ | PROT_WRITE, MAP_SHARED, fd, DDR3_BASE);
	if (ddr3 == MAP_FAILED) {
		perror("mmap ddr3");
		munmap((void *)lw_bridge, LW_BRIDGE_SPAN);
		close(fd);
		return 1;
	}

	/*
	 * BitNet base offset within lightweight bridge.
	 * Adjust BITNET_OFFSET to match your Platform Designer assignment.
	 * Default: 0x0 (first slave on the bridge).
	 */
	#define BITNET_OFFSET 0x0
	bitnet = (volatile uint32_t *)((uint8_t *)lw_bridge + BITNET_OFFSET);

	/* Verify we can read a register */
	printf("STATUS reg = 0x%08X\n\n", reg_read(bitnet, REG_STATUS));

	passed += test_identity(bitnet, ddr3);
	printf("\n");
	passed += test_zero_weights(bitnet, ddr3);
	printf("\n");
	passed += test_negate(bitnet, ddr3);
	printf("\n");
	passed += test_mixed(bitnet, ddr3);
	printf("\n");
	passed += test_clamp(bitnet, ddr3);
	printf("\n");

	printf("========================================\n");
	printf("  RESULTS: %d / %d passed\n", passed, total);
	printf("========================================\n");

	munmap((void *)ddr3, DDR3_SPAN);
	munmap((void *)lw_bridge, LW_BRIDGE_SPAN);
	close(fd);

	return (passed == total) ? 0 : 1;
}
