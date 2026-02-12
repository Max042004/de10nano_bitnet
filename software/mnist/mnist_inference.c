/*
 * mnist_inference.c — Live MNIST inference on BitNet FPGA accelerator
 *
 * Runs a 3-layer BitNet b1.58 MLP (784->256->128->10) on the DE10-Nano.
 * Supports loading PGM/raw images from files or directories for live
 * digit recognition, plus a benchmark mode with embedded test data.
 *
 * Usage:
 *   sudo ./mnist_inference <image1.pgm> [image2.pgm] ...
 *   sudo ./mnist_inference --dir /path/to/images/
 *   sudo ./mnist_inference --benchmark
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <dirent.h>

/* Reuse mmap, register access, weight packing, etc. */
#include "../bitnet_test/bitnet_test_common.h"

/* Generated model weights and test data */
#include "generated/mnist_weights.h"
#include "generated/mnist_test_data.h"

/* ------------------------------------------------------------------ */
/* DDR3 weight layout: layers placed sequentially                     */
/* ------------------------------------------------------------------ */

static uint32_t ddr3_l1_offset;
static uint32_t ddr3_l2_offset;
static uint32_t ddr3_l3_offset;

static void load_weights_to_ddr3(void)
{
	uint32_t offset = 0;  /* in uint32 units */

	/* Layer 1 */
	ddr3_l1_offset = 0;
	memcpy((void *)&g_ddr3[offset], l1_packed, L1_DDR3_BYTES);
	offset += L1_DDR3_BYTES / 4;

	/* Layer 2 */
	ddr3_l2_offset = offset * 4;
	memcpy((void *)&g_ddr3[offset], l2_packed, L2_DDR3_BYTES);
	offset += L2_DDR3_BYTES / 4;

	/* Layer 3 */
	ddr3_l3_offset = offset * 4;
	memcpy((void *)&g_ddr3[offset], l3_packed, L3_DDR3_BYTES);
	offset += L3_DDR3_BYTES / 4;

	printf("Weights loaded to DDR3 (%u bytes).\n", (unsigned)(offset * 4));
}

/* ------------------------------------------------------------------ */
/* Software ReLU and argmax                                           */
/* ------------------------------------------------------------------ */

static void relu_int8(int8_t *buf, int n)
{
	int i;
	for (i = 0; i < n; i++)
		if (buf[i] < 0)
			buf[i] = 0;
}

static int argmax_int8(const int8_t *buf, int n)
{
	int best = 0;
	int i;
	for (i = 1; i < n; i++)
		if (buf[i] > buf[best])
			best = i;
	return best;
}

/* ------------------------------------------------------------------ */
/* Run one layer on FPGA accelerator                                  */
/* ------------------------------------------------------------------ */

static int fpga_layer(const int8_t *activations, int8_t *results,
                      int M, int K, int shift, uint32_t ddr3_byte_offset,
                      uint32_t *cycles_out)
{
	write_activations(activations, K);

	reg_write(g_bitnet, REG_WEIGHT_BASE, DDR3_BASE + ddr3_byte_offset);
	reg_write(g_bitnet, REG_DIM_M,       (uint32_t)M);
	reg_write(g_bitnet, REG_DIM_K,       (uint32_t)K);
	reg_write(g_bitnet, REG_SHIFT_AMT,   (uint32_t)shift);

	reg_write(g_bitnet, REG_CTRL, 0x1);

	if (wait_done(g_bitnet, 2000000) < 0) {
		fprintf(stderr, "FPGA timeout! M=%d K=%d\n", M, K);
		return -1;
	}

	if (cycles_out)
		*cycles_out = reg_read(g_bitnet, REG_PERF_CYCLES);

	read_results(results, M);
	return 0;
}

/* ------------------------------------------------------------------ */
/* FPGA 3-layer inference                                             */
/* ------------------------------------------------------------------ */

static int fpga_inference(const int8_t *image, uint32_t layer_cycles[3])
{
	int8_t buf1[L1_M];
	int8_t buf2[L2_M];
	int8_t buf3[L3_M];

	if (fpga_layer(image, buf1, L1_M, L1_K, L1_SHIFT,
	               ddr3_l1_offset, &layer_cycles[0]) < 0)
		return -1;
	relu_int8(buf1, L1_M);

	if (fpga_layer(buf1, buf2, L2_M, L2_K, L2_SHIFT,
	               ddr3_l2_offset, &layer_cycles[1]) < 0)
		return -1;
	relu_int8(buf2, L2_M);

	if (fpga_layer(buf2, buf3, L3_M, L3_K, L3_SHIFT,
	               ddr3_l3_offset, &layer_cycles[2]) < 0)
		return -1;

	return argmax_int8(buf3, L3_M);
}

/* ------------------------------------------------------------------ */
/* ARM software inference (for benchmark comparison)                  */
/* ------------------------------------------------------------------ */

static void arm_matvec(const int8_t *weights, const int8_t *acts,
                       int M, int K, int shift, int8_t *results)
{
	int row;
	for (row = 0; row < M; row++)
		results[row] = compute_expected_row(&weights[row * K], acts, K, shift);
}

static int arm_inference(const int8_t *image)
{
	int8_t buf1[L1_M];
	int8_t buf2[L2_M];
	int8_t buf3[L3_M];

	arm_matvec(l1_weights, image, L1_M, L1_K, L1_SHIFT, buf1);
	relu_int8(buf1, L1_M);
	arm_matvec(l2_weights, buf1, L2_M, L2_K, L2_SHIFT, buf2);
	relu_int8(buf2, L2_M);
	arm_matvec(l3_weights, buf2, L3_M, L3_K, L3_SHIFT, buf3);

	return argmax_int8(buf3, L3_M);
}

/* ------------------------------------------------------------------ */
/* Timing helper                                                      */
/* ------------------------------------------------------------------ */

static double get_time_us(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* ------------------------------------------------------------------ */
/* PGM image loader                                                   */
/* ------------------------------------------------------------------ */

/*
 * Load a PGM (P5 binary) image. Parses header, skips comments,
 * reads raw pixel bytes into caller-provided buffer.
 * Returns 0 on success, -1 on error.
 * buf must have room for at least width*height bytes.
 */
static int load_pgm(const char *filename, uint8_t *buf, int buf_size,
                    int *out_width, int *out_height)
{
	FILE *fp;
	char line[256];
	int width, height, maxval;
	int header_fields;

	fp = fopen(filename, "rb");
	if (!fp) {
		perror(filename);
		return -1;
	}

	/* Read magic */
	if (!fgets(line, sizeof(line), fp)) {
		fprintf(stderr, "%s: cannot read header\n", filename);
		fclose(fp);
		return -1;
	}
	if (line[0] != 'P' || line[1] != '5') {
		fprintf(stderr, "%s: not a P5 PGM file\n", filename);
		fclose(fp);
		return -1;
	}

	/* Read width, height, maxval — skip comment lines starting with '#' */
	header_fields = 0;
	width = height = maxval = 0;
	while (header_fields < 3) {
		if (!fgets(line, sizeof(line), fp)) {
			fprintf(stderr, "%s: truncated PGM header\n", filename);
			fclose(fp);
			return -1;
		}
		if (line[0] == '#')
			continue;

		if (header_fields == 0) {
			if (sscanf(line, "%d %d", &width, &height) == 2)
				header_fields += 2;
			else if (sscanf(line, "%d", &width) == 1)
				header_fields += 1;
		} else if (header_fields == 1) {
			if (sscanf(line, "%d", &height) == 1)
				header_fields += 1;
		} else if (header_fields == 2) {
			if (sscanf(line, "%d", &maxval) == 1)
				header_fields += 1;
		}
	}

	if (width <= 0 || height <= 0 || maxval <= 0 || maxval > 255) {
		fprintf(stderr, "%s: invalid PGM dimensions %dx%d maxval=%d\n",
		        filename, width, height, maxval);
		fclose(fp);
		return -1;
	}

	if (width * height > buf_size) {
		fprintf(stderr, "%s: image too large (%dx%d = %d pixels, max %d)\n",
		        filename, width, height, width * height, buf_size);
		fclose(fp);
		return -1;
	}

	/* Read raw pixel data */
	if ((int)fread(buf, 1, width * height, fp) != width * height) {
		fprintf(stderr, "%s: truncated pixel data\n", filename);
		fclose(fp);
		return -1;
	}

	fclose(fp);
	*out_width = width;
	*out_height = height;
	return 0;
}

/* ------------------------------------------------------------------ */
/* Nearest-neighbor resize                                            */
/* ------------------------------------------------------------------ */

static void resize_nearest(const uint8_t *src, int src_w, int src_h,
                           uint8_t *dst, int dst_w, int dst_h)
{
	int y, x;
	for (y = 0; y < dst_h; y++) {
		int sy = y * src_h / dst_h;
		for (x = 0; x < dst_w; x++) {
			int sx = x * src_w / dst_w;
			dst[y * dst_w + x] = src[sy * src_w + sx];
		}
	}
}

/* ------------------------------------------------------------------ */
/* Image preprocessing: load file -> INT8 activations                 */
/* ------------------------------------------------------------------ */

#define MAX_IMG_DIM 1024
#define MNIST_DIM   28
#define MNIST_PIXELS (MNIST_DIM * MNIST_DIM)

/*
 * Load an image file and convert to 784 INT8 activations.
 * Supports PGM (P5) and raw 784-byte binary.
 * Sets *resized = 1 if input was not 28x28.
 * Sets *orig_w, *orig_h to original dimensions.
 * Returns 0 on success, -1 on error.
 */
static int preprocess_image(const char *filename, int8_t *output,
                            int *orig_w, int *orig_h, int *resized)
{
	uint8_t raw_buf[MAX_IMG_DIM * MAX_IMG_DIM];
	uint8_t img28[MNIST_PIXELS];
	int width, height;
	FILE *fp;
	uint8_t magic[2];
	int i;

	*resized = 0;
	*orig_w = 28;
	*orig_h = 28;

	/* Peek at first 2 bytes to detect format */
	fp = fopen(filename, "rb");
	if (!fp) {
		perror(filename);
		return -1;
	}
	if (fread(magic, 1, 2, fp) != 2) {
		fprintf(stderr, "%s: cannot read file\n", filename);
		fclose(fp);
		return -1;
	}
	fclose(fp);

	if (magic[0] == 'P' && magic[1] == '5') {
		/* PGM format */
		if (load_pgm(filename, raw_buf, sizeof(raw_buf),
		             &width, &height) < 0)
			return -1;

		*orig_w = width;
		*orig_h = height;

		if (width == MNIST_DIM && height == MNIST_DIM) {
			memcpy(img28, raw_buf, MNIST_PIXELS);
		} else {
			resize_nearest(raw_buf, width, height,
			               img28, MNIST_DIM, MNIST_DIM);
			*resized = 1;
		}
	} else {
		/* Assume raw 784-byte binary */
		fp = fopen(filename, "rb");
		if (!fp) {
			perror(filename);
			return -1;
		}

		fseek(fp, 0, SEEK_END);
		long fsize = ftell(fp);
		fseek(fp, 0, SEEK_SET);

		if (fsize != MNIST_PIXELS) {
			fprintf(stderr, "%s: not PGM and size %ld != %d (expected raw 784 bytes)\n",
			        filename, fsize, MNIST_PIXELS);
			fclose(fp);
			return -1;
		}

		if ((int)fread(img28, 1, MNIST_PIXELS, fp) != MNIST_PIXELS) {
			fprintf(stderr, "%s: read error\n", filename);
			fclose(fp);
			return -1;
		}
		fclose(fp);
	}

	/* Convert uint8 [0,255] -> int8 [0,127]
	 * Matches training: ToTensor() [0,1] * 127 -> [0,127] */
	for (i = 0; i < MNIST_PIXELS; i++)
		output[i] = (int8_t)(img28[i] * 127 / 255);

	return 0;
}

/* ------------------------------------------------------------------ */
/* Check if filename has an image extension                           */
/* ------------------------------------------------------------------ */

static int has_image_ext(const char *name)
{
	const char *dot = strrchr(name, '.');
	if (!dot)
		return 0;
	return (strcasecmp(dot, ".pgm") == 0 ||
	        strcasecmp(dot, ".bin") == 0 ||
	        strcasecmp(dot, ".raw") == 0);
}

/* ------------------------------------------------------------------ */
/* Mode: process individual image files                               */
/* ------------------------------------------------------------------ */

static int mode_files(int argc, char **argv, int start_idx)
{
	int count = 0;
	int i;

	for (i = start_idx; i < argc; i++) {
		int8_t activations[MNIST_PIXELS];
		uint32_t layer_cycles[3];
		int orig_w, orig_h, resized;

		if (preprocess_image(argv[i], activations,
		                     &orig_w, &orig_h, &resized) < 0) {
			fprintf(stderr, "  Skipping %s\n", argv[i]);
			continue;
		}

		int pred = fpga_inference(activations, layer_cycles);
		if (pred < 0) {
			fprintf(stderr, "  %s: FPGA timeout\n", argv[i]);
			continue;
		}

		uint32_t total_cycles = layer_cycles[0] + layer_cycles[1] + layer_cycles[2];
		double us = total_cycles / 100.0;  /* 100 MHz clock */

		count++;
		if (resized) {
			printf("[%d] %s (%dx%d -> 28x28) -> %d  (%u cycles, %.2f us)\n",
			       count, argv[i], orig_w, orig_h, pred,
			       (unsigned)total_cycles, us);
		} else {
			printf("[%d] %s (28x28) -> %d  (%u cycles, %.2f us)\n",
			       count, argv[i], pred,
			       (unsigned)total_cycles, us);
		}
	}

	if (count == 0) {
		fprintf(stderr, "No images processed.\n");
		return 1;
	}
	printf("\n%d image(s) processed.\n", count);
	return 0;
}

/* ------------------------------------------------------------------ */
/* Mode: scan directory for image files                               */
/* ------------------------------------------------------------------ */

static int mode_dir(const char *dirpath)
{
	DIR *dp;
	struct dirent *ent;
	int count = 0;
	char filepath[512];

	dp = opendir(dirpath);
	if (!dp) {
		perror(dirpath);
		return 1;
	}

	printf("Scanning %s ...\n\n", dirpath);

	while ((ent = readdir(dp)) != NULL) {
		if (!has_image_ext(ent->d_name))
			continue;

		snprintf(filepath, sizeof(filepath), "%s/%s", dirpath, ent->d_name);

		int8_t activations[MNIST_PIXELS];
		uint32_t layer_cycles[3];
		int orig_w, orig_h, resized;

		if (preprocess_image(filepath, activations,
		                     &orig_w, &orig_h, &resized) < 0) {
			fprintf(stderr, "  Skipping %s\n", ent->d_name);
			continue;
		}

		int pred = fpga_inference(activations, layer_cycles);
		if (pred < 0) {
			fprintf(stderr, "  %s: FPGA timeout\n", ent->d_name);
			continue;
		}

		uint32_t total_cycles = layer_cycles[0] + layer_cycles[1] + layer_cycles[2];
		count++;

		if (resized) {
			printf("[%d] %s (%dx%d -> 28x28) -> %d  (%u cycles)\n",
			       count, ent->d_name, orig_w, orig_h, pred,
			       (unsigned)total_cycles);
		} else {
			printf("[%d] %s -> %d  (%u cycles)\n",
			       count, ent->d_name, pred,
			       (unsigned)total_cycles);
		}
	}

	closedir(dp);

	if (count == 0) {
		fprintf(stderr, "No .pgm/.bin/.raw files found in %s\n", dirpath);
		return 1;
	}
	printf("\n%d image(s) processed.\n", count);
	return 0;
}

/* ------------------------------------------------------------------ */
/* Mode: benchmark with embedded test data                            */
/* ------------------------------------------------------------------ */

static int mode_benchmark(void)
{
	int i;
	int fpga_correct = 0, arm_correct = 0, match_count = 0;
	uint64_t total_fpga_cycles = 0;
	double total_arm_us = 0;
	uint64_t total_l1_cyc = 0, total_l2_cyc = 0, total_l3_cyc = 0;

	printf("Model: L1(%d->%d, shift=%d), L2(%d->%d, shift=%d), "
	       "L3(%d->%d, shift=%d)\n",
	       L1_K, L1_M, L1_SHIFT,
	       L2_K, L2_M, L2_SHIFT,
	       L3_K, L3_M, L3_SHIFT);

	printf("\nRunning %d test images...\n", NUM_TEST_IMAGES);

	for (i = 0; i < NUM_TEST_IMAGES; i++) {
		uint32_t layer_cycles[3];
		int label = test_labels[i];

		/* FPGA inference */
		int fpga_pred = fpga_inference(test_images[i], layer_cycles);
		if (fpga_pred < 0) {
			printf("[%3d/%d] FPGA TIMEOUT\n", i + 1, NUM_TEST_IMAGES);
			continue;
		}
		uint32_t img_cycles = layer_cycles[0] + layer_cycles[1] + layer_cycles[2];
		total_fpga_cycles += img_cycles;
		total_l1_cyc += layer_cycles[0];
		total_l2_cyc += layer_cycles[1];
		total_l3_cyc += layer_cycles[2];

		/* ARM inference (timed) */
		double t0 = get_time_us();
		int arm_pred = arm_inference(test_images[i]);
		double arm_us = get_time_us() - t0;
		total_arm_us += arm_us;

		/* Score */
		if (fpga_pred == label) fpga_correct++;
		if (arm_pred == label)  arm_correct++;
		if (fpga_pred == arm_pred) match_count++;

		const char *status = (fpga_pred == arm_pred) ? "OK" : "MISMATCH";

		printf("[%3d/%d] Label=%d  FPGA=%d  ARM=%d  %s  "
		       "(L1:%u, L2:%u, L3:%u cyc)\n",
		       i + 1, NUM_TEST_IMAGES, label, fpga_pred, arm_pred,
		       status,
		       (unsigned)layer_cycles[0],
		       (unsigned)layer_cycles[1],
		       (unsigned)layer_cycles[2]);
	}

	/* Summary */
	double avg_fpga_cycles = (double)total_fpga_cycles / NUM_TEST_IMAGES;
	double avg_fpga_us = avg_fpga_cycles / 100.0;  /* 100 MHz clock */
	double avg_arm_us = total_arm_us / NUM_TEST_IMAGES;
	double speedup = (avg_fpga_us > 0) ? avg_arm_us / avg_fpga_us : 0;

	printf("\n=== Results ===\n");
	printf("FPGA accuracy:    %d/%d (%.2f%%)\n",
	       fpga_correct, NUM_TEST_IMAGES,
	       100.0 * fpga_correct / NUM_TEST_IMAGES);
	printf("ARM accuracy:     %d/%d (%.2f%%)\n",
	       arm_correct, NUM_TEST_IMAGES,
	       100.0 * arm_correct / NUM_TEST_IMAGES);
	printf("FPGA avg:         %.0f cycles/image (%.2f us @ 50 MHz)\n",
	       avg_fpga_cycles, avg_fpga_us);
	printf("  L1 avg: %.0f cyc  L2 avg: %.0f cyc  L3 avg: %.0f cyc\n",
	       (double)total_l1_cyc / NUM_TEST_IMAGES,
	       (double)total_l2_cyc / NUM_TEST_IMAGES,
	       (double)total_l3_cyc / NUM_TEST_IMAGES);
	printf("ARM avg:          %.2f us/image\n", avg_arm_us);
	printf("Speedup:          %.2fx\n", speedup);
	printf("FPGA vs ARM match: %d/%d\n", match_count, NUM_TEST_IMAGES);

	return 0;
}

/* ------------------------------------------------------------------ */
/* Usage                                                              */
/* ------------------------------------------------------------------ */

static void usage(const char *progname)
{
	fprintf(stderr,
		"Usage:\n"
		"  %s <image.pgm> [image2.pgm] ...   Process image files\n"
		"  %s --dir <path>                    Scan directory for .pgm/.bin/.raw files\n"
		"  %s --benchmark                     Run 100 embedded test images\n",
		progname, progname, progname);
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
	int ret;

	if (argc < 2) {
		usage(argv[0]);
		return 1;
	}

	/* Determine mode */
	int is_benchmark = (strcmp(argv[1], "--benchmark") == 0);
	int is_dir = (strcmp(argv[1], "--dir") == 0);

	if (is_dir && argc < 3) {
		fprintf(stderr, "--dir requires a path argument\n");
		usage(argv[0]);
		return 1;
	}

	if (is_benchmark)
		printf("=== MNIST BitNet Benchmark (100 embedded test images) ===\n");
	else
		printf("=== MNIST BitNet Inference ===\n");

	/* Init memory mapping */
	if (mmap_init() < 0) {
		fprintf(stderr, "mmap_init failed (run as root?)\n");
		return 1;
	}

	/* Load packed weights to DDR3 */
	load_weights_to_ddr3();
	printf("\n");

	/* Run selected mode */
	if (is_benchmark)
		ret = mode_benchmark();
	else if (is_dir)
		ret = mode_dir(argv[2]);
	else
		ret = mode_files(argc, argv, 1);

	mmap_cleanup();
	return ret;
}
