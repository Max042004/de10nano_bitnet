# Makefile for BitMamba FPGA integration test
# Cross-compile: make CC=arm-linux-gnueabihf-gcc
# Native (on DE10-Nano): make

CC ?= gcc
CFLAGS = -O2 -Wall -std=c99 -D_DEFAULT_SOURCE
LDLIBS = -lm

all: test_fpga_driver

test_fpga_driver: test_fpga_driver.c bitnet_fpga.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f test_fpga_driver

.PHONY: all clean
