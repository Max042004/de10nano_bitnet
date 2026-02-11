# Makefile — Build BitNet HPS test suite
#
# Usage:
#   make all    # Build test binary
#   make clean  # Remove binary
#
# Variables (override on command line or environment):
#   CROSS_COMPILE  — toolchain prefix (default: arm-linux-gnueabihf-)

CC             = gcc
CFLAGS         = -O2 -Wall -Wextra -Wno-unused-parameter

# Binary
BIN            = test_bitnet_comprehensive

# Sources
SRC            = test_bitnet_comprehensive.c
HDR            = bitnet_test_common.h

.PHONY: all clean run help

all: $(BIN)

$(BIN): $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ $<

run: all
	@echo "=== Running comprehensive tests ===" | tee result.log
	./$(BIN) 2>&1 | tee -a result.log

clean:
	rm -f $(BIN) result.log

help:
	@echo "BitNet HPS Test Suite — Build"
	@echo ""
	@echo "Targets:"
	@echo "  all    Build test binary (default)"
	@echo "  run    Build + run, output to result.log"
	@echo "  clean  Remove built binary and result.log"
	@echo "  help   Show this help"
	@echo ""
	@echo "Run a subset:  sudo ./$(BIN) W    # weight type basics only"
	@echo "               sudo ./$(BIN) A    # category A only"
	@echo "               sudo ./$(BIN) A1   # single test"
	@echo ""
	@echo "Variables:"
	@echo "  CROSS_COMPILE  Toolchain prefix  [$(CROSS_COMPILE)]"
