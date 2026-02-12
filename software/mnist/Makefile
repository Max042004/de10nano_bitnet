CC = arm-linux-gnueabihf-gcc
CFLAGS = -O2 -Wall
TARGET = mnist_inference

all: $(TARGET)

$(TARGET): mnist_inference.c generated/mnist_weights.h generated/mnist_test_data.h
	$(CC) $(CFLAGS) -o $@ $< -lrt

run: $(TARGET)
	sudo ./$(TARGET) | tee result.log

clean:
	rm -f $(TARGET) result.log
