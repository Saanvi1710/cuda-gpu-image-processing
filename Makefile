NVCC = nvcc
CFLAGS = -O2

TARGET = image_processor
SOURCE = image_processing.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(CFLAGS) $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) data output
