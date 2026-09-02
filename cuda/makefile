all: nbody

nbody: nbody.cu
	nvcc -O3 -lineinfo -g -o nbody nbody.cu

fmt:
	clang-format -i --style=llvm *.cu

clean:
	rm nbody

.PHONY: all fmt clean
