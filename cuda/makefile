all:
	nvcc -O3 -lineinfo -g -o nbody nbody.cu

fmt:
	clang-format -i --style=llvm *.cu
