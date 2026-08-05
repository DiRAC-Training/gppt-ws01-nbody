all:
	nvcc -O3 --use_fast_math -lineinfo -g main.cu

fmt:
	clang-format -i --style=llvm *.cu
