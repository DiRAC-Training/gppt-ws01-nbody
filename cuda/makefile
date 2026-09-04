all: nbody

nbody: nbody.cu util.hpp
	nvcc -O3 -lineinfo -g -o nbody nbody.cu

nbody_tiled: nbody_tiled.cu util.hpp
	nvcc -O3 -lineinfo -g -o nbody_tiled nbody_tiled.cu

fmt:
	clang-format -i --style=llvm *.cu

clean:
	rm nbody nbody_tiled

.PHONY: all fmt clean
