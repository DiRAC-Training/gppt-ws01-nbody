all: cpu gpu

cpu: nbody_start.f90 | build
	gfortran -cpp -DMAIN -O2 nbody_start.f90 -o build/main_cpu
	gfortran -cpp -DTEST -O2 nbody_start.f90 -o build/test_cpu

gpu: nbody_start.f90 | build
	nvfortran -cpp -DMAIN -mp=gpu -Minfo=mp nbody_start.f90 -o build/main_gpu
	nvfortran -cpp -DTEST -mp=gpu -Minfo=mp nbody_start.f90 -o build/test_gpu

tiled: cpu_tiled gpu_tiled

cpu_tiled: nbody_tiled_start.f90 | build
	gfortran -cpp -DMAIN -O2 nbody_tiled_start.f90 -o build/main_cpu_tiled
	gfortran -cpp -DTEST -O2 nbody_tiled_start.f90 -o build/test_cpu_tiled

gpu_tiled: nbody_tiled_start.f90 | build
	nvfortran -cpp -DMAIN -mp=gpu -Minfo=mp nbody_tiled_start.f90 -o build/main_gpu_tiled
	nvfortran -cpp -DTEST -mp=gpu -Minfo=mp nbody_tiled_start.f90 -o build/test_gpu_tiled

solutions: nbody.f90 | build
	gfortran -cpp -DMAIN -O2 nbody.f90 -o build/solution_cpu
	nvfortran -cpp -DMAIN -mp=gpu -Minfo=mp nbody.f90 -o build/solution_gpu
	nvfortran -cpp -DMAIN -DTILED -mp=gpu -Minfo=mp nbody.f90 -o build/solution_gpu_tiled

build:
	mkdir build

clean:
	rm -f *.mod *.csv
	rm -rf build

.PHONY: all tiled clean
