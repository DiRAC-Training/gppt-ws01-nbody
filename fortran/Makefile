all: cpu gpu

ifdef DOUBLE_PRECISION
FLAGS:=
else
FLAGS:=-DSINGLE_PRECISION
endif

cpu: nbody_start.f90 | build
	gfortran -cpp -DMAIN $(FLAGS) -O2 nbody_start.f90 -o build/main_cpu
	gfortran -cpp -DTEST $(FLAGS) -O2 nbody_start.f90 -o build/test_cpu

gpu: nbody_start.f90 | build
	nvfortran -cpp -DMAIN $(FLAGS) -mp=gpu -Minfo=mp nbody_start.f90 -o build/main_gpu
	nvfortran -cpp -DTEST $(FLAGS) -mp=gpu -Minfo=mp nbody_start.f90 -o build/test_gpu

tiled: nbody_tiled_start.f90 | build
	nvfortran -cpp -DMAIN  $(FLAGS) -DTILED -mp=gpu -Minfo=mp nbody_start.f90 -o build/main_gpu_tiled
	nvfortran -cpp -DTEST $(FLAGS) -DTILED -mp=gpu -Minfo=mp nbody_start.f90 -o build/test_gpu_tiled

solutions: nbody.f90 | build
	gfortran -cpp -DMAIN $(FLAGS) -O2 nbody.f90 -o build/solution_cpu
	nvfortran -cpp -DMAIN $(FLAGS) -mp=gpu -Minfo=mp nbody.f90 -o build/solution_gpu
	nvfortran -cpp -DMAIN $(FLAGS) -DTILED -mp=gpu -Minfo=mp nbody.f90 -o build/solution_gpu_tiled

build:
	mkdir build

clean:
	rm -f *.mod *.csv
	rm -rf build

.PHONY: all tiled clean
