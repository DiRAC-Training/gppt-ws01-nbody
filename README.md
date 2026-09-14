# GPPT porting & profiling exercises

## Porting exercises (Day 1)

- [C++ CUDA Porting](https://github.com/DiRAC-Training/gppt-ws01-nbody/tree/main/cuda_01_introduction)
- [Fortran OpenMP Porting](https://github.com/DiRAC-Training/gppt-ws01-nbody/tree/main/fortran_01_02)

## Profiling & optimisation exercises (Day 2)

- [Fortran OpenMP](https://github.com/DiRAC-Training/gppt-ws01-nbody/blob/main/fortran_01_02/02_profiling_and_optimisation.md)
- [C++ CUDA](https://github.com/DiRAC-Training/gppt-ws01-nbody/tree/main/cuda_02_optimisation)



# Visualising output

You may find it useful to visualise the output of the simulation. We have provided a Python visualisation script in `../visualisation/plot.py` that can be used to render CSV outputs to PNG images:

```bash
../visualisation/plot.py *.csv
```

If the Python packages are not already available, see the README.md in the visualisation folder for instructions on getting this script running.

There is also a video plotting tool that plots all CSVs in the current folder and outputs an MP4 animation of the simulation:

```bash
../visualisation/plot_video.sh
```

## WARNING: Timing can be tricky

Do not rely **solely** on the wall-clock number the program prints on its own to
decide whether a change helped. You are on a shared node, so your runtime will be affected by other users on the system.

