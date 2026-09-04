# CUDA N-Body — Optimisation Exercise

You are given a working CUDA version of a direct-summation n-body simulation
found in a single source file, `nbody.cu`. It is correct but unoptimised and
includes some deliberate performance issues that you will have to find and fix.

Your job is to find the bottlenecks using a profiler and fix them. You will
make all of your changes in `nbody.cu`.

There are two tasks, split by a break. Task 1 is about measurement and the wins
you can get without restructuring the hot loop. Task 2 restructures it.

## What you'll learn

By the end of this exercise you should be able to:

- Identify data transfers and large kernels with Nsight Systems
- Use Nsight Compute to explore how resource limits affect kernel performance
- Tune a kernel's block size using profiling data
- Implement a shared-memory tiling optimisation

## Before you start

### Build and run

```sh
make && ./nbody
```

The unit tests run automatically at startup and the program aborts if any fail.
Leave them on — they are a cheap first check on every change. However, keep in
mind that they only run on two particles, so they will not be able to catch
everything you can break in Task 2.

To run only the unit tests:

```sh
make && ./nbody --only_unit_tests
```

In order to check correctness, you will be comparing any new output to the original output `final.csv`.

**Rename this to save it:**

```bash
make && ./nbody && mv final.csv final_og.csv
```

### Visualising output

You may find it useful to visualise the output of the simulation. We have provided a Python visualisation script in `../visualisation/plot.py` that can be used to render CSV outputs to PNG images:

```bash
../visualisation/plot.py *.csv
```

If the Python packages are not already available, see the README.md in the visualisation folder for instructions on getting this script running.

There is also a video plotting tool that plots all CSVs in the current folder and outputs an MP4 animation of the simulation:

```bash
../visualisation/plot_video.sh
```

### WARNING: Timing can be tricky

Do **not** rely solely on the wall-clock number the program prints on its own to
decide whether a change helped. You are on a shared node, so your runtime will be affected by other users on the system.

## **Task 1a** - Identify a bottleneck with `nsys`

**Try to identify potential bottlenecks in the main loop in `nbody.cu` before running the profiler.** Just from reading the code, can you get a sense of which kernels may dominate? Can you identify data transfers and synchronisation points that may need investigating?

**Profile the code with `nsys`**:

```sh
nsys profile -o report ./nbody
nsys stats report.nsys-rep
```

**Inspect the output to help answer the questions:**

1. Which kernel takes the most time? Take note of this for later tasks.
2. Where in the main loop is there an obvious bottleneck?

The bottleneck is most clearly seen in the timeline view of the Nsight Systems UI. See the [guidance document](guidance_on_nsys_ui.md) for more information on using this UI with CSD3. You can still identify it with just the text output from `nsys stats`.

<details>
<summary>Hint</summary>

Look closer at the memory transfers. In the timeline view you should see a large block of async memory transfer happening. In `nsys stats` note how long the memory transfers take compared to the kernel runtime.

</details>

<details>
<summary>Hint</summary>

Read the code in the main loop. You should be able to identify lines that are transferring data.

</details>

<details>
<summary>Hint</summary>

The variables being transferred are `pos` and `pos_prev` (and their GPU partners `pos_d` and `pos_prev_d`. Within the main loop, where are these variables being used? Why are they being transferred?

</details>

<details>
<summary>Solution</summary>

In the main loop, there are the following data transfers:

```cpp
thrust::copy(pos.begin(), pos.end(), pos_d.begin());
thrust::copy(pos_prev.begin(), pos_prev.end(), pos_prev_d.begin());

...

thrust::copy(pos_d.begin(), pos_d.end(), pos.begin());
thrust::copy(pos_prev_d.begin(), pos_prev_d.end(), pos_prev.begin());
```

These are the only transfers occurring within the main loop and are the main bottleneck visible with Nsight Systems.

Consider: **Why might these lines be accidentally left in during porting?**

</details>

## Task 1b - Fix the bottleneck

**Remove or move appropriate data transfers to speed up the code without introducing a bug.**

Remember to build and profile regularly. You can use `nsys stats` to quickly check relevant metrics.

**Check correctness:**

Since we are only changing data transfers in this task, the output should be *identical*, so we can check overall correctness with:

```bash
diff final.csv final_og.csv
```

If the diff shows any different, you may find it useful to look at the plot of the data:

```sh
../visualisation/plot.py final.csv
```

<details>
<summary>Hint</summary>

`pos` must be transferred before being dumped in:

```cpp
if (t >= next_dump) {
  dump_to_file(format_fname(dump_counter), pos);
  dump_counter += 1;
  next_dump += t_between_dump;
}
```

But in the current version of the code this happens every single timestep. Make sure the transfer happens if and when it's actually needed.

</details>

<details>
<summary>Solution</summary>

You should have now removed all data transfers from the main loop with the exception of the device to host transfer that should now happen inside the dump check:

```cpp
if (t >= next_dump) {
  thrust::copy(pos_d.begin(), pos_d.end(), pos.begin());
  dump_to_file(format_fname(dump_counter), pos);
  dump_counter += 1;
  next_dump += t_between_dump;
}
```

Your profile should now reveal that the amount of time spent on data transfers is small compared to the actual runtime of the kernels.

</details>

## Task 1c - Tuning parameters with Nsight Compute

In this task you'll be mainly exploring how some Nsight Compute metrics change in response to block size. You'll use *Duration* to pick the most optimal block size for the hardware.

You should have already found the dominant kernel: `calc_acc` so let's focus on that.

**Look at the code for `calc_acc`. Before profiling, do you think it will be compute-bound or memory-bound? Why?**

**Now profile `calc_acc` with `ncu`.**

To measure a kernel directly with Nsight Compute, use:

```sh
ncu --section SpeedOfLight --section Occupancy --section WarpStateStats \
    --section LaunchStats --launch-skip 1 --launch-count 1 -k <kernel_name> \
    ./nbody --quiet --disable_unit_tests
```

- `--launch-skip 1 --launch-count 1` profiles a single launch of the selected
  kernel, skipping the first one. A kernel's first launch pays one-off costs such
  as module loading and cold caches, so it is not representative of the rest.
  Without this you profile every launch, which may take a very long time.
- Do **not** use `--set full`. It will take too long.
- `-k <kernel_name>` selects the kernel by name. Change it to profile individual
  kernels as needed.
- `--quiet` to limit the simulation output.
- **IMPORTANT**: `--disable_unit_tests` to ensure the actual kernel launch is profiled, not the one called in the unit tests.

The number to note is *Duration* in the Speed Of Light section. One run should be
enough to get an accurate enough measurement but feel free to check this by changing `--launch-count`.

---

The output of `ncu` can be overwhelming. Focus on the Speed Of Light section for `calc_acc`. Compare
Compute (SM) Throughput with DRAM Throughput.

**Is this kernel memory bound or compute bound?**

---

Find `block_size` in `main()`.

**From your understanding of GPU architecture so far; do you think this value is optimal? Why?**

---

**Vary `block_size` without the profiler.**

We have exposed the block size with the flag `--bs` for easy access. Try running a sweep across some block sizes *without the profiler*:

```bash
for bs in 32 64 128 256 512 1024; do
  ./nbody --quiet --bs $bs --disable_unit_tests -n 100000
done
```

Note that we have increased the number of particles so that we *saturate* the GPU, i.e. we have many more particles than active threads. This is related to the concept of *occupancy*.

What do you notice about the results?

---

**Vary `block_size` with the profiler.**

For example:

```sh
for bs in 32 64 128 256 512 1024; do
    ncu --section SpeedOfLight --section Occupancy --section WarpStateStats \
    --section LaunchStats --launch-skip 1 --launch-count 1 -k calc_acc \
    ./nbody --quiet --bs $bs --disable_unit_tests -n 100000 >> profile_bs_$bs.txt
done
```

The outputted `profile_bs_*.txt` will contain the profile outputs.

**Compare metrics across the different runs, focusing particularly on the smallest, 32, largest, 1024, and your previously identified optimum.**

What changes in the occupancy section as the block size varies? Why?

How do you see some of the recommendations changing with block size?

## Task 1.5: Reflection

At this point you've used `nsys` to find the dominant kernel, `calc_acc`, and diagnose a performance bottleneck. You've tuned block size for overall time and explored how the block size can change various metrics in `ncu`.

Profiling is one of the key pillars of GPU programming. If you get used to using these tools often, you will quickly optimise your codes.

**How do you think you will use these tools in your own work?**

**Do you already use these tools in a different way?**

## Task 2: Tiling optimisation

Recall that shared memory is the GPU's programmable cache and, by default, data loaded from DRAM isn't reused. In `calc_acc` we can identify the potential for data reuse in the following lines:

```cpp
  for (int j = 0; j < N; ++j) {
    const Vec2 accll = calc_acc_pair(pi, pos[j], mass[j], eps);
    accl.x += accll.x;
    accl.y += accll.y;
  }
```

The idea of tiling: in `calc_acc`, every thread requests the entire `pos` and `mass`
arrays. Threads in a block request the same data over and over again. Instead,
we can have the block cooperatively load a **tile** of particle data into
shared memory, have every thread in the block consume that tile, then move to
the next one. This makes the reuse explicit.

In this task you will implement a tiled version of this kernel, `calc_acc_tiled`, using shared memory, then find the best tile size by
measurement.

After completing this you should be able to explain how tile size affects the
kernel, whether its best value matches the best block size you found in Task 1,
and how the comparison changes as the problem grows beyond one "wave per SM".

---

**Consider the overall algorithm:**

```
for each tile:
    cooperatively load this tile's positions and masses into shared memory
    <barrier>
    for each particle in the tile:
        accumulate acceleration from it
    <barrier>
```

Each thread still owns one particle `i` and accumulates into one `Vec2`. What
changes is the `j` loop, which becomes two nested loops:

The accumulator lives across the whole outer loop, so it has to be declared
before it and written back after it.

One key aspect to this algorithm is that it requires many threads to synchronise. In simpler algorithms it's easy to ignore the fact that the kernel runs in parallel. But synchronisation requires us to start thinking in terms of blocks or warps of threads executing at the same time.

**Keep in mind that each kernel really runs many copies of itself and some of those copies can communicate through synchronisation.**

---

In `nbody.cu` there is a version of `calc_acc_tiled` with some of the code replaced with `???`. There are related TODO comments near each `???`. We will address these in turn:

**TODO 1: Declaring a shared array**

Shared memory is (usually) declared *statically* so we have to specify a compile-time size. 

Here, we have decided to make the block size and the tile size identical and use  `block_size` as the tile size.

```cpp
__shared__ Vec2 shPosition[block_size];
```

Oh no! Our block_size is defined at runtime!

**Remove the runtime variable `block_size` from `main` and instead declare this at the top as a `const int block_size = ...`.**

**Fill in the second shared array that should hold masses.**

**TODO 2: Calculating the thread index**

It's the same as every time before. You got this.

**Solution**

```cpp
const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
```

**TODO 3 & 4: Global and local indices**

Now we're inside the loop over tiles and we need to populate the shared arrays with data from the global arrays. `idx` is a global index into the `pos` and `mass` arrays while the index in the brackets of `shPosition[???]` is a local index into the shared memory.

**What should the value of both these indices be?**

**Hint**

It's useful to consider this from the perspective of one block of threads all running in parallel. Each thread inside the block has a unique index assigned to it in `threadIdx.x`. Since we chose to make the tile size and block size equal, we can also use this value to index into the shared memory space. So the local index should be:

```cpp
shMass[threadIdx.x] = (idx < N) ? mass[idx] : real(0.0);
```

**Hint**

The point of this algorithm is to process the data one tile at a time, in chunks of size `block_size`. So the first time through the outer tile loop, `tile_start == 0`, and we copy all values from 0 to `block_size` into the shared memory:

|0|1|2|...|31| SHARED
 ^ ^ ^        
|0|1|2|...|31| GLOBAL

The second time through the loop, `tile_start == 1 * block_size`, and we copy from `1*block_size` to `2*block_size`:

|0 |1 |2 |...|31| SHARED
 ^ ^ ^ ^        
|32|33|34|...|63| GLOBAL

So the global index into `pos` and `mass` must combine the offset into the global array given by `tile_start` with the index of the thread, `threadIdx.x`:

```cpp
const int idx = tile_start + threadIdx.x;
```

**TODO 5: What variables do we access?**

**Fill in the appropriate arguments in the call to `calc_acc_pair`.**

**Hint**

The first argument is actually the same as before. Each instance of the kernel still owns the calculation for the i-th particle, so the call should look like:

```cpp
calc_acc_pair(pi, TODO, TODO, eps)
```

**Hint**

Previously the `calc_acc_pair` call looked like this:

```cpp
calc_acc_pair(pi, pos[j], mass[j], eps)
```

But now, we've saved a tile of `pos[j]` and `mass[j]` to shared memory. Access the equivalent values in shared memory.

**Solution**

```cpp
calc_acc_pair(pi, shPosition[j], shMass[j], eps);
```

**TODO 6: When to sync?**

Consider this algorithm without synchronisation and remember this kernel is run by a block of threads in parallel. If some of the threads finish saving their pieces of the tile and move on to processing the tile, they may access parts of the tile that haven't been populated by slower threads. So we need a method of synchronising at particular points. We do this with the builtin function `__syncthreads();`, but where should we synchronise?

**Consider the three suggested synchronisation points. Which should be implemented?**

**Hint**

We've already described a situation where faster threads could access unpopulated data, so we need a sync after the copy into shared memory:

```cpp
const int idx = tile_start + threadIdx.x;
shPosition[threadIdx.x] = (idx < N) ? pos[idx] : Vec2{0.0, 0.0};
shMass[threadIdx.x] = (idx < N) ? mass[idx] : real(0.0);
// Sync before we use the data
__syncthreads();
```

**Hint**

We also need a synchronisation point before we start copying a new tile. **Why?**

**Solution**

Again, some threads will process faster than others, so if a fast thread were finished with processing, it would continue on into the next iteration of the tile loop and start copying the next tile's data into shared memory, *overwriting the existing data* and potentially affecting any slow threads that are still processing that tile. So we need a sync point *either* before the tile copy operation, or after processing, as in the solution:

```cpp
for (int j = 0; j < THREADS_PER_BLOCK; j++) {
  // Add the acceleration from jth particle to this
  const Vec2 accll = calc_acc_pair(pi, shPosition[j], shMass[j], eps);
  accl.x += accll.x;
  accl.y += accll.y;
}
__syncthreads();
```

**TODO 7: Where to store the result?**

**Where in the global array should this thread's `accl` be stored?**

**Solution**

The tiling part of this algorithm only affects the way in which the `j` particles are processed. We've kept the convention of assigning one `i` particle per thread, so just as we load the position as before, `pos[gtid]` we can store `accl` in the same place as before:

```cpp
if (gtid < N)
  acc[gtid] = accl;
```

**TODO 8: What's with the `?:` operators?**

You'll have noticed the use of operators like `(idx < N) ? pos[idx] : Vec2{0.0, 0.0}`. **Can you think why we've done this?**

**Solution**

We guard with ternary as threads trying to access `i >= N` cannot return early - they must still reach every `__syncthreads()` and help load tiles for the rest of the block. Alternatively, using padding avoids this branching.

---

Phew! Let's now test this version with the unit tests:

**Update the calls to `calc_acc` in `test_calc_acc_x` and `test_calc_acc_y` to use the new tiled version.**

---

### What to do

2. Switch the main loop to launch it instead of `calc_acc`. Keep `calc_acc`
   around — you want to compare.
3. Update the launch in `test_calc_acc_x` and `test_calc_acc_y` too. Those tests
   launch the kernel directly, so until you change them your tiled kernel is not
   being tested at all.
4. Sweep the tile size: 32, 64, 128, 256. Measure each with `ncu`.
5. Compare your best tiled kernel with your best Task 1 kernel using `ncu`
   Duration and supporting metrics. Do not base the comparison on the program's
   printed timer alone.
6. Increase `N_PARTICLES` until the Launch Statistics section reports more than
   one wave per SM. Profile both kernels again at the same larger `N`, and
   compare how their efficiency changes.

<details>
<summary>Hint 1 — the shape of the kernel</summary>

</details>

<details>
<summary>Hint 2 — declaring the shared arrays</summary>

You need something like:

```cuda
__shared__ Vec2 pos_s[TILE];
__shared__ real mass_s[TILE];
```

Note what that one constant now couples together: the shared array size, the
launch block size, the number of tiles, and the inner loop bound. Tile size *is*
block size here, which is why the sweep in step 4 changes both at once.

</details>

<details>
<summary>Hint 3 — where the two barriers go, and why both are needed</summary>

- After the loads, before the inner loop — so nobody reads a slot before its
  owner has written it.
- After the inner loop, before the next tile's loads — so nobody overwrites a
  slot another thread has not finished reading.

Dropping the second one is the classic bug. It usually still produces
plausible-looking output, which is why you need to compare against your Task 1
results rather than trusting that the simulation still looks right.

</details>

<details>
<summary>Hint 4 — why threads past the end cannot simply return</summary>

`__syncthreads()` has to be reached by *every* thread in the block. If the
out-of-range threads take `calc_acc`'s early `return`, the rest of the block
waits at a barrier those threads will never arrive at. Depending on the launch
configuration this may appear to work, hang, or quietly give wrong answers — all
worse than a clean failure.

So every thread runs the whole loop structure, and you guard the *values* rather
than the control flow. A conditional expression keeps every thread on the same
path through the barriers; an `if` that skips one does not.

That leaves the question of what an unused slot should contain. Think about what
makes a particle contribute *exactly nothing* to the sum — get that right and the
inner loop needs no condition at all. Look at what `calc_acc_pair` multiplies by.

</details>

<details>
<summary>Hint 5 — tuning the tile size</summary>

Sweep it rather than assuming the answer is the same as Task 1's best block
size. Changing the tile size changes several things at once: the shared-memory
footprint, the number of active blocks and warps, the number of tiles, and the
cost of each barrier.

Use Duration to decide which version is faster. Use occupancy, memory throughput
and warp-state metrics to explain the result. An individual metric improving is
not sufficient evidence that the kernel became faster.

</details>

<details>
<summary>Hint 6 — scaling beyond one wave</summary>

Look for **Waves Per SM** in the Launch Statistics section. Increase
`N_PARTICLES` until it is greater than one, then profile the naive and tiled
kernels at exactly the same `N` and with their respective tuned launch
configurations.

The direct-summation calculation performs O(N^2) pair interactions, so raw
Duration is not enough when comparing different particle counts. Also compare
`Duration / N^2`, or calculate pair interactions per second. Use `ncu` for the
kernel measurement; the program's printed timer is only a whole-application
sanity check.

</details>

Each value of `N` is a different set of particles, so do not compare
trajectories between them in this experiment. The purpose here is to study
computational scaling.

### Checking you are right

- [ ] **Diff against Task 1 — this is your safety net.** Your tiled kernel computes
  the same quantity as your Task 1 kernel, and if you followed the steps above it
  also sums in the same order: working tile by tile still visits the source
  particles `0, 1, 2, ... N-1`, and padded slots contribute exactly zero. Dump
  `0000.csv` from each and compare — to the precision written out they should
  agree exactly. A visible difference means a barrier or an indexing bug, not
  rounding.
- [ ] **Run the unit tests too, but do not stop there.** They use two particles, so
  the whole calculation fits in a single tile. That does exercise your padding —
  at a tile size of 32, thirty of the thirty-two slots are unused — but with only
  one tile there is never a second pass over the shared arrays, so a missing
  barrier costs nothing and the tests still pass. Passing them means you have not
  broken the obvious things; it does not mean your tiling is correct.
- [ ] **Test the tail specifically.** Most tiling bugs live in the last, partly-full
  tile. Run with an `N_PARTICLES` that is *not* a multiple of your tile size. If
  it works at N = 1024 with tile 128 but not at N = 1000, your padding is wrong.
- [ ] **Check the shared memory is actually being used.** In the Launch Statistics
  section, Static Shared Memory Per Block should be non-zero and consistent with
  your tile size × (`sizeof(Vec2)` + `sizeof(real)`). If it is zero, you are not
  running the kernel you think you are.
- [ ] **Sanity-check the memory metrics.** Compare the relevant cache and shared-
  memory metrics with Task 1 and check that they are consistent with source
  particles being loaded cooperatively. If they are unchanged, confirm that you
  profiled `calc_acc_tiled` rather than `calc_acc`.
- [ ] **Check scaling with `ncu`.** At each particle count compare the two kernels at
  the same `N`, then normalize by N^2 when comparing efficiency across particle
  counts. Keep the printed timer separate from the kernel-level evidence.

### Before you move on

In your own words, without looking back at the Goal section: how does tile
size affect the kernel, does its best value match the best block size you
found in Task 1, and how does that comparison change once the problem grows
beyond one wave per SM?

---

## If you finish early

Both of these are standard published n-body techniques (see references) that can
be explored on top of a working tiled kernel:

- **Pack position and mass together.** `Vec2` + `real` is currently two separate
  loads from two arrays. A single 16-byte structure would be one wider load.
  What does that do to the shared memory traffic and the instruction count?
- **Thread coarsening.** Give each thread 2 or 4 particles instead of one. The
  tile is then loaded once and reused several times, and the loads amortise
  further. Watch what it does to Registers Per Thread, and whether that starts
  limiting occupancy.

### References

- Nyland, Harris and Prins,
  [*Fast N-Body Simulation with CUDA*](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda),
  GPU Gems 3 chapter 31 (NVIDIA, 2007).
- Volkov,
  [*Better Performance at Lower Occupancy*](https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf),
  GTC 2010.
