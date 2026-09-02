# CUDA N-Body — Optimisation Exercise

You are given a working CUDA version of the direct-summation n-body simulation
found in a single source file, `nbody.cu`. It is correct, produces the right
trajectories, and it is unoptimised with some naive errors that a new GPU
developer might make.

Your job is to find out **why** it is unoptimised and improve it — using the
profiler to decide what to change, rather than guessing. You will make all of
your changes in `nbody.cu`.

There are two tasks, split by a break. Task 1 is about measurement and the wins
you can get without restructuring the hot loop. Task 2 restructures it.

---

## The code

The simulation is a 2D direct-summation n-body: every particle feels the
gravitational pull of every other particle, so each step is O(N^2).

| Function         | Runs on    | What it does                                |
|------------------|------------|---------------------------------------------|
| `calc_acc_pair`  | device     | Acceleration on one particle from one other |
| `calc_acc`       | device     | Acceleration on one particle from the rest  |
| `calc_acc_tiled` | device     | Empty. You fill this in during Task 2       |
| `advance_pos`    | device     | Velocity-Verlet position update             |

Time integration is velocity-Verlet, which requires information about the
previous position from `pos_prev`. The new position is written over the old one
as soon as it has been read, then the two buffers are swapped after the update.

`epsilon` is a softening length that stops the force blowing up when two
particles are nearly coincident.

### Build and run

```sh
make
./nbody
```

The unit tests run automatically at startup and the program aborts if any fail.
Leave them on — they are a cheap first check on every change. However, keep in
mind that they only run on two particles, so they will not be able to catch
everything you can break in Task 2.

---

## Measuring: read this before you change anything

Do **not** rely solely on the wall-clock number the program prints on its own to
decide whether a change helped. You are on a shared node, so that number
includes contention and work outside the kernel you are trying to improve.

To gain meaningful measures of the code while completing the tasks, you will
need both of the following.

To measure a kernel directly with Nsight Compute, use:

```sh
ncu --section SpeedOfLight --section Occupancy --section WarpStateStats \
    --section LaunchStats --launch-skip 1 --launch-count 1 -k <kernel_name> \
    ./nbody
```

- `--launch-skip 1 --launch-count 1` profiles a single launch of the selected
  kernel, skipping the first one. A kernel's first launch pays one-off costs such
  as module loading and cold caches, so it is not representative of the rest.
  Without this you profile every launch, which may take a very long time.
- Do **not** use `--set full`. It will take too long.
- `-k <kernel_name>` selects the kernel by name. Change it to profile individual
  kernels as needed.

The number to use is **Duration**, in the Speed Of Light section. One run is
enough — repeated `ncu` measurements of the same kernel agree to a fraction of a
percent, so there is no need to average several, at least for the purpose of
this exercise.

For a view of the whole timeline — which kernels ran and how long each took, plus
everything that is *not* inside a kernel (memory transfers, launch overhead etc.)
— use Nsight Systems:

## Task 1a - Identify a bottleneck with `nsys`

Start by profiling the code with `nsys`:

```sh
nsys profile -o report ./nbody
nsys stats report.nsys-rep
```

**Inspect the output to help answer the questions:**

1. What kernel take the most time? Take note of this for later tasks.
2. Where in the main loop is there an obvious bottleneck?

The bottleneck is most clearly seen in the timeline view of the Nsight Systems UI. See the guidance document for more information on using this UI with CSD3. You can still identify it with just the text output from `nsys stats`.

Feel free to peek at the hints below if you get stuck.

**Hint 1**

Look closer at the memory transfers. In the timeline view you should see a large block of async memory transfer happening. In `nsys stats` note how long the memory transfers take compared to the kernel runtime.

**Hint 2**

Read the code in the main loop. You should be able to identify lines that are transferring data.

**Solution**

In the main loop, there are the following data transfers:

```cpp
thrust::copy(pos.begin(), pos.end(), pos_d.begin());
thrust::copy(pos_prev.begin(), pos_prev.end(), pos_prev_d.begin());

...

thrust::copy(pos_d.begin(), pos_d.end(), pos.begin());
thrust::copy(pos_prev_d.begin(), pos_prev_d.end(), pos_prev.begin());
```

These are the only transfers occurring within the main loop and are the main bottleneck visible with Nsight Systems.

## Task 1b - Fix the bottleneck

**Remove or move appropriate data transfers to speed up the code without introducing a bug.**

Remember to:

1. Profile regularly.
2. Compare `final.csv` to a previously saved `final.csv`.

**Hint 1**

The variables being transferred are `pos` and `pos_prev` (and their GPU partners `pos_d` and `pos_prev_d`. Within the main loop, where are these variables being used? Why are they being transferred?

**Hint 2**

`pos_prev` is in fact never used within the main loop, so transferring it doesn't need to happen at all! But what about `pos`...?

**Hint 3**

`pos` must be transferred before being dumped in:

```cpp
if (t >= next_dump) {
  dump_to_file(format_fname(dump_counter), pos);
  dump_counter += 1;
  next_dump += t_between_dump;
}
```

But in the current version of the code this happens every single timestep. Make sure the transfer happens if and when it's actually needed.

**Solution**

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

## Task 1c - Tuning parameters with Nsight Compute

**Hint 2 — identify the limiting resource**

Start with the Speed Of Light section for the kernel you identified. Compare
Compute (SM) Throughput with DRAM Throughput rather than assuming that an n-body
kernel must be limited by memory traffic. Use the larger value to decide which
resource is worth investigating, then use the remaining sections to test that
explanation.

This also tells you what each fix can possibly buy you. If the kernel is not
waiting on memory, then tidying up the host/device transfers will not shorten
it, however much it cleans up the timeline — check that against what you just
measured.

**Hint 3 — the launch configuration**

Find `block_size` in `main()`. Ask whether the value chosen is optimal.

Try a sweep: 32, 64, 128, 256, 512, 1024. Measure each with `ncu`. Note that
1024 threads is the hardware maximum block size — you cannot go higher.

Things to look at while you sweep:

- Theoretical Occupancy and Achieved Occupancy, in the Occupancy section.
- Duration and Compute (SM) Throughput in the Speed Of Light section.

Across the sweep, work out which of these actually tracks Duration. Occupancy can
saturate — once Theoretical Occupancy reaches 100% it cannot report any further
improvement, even while Duration keeps changing. Whichever metric follows
Duration across the whole sweep is the one worth trusting when you judge later
changes.

**Hint 4 — about the profiler's advice**

Alongside the metrics, `ncu` prints rule-based suggestions — indented paragraphs
tagged `OPT`, each with an `Est. Speedup` figure. Read them, but treat them as
hypotheses to test, not instructions. The estimates describe what might happen
if one reported limit were removed; they do not guarantee that the suggested
change will improve this kernel.

If an `OPT` block tells you to try something, change one factor and measure it.
Base your conclusion on the resulting Duration and supporting metrics, not the
estimate alone.

### Checking you are right

You have no external reference timings, so verify by internal consistency:

- **Correctness first.** The unit tests must still pass. Beyond that, keep a copy
  of `0000.csv` from the original build and compare against it after each change.
  Nothing you do in Task 1 changes the arithmetic or the order it is summed in,
  so the dumps should be *identical* — an empty `diff` is the pass condition, and
  any output at all means something is likely wrong.

  <!-- When `diff` is not empty it is poor at telling you how bad the damage is when -->
  <!-- thousands of lines differing in the last decimal look much the same as a -->
  <!-- handful of particles in completely the wrong place. It might help to plot the -->
  <!-- dumps to see which it is: -->

  <!-- ```sh -->
  <!-- ../visualisation/plot.py 0000.csv -->
  <!-- ``` -->

  <!-- This writes `0000.csv.png`. Compare the images from before and after your -->
  <!-- change — particles that have been dropped or flung to the wrong place are -->
  <!-- obvious in a plot and easy to miss in a diff. -->
- **Anything you change between the kernel launches** should change almost
  nothing in `ncu` Duration — it is not in the kernel — but should visibly
  change the `nsys` timeline (GUI) and the memory report (CLI). If your `ncu`
  number improved a lot from a change of that kind alone, be suspicious.
- **Anything you change in the launch configuration** should still ensure Grid
  Size x Block Size covers `N_PARTICLES`, and that Threads in the Launch
  Statistics section is still ≥ `N_PARTICLES`. If the trajectories changed, you
  probably dropped particles.
- If a change made things *worse*, keep the measurement. Knowing which
  plausible-sounding optimisations do not work *is* the point.

---

## Break

We will review the profiling approach and discuss the measurements before
introducing tiling. Task 2 assumes that you are starting from a correct Task 1
implementation, so check your solution against the one shown and ask for help if
it is not and you are unsure how to fix it.

---

## Task 2 — Tiling

Continue from your completed Task 1 implementation. Before proceeding, confirm
that `nbody.cu` builds successfully and passes its unit tests.

### Goal

Implement `calc_acc_tiled` using shared memory, then find the best tile size by
measurement.

The idea: in `calc_acc`, every thread requests the entire `pos` and `mass`
arrays. Threads in a block request the same source particles over and over
again, although the cache may satisfy some of those repeated accesses. Instead,
have the block cooperatively load a **tile** of particles into shared memory,
have every thread in the block consume that tile, then move to the next one.
This **makes the reuse explicit**: the loads are shared; the arithmetic is not.

You should also come out of this able to explain how tile size affects the
kernel, whether its best value matches the best block size you found in Task 1,
and how the comparison changes as the problem grows beyond one "wave per SM".

### What to do

1. Implement `calc_acc_tiled`, in the stages below. The `TODO` comments in the
   stub mark where each one goes. Rebuild and run the unit tests as you go
   rather than writing the whole kernel first. Hint 1 sketches the shape all of
   these are building towards, if you would rather see it whole first.
   1. **Promote the block size to a compile-time constant.** Move `block_size`
      out of `main()` and make it a `const int` at file scope, then update the
      launch configuration in `main()` and in both `calc_acc` tests to use it.
      This is a requirement for using shared memory because a `__shared__` array
      needs a size known at compile time, and a variable local to `main()` is
      not visible inside a kernel at all.
   2. **Declare the shared arrays.** Two of them, sized by the block size
      constant: one of `Vec2` for positions, one of `real` for masses. Look back
      at the shared memory example from the performance session for the syntax,
      or Hint 2 if you are stuck.
   3. **Set up the per-thread state.** The same three lines `calc_acc` opens
      with: the global thread index, this thread's own particle read from `pos`,
      and a `Vec2` accumulator zeroed. Leave out its `if (gtid >= N) return;` —
      step 8 deals with threads past the end, and returning here would break the
      barriers you add in step 6.
   4. **Write the outer loop over tiles.** It advances a tile start from `0` to
      `N` in steps of the block size. Inside it, each thread loads exactly one
      particle into shared memory: it reads element `tile_start + threadIdx.x`
      from global memory and stores it into shared slot `threadIdx.x`.
   5. **Write the inner loop over the tile.** It runs from `0` to the block size,
      *not* to `N` — every thread walks through the full current tile,
      accumulating with `calc_acc_pair` exactly as before, but now reading the
      positions and masses from shared memory instead of global.
   6. **Add the barriers.** You need `__syncthreads()` twice per tile, not only
      once. Identify where they should go. If you are stuck, Hint 3 covers where
      and why both are needed.
   7. **Write the result back** to `acc`. This sits after every barrier, so an
      ordinary `if` on the same condition you left out in step 3 is all you need
      here.
   8. **Make the final tile correct** when `N` is not a whole number of tiles.
      Some threads will have no particle to load, but they must still reach every
      `__syncthreads()`, so guard the load with a conditional expression rather
      than a branch that skips it. There are several possible ways to handle the
      unused slots, but we recommend padding them with values that contribute
      nothing to the sum. If you are stuck, Hint 4 covers both.

      Note the default `N_PARTICLES` divides exactly by every tile size in the
      sweep, so this issue will not be encountered in runs until you change the
      number of particles — but the unit tests do exercise it, since two
      particles in a tile of 32 is a partly-filled tile.
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

**Hint 1 — the shape of the kernel**

Each thread still owns one particle `i` and accumulates into one `Vec2`. What
changes is the `j` loop, which becomes two nested loops:

```
for each tile:
    cooperatively load this tile's positions and masses into shared memory
    <barrier>
    for each particle in the tile:
        accumulate acceleration from it
    <barrier>
```

The accumulator lives across the whole outer loop, so it has to be declared
before it and written back after it.

**Hint 2 — declaring the shared arrays**

You need something like:

```cuda
__shared__ Vec2 pos_s[TILE];
__shared__ real mass_s[TILE];
```

Note what that one constant now couples together: the shared array size, the
launch block size, the number of tiles, and the inner loop bound. Tile size *is*
block size here, which is why the sweep in step 4 changes both at once.

**Hint 3 — where the two barriers go, and why both are needed**

- After the loads, before the inner loop — so nobody reads a slot before its
  owner has written it.
- After the inner loop, before the next tile's loads — so nobody overwrites a
  slot another thread has not finished reading.

Dropping the second one is the classic bug. It usually still produces
plausible-looking output, which is why you need to compare against your Task 1
results rather than trusting that the simulation still looks right.

**Hint 4 — why threads past the end cannot simply return**

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

**Hint 5 — tuning the tile size**

Sweep it rather than assuming the answer is the same as Task 1's best block
size. Changing the tile size changes several things at once: the shared-memory
footprint, the number of active blocks and warps, the number of tiles, and the
cost of each barrier.

Use Duration to decide which version is faster. Use occupancy, memory throughput
and warp-state metrics to explain the result. An individual metric improving is
not sufficient evidence that the kernel became faster.

**Hint 6 — scaling beyond one wave**

Look for **Waves Per SM** in the Launch Statistics section. Increase
`N_PARTICLES` until it is greater than one, then profile the naive and tiled
kernels at exactly the same `N` and with their respective tuned launch
configurations.

The direct-summation calculation performs O(N^2) pair interactions, so raw
Duration is not enough when comparing different particle counts. Also compare
`Duration / N^2`, or calculate pair interactions per second. Use `ncu` for the
kernel measurement; the program's printed timer is only a whole-application
sanity check.

Each value of `N` is a different set of particles, so do not compare
trajectories between them in this experiment. The purpose here is to study
computational scaling.

### Checking you are right

- **Diff against Task 1 — this is your safety net.** Your tiled kernel computes
  the same quantity as your Task 1 kernel, and if you followed the steps above it
  also sums in the same order: working tile by tile still visits the source
  particles `0, 1, 2, ... N-1`, and padded slots contribute exactly zero. Dump
  `0000.csv` from each and compare — to the precision written out they should
  agree exactly. A visible difference means a barrier or an indexing bug, not
  rounding.
- **Run the unit tests too, but do not stop there.** They use two particles, so
  the whole calculation fits in a single tile. That does exercise your padding —
  at a tile size of 32, thirty of the thirty-two slots are unused — but with only
  one tile there is never a second pass over the shared arrays, so a missing
  barrier costs nothing and the tests still pass. Passing them means you have not
  broken the obvious things; it does not mean your tiling is correct.
- **Test the tail specifically.** Most tiling bugs live in the last, partly-full
  tile. Run with an `N_PARTICLES` that is *not* a multiple of your tile size. If
  it works at N = 1024 with tile 128 but not at N = 1000, your padding is wrong.
- **Check the shared memory is actually being used.** In the Launch Statistics
  section, Static Shared Memory Per Block should be non-zero and consistent with
  your tile size × (`sizeof(Vec2)` + `sizeof(real)`). If it is zero, you are not
  running the kernel you think you are.
- **Sanity-check the memory metrics.** Compare the relevant cache and shared-
  memory metrics with Task 1 and check that they are consistent with source
  particles being loaded cooperatively. If they are unchanged, confirm that you
  profiled `calc_acc_tiled` rather than `calc_acc`.
- **Check scaling with `ncu`.** At each particle count compare the two kernels at
  the same `N`, then normalize by N^2 when comparing efficiency across particle
  counts. Keep the printed timer separate from the kernel-level evidence.

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
