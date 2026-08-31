# Fortran N-Body — OpenMP Target Offload Exercise

You are given a working, CPU-only n-body simulation in a single source file,
`nbody_start.f90`. It is correct and produces the right trajectories, but it
runs entirely on the host. Your job is to accelerate it on an Nvidia GPU
using OpenMP `target` offloading, **without changing the maths** — only add
directives.

You will make all of your changes in `nbody_start.f90`. `nbody.f90` in this
same directory is the finished solution; try not to look at it until you've
had a go.

---

## The code

The simulation is a 2D direct-summation n-body: every particle feels the
gravitational pull of every other particle, so the acceleration calculation is O(N^2).

| Subroutine / function       | What it does                                          |
|------------------------------|-------------------------------------------------------|
| `calc_acc`                   | Acceleration on every particle from every other one. **O(N^2), the hot loop, and your main target.** |
| `advance_pos`                 | Velocity-Verlet position update, O(N) per step.        |
| `run_sim`                     | Driver: builds the system, time-steps it, times it.    |
| `create_solar_system` / `generate_random_star_system` | Build initial conditions — CPU-only setup code, not part of this exercise. |
| `test_calc_acc`, `test_advance_pos`, `test_calc_stable_orbit` | Unit tests, run automatically by the `TEST` build. |

Time integration is velocity-Verlet, which needs the previous position from
`pos_prev`. `pos_temp` is scratch space used to shuffle `pos` into
`pos_prev` after each update. `epsilon` is a softening length that stops the
force blowing up when two particles are nearly coincident.

### Build and run

Two compilers matter here:

- **`gfortran`**, with no OpenMP flag, compiles `!$omp` lines as ordinary
  comments — this is how you get a pure CPU baseline, including from a file
  that already has offload directives in it.
- **`nvfortran`** with `-mp=gpu` is what actually offloads
  to the GPU. `-Minfo=mp` is useful; it prints, for every `target`
  region, whether it turned into a GPU kernel.

```sh
# CPU baseline (directives ignored)
gfortran -cpp -DMAIN -O2 nbody_start.f90 -o main_cpu
gfortran -cpp -DTEST -O2 nbody_start.f90 -o test_cpu

# GPU offload build, once you've added directives
nvfortran -cpp -DMAIN -mp=gpu -Minfo=mp nbody_start.f90 -o main_gpu
nvfortran -cpp -DTEST -mp=gpu -Minfo=mp nbody_start.f90 -o test_gpu
```

Run `./main_cpu` now and note the printed "Time to complete" — that's your
baseline. On the reference machine this exercise was built on (an RTX 3060
laptop GPU), the CPU baseline takes about **20 s** for 20,000 particles; a
fully offloaded version takes about **1.2 s**. Your numbers will differ —
what matters is the relative change as you complete each task, not matching
these exactly.

`./test_cpu` runs the unit tests and aborts if any fail. Keep them passing
throughout — they're a cheap, fast check after every change, though bear in
mind they only exercise 1–2 particles, so they can't catch every mistake
(see Task 3).

---

## Task 1 — Offload the compute kernels

### Goal

Get the two hot loops and the position update running on the GPU, using
`!$omp target teams distribute parallel do` on each one.

- `target` says "run this on the device".
- `teams distribute` spreads the loop's iterations across GPU thread teams.
- `parallel do` further splits each team's share across its threads.

Used together on a loop's outer index, this is the standard way to map a
parallelisable Fortran `do` loop onto the GPU with OpenMP.

### What to do

There are three loops to offload, each marked with a `TODO (Task 1x)`
comment in `nbody_start.f90`:

1. **Task 1a** — the loop in `calc_acc` that zeroes `acc` before
   accumulating into it.
2. **Task 1b** — the pairwise force loop in `calc_acc`. This is the O(N²)
   loop and does almost all of the work in the program. Put the directive on
   the **outer** `i` loop only; the inner `j` loop stays a plain sequential
   loop that each GPU thread runs on its own.
3. **Task 1c** — the position-update loop in `advance_pos`.

For each one: add `!$omp target teams distribute parallel do` on the line
directly above the loop, and `!$omp end target teams distribute parallel
do` directly below it (after the matching `end do`/`enddo`).

Build with `nvfortran -mp=gpu -Minfo=mp` after each change and check the
compiler confirms it generated a GPU kernel for that loop. Then re-run the
unit tests (`./test_gpu`) — they should still pass.

### Checking you are right

- `-Minfo=mp` should report a `Generating "nvkernel_..."` line for each of
  the three loops once all of Task 1 is done.
- Unit tests still pass.
- The program is very likely **not faster yet**, possibly slower than the
  CPU baseline. That's expected, not a bug — see Task 2.

---

## Task 2 — Stop shipping data across PCIe on every kernel launch

### Goal

With no data directives, every `target` region you added in Task 1 is its
own island: on entry, OpenMP copies whatever it reads onto the device; on
exit, it copies back whatever it wrote. `calc_acc` and `advance_pos` are
each called once per time step, so right now you're paying a host↔device
transfer for `pos`, `mass`, `acc` and `pos_prev` on **every single step**,
even though the arrays already hold the right values on the device from the
step before.

The fix is `!$omp target data`, which opens a region that keeps its mapped
arrays resident on the device for as long as the region is open, regardless
of how many `target` kernels run inside it.

### What to do

Two more `TODO` comments in `run_sim`:

1. **Task 2a** — the one-off call to `calc_acc` that computes the initial
   acceleration, before the main loop. Wrap it in a `target data` region.
   Think about what each array needs: `pos` and `mass` are read by the
   device and never written by it here, `acc` is written by the device and
   is what the host needs afterwards.
2. **Task 2b** — the main `do while` time-stepping loop. Wrap the whole loop
   in one `target data` region so `pos`, `mass`, `pos_prev`, `acc` and
   `pos_temp` all stay on the device for every step, and only move when
   they actually need to. For each array, work out which OpenMP map type
   fits how it's used across the *whole loop*, not just one call:
   - `map(to: ...)` — host has a value the device needs, device never
     writes it back.
   - `map(from: ...)` — device produces a value the host needs, host's
     initial value doesn't matter.
   - `map(tofrom: ...)` — both directions matter.
   - `map(alloc: ...)` — the device needs space, but neither side cares
     about the other's values (pure scratch).

   Ask, for each of `pos`, `mass`, `pos_prev`, `acc`, `pos_temp`: does the
   *host* ever need this array's final value after the loop, and does the
   *device* ever need a value the host set before the loop?

### Checking you are right

- Unit tests still pass.
- The GPU build should now be substantially faster than the CPU baseline —
  not just faster than your Task 1 result. If it isn't, you've probably
  mapped something `tofrom` that only needed `alloc`, or vice versa (a
  `tofrom`/`from` where an `alloc` would do just adds unneeded transfer; an
  `alloc` where the host actually needed the result gives you stale or
  garbage data on the host — which the next check will catch).
- Compare `trajectory.csv` from `./main_gpu` against a copy saved from
  `./main_cpu`. The physics hasn't changed, so the numbers should agree to
  the precision written out. If they don't, you likely mapped `pos` (or
  something it depends on) the wrong way and the host is reading a stale or
  uninitialised copy.

---

## Task 3 — Fix the unit test

### Goal

`test_calc_acc` calls `calc_acc` directly, not through `run_sim`, so it
never benefits from the `target data` region you added in Task 2 — it's
its own island, same as Task 1 was before Task 2.

### What to do

There's a `TODO (Task 3)` comment right above the first call to `calc_acc`
in `test_calc_acc`. Give it the same kind of `target data` region you used
for Task 2a.

Note the *second* call to `calc_acc` a few lines later, on the same arrays,
is deliberately left without one — it relies on `calc_acc`'s own `target`
regions doing implicit mapping on entry/exit, which is correct (if
inefficient) on its own. This mirrors the finished solution: not every
call needs an explicit data region, only the ones that matter for
performance. Recognising the difference is the point of this task.

### Checking you are right

- `./test_gpu` passes, including `test_calc_acc`.
- This task only affects a test that runs once at start-up — it should not
  change the timed portion of `./main_gpu` at all.

---

## Task 4 — Measure and confirm it's really running on the GPU

Passing tests and a lower printed time are good signs, but they don't prove
the work is on the GPU rather than, say, silently falling back to the host.
Do at least one of the following:

- Run `nvidia-smi` in another terminal while `./main_gpu` is running (use a
  larger particle count, e.g. edit `n_particle_range` in the `MAIN` program
  block to something like `[200000]`, so the run lasts long enough to
  observe). You should see GPU utilisation and the process listed.
- Set `OMP_TARGET_OFFLOAD=MANDATORY` before running. This tells the OpenMP
  runtime to abort with an error instead of silently running on the host if
  offload isn't actually happening for a `target` region — a good sanity
  check to leave on while you're developing.
- Compile with `-Minfo=mp` (as recommended throughout) and check every
  `target` region you added reports a generated GPU kernel, not a host
  fallback.

Once you're confident it's genuinely running on the GPU, compare the
printed "Time to complete" between `./main_cpu` and `./main_gpu` at a couple
of different particle counts (edit `n_particle_range` in the `MAIN` program
block). How does the speedup change as N grows? Since the hot loop is
O(N²), think about what that implies for how much of the total time is
spent in `calc_acc` versus everything else, at small vs. large N.

---

## Advanced Task (optional) — Tiling for GPU Shared-Memory Reuse

This task is a bigger structural change than Tasks 1-4, and assumes you've
completed those first. It's entirely optional — a good thing to reach for if
you finish early, or to come back to later.

### The concept

A GPU has two memory tiers that matter here. **Global memory** is large and
visible to every thread on the device, but slow to access relative to how
fast the GPU can compute. **Shared memory** is much faster, but tiny (tens
of KB), and private to one *thread block* — a group of threads the GPU
schedules together, which can synchronise with each other and share that
fast memory.

In `calc_acc`, every GPU thread owns one particle `i` and loops over all `n`
source particles `j`, reading `pos(j,:)` and `mass(j)` straight from global
memory every time. Threads in the same block work through that loop at
roughly the same pace, so at any given moment, many of them are asking
global memory for the *same* source particle at the *same* time — the same
value fetched from slow memory once per thread, instead of once per block.

**Tiling** makes the reuse explicit instead of hoping the cache absorbs it.
Split the `n` source particles into small chunks ("tiles"), sized to match
the block. For each tile: every thread in the block cooperates to copy
exactly one particle from global memory into the block's shared memory (so
the whole tile costs one coordinated read per particle, not one per thread
per particle), then every thread reads that tile back out of shared memory
as many times as it needs, before the block moves on to the next tile. In
pseudocode:

```
for each tile of source particles:
    every thread loads one particle into the block's shared tile arrays
    <barrier: wait for the whole tile to be loaded>
    every thread accumulates its own acceleration from the tile
    <barrier: wait for everyone to finish reading before it's overwritten>
```

### The OpenMP mechanism

OpenMP has no `__shared__` keyword the way CUDA does. To reach the same
hardware feature, you have to give up the convenient combined
`target teams distribute parallel do` directive you've used everywhere so
far, and separate it into two directives:

- `!$omp distribute` — spreads loop iterations across GPU thread **teams**.
  A team is OpenMP's name for what CUDA calls a thread block.
- `!$omp parallel`, nested inside the `distribute` loop's body — spreads
  work across the **threads within one team**.

Declare the tile-staging arrays as ordinary local arrays, and list them in
a `private()` clause on `distribute`. That clause is what asks for one
instance of the array *per team*, shared by every thread in that team —
`nvfortran` places an array like that in real GPU shared memory. You can
(and should) confirm this yourself: compile with `-Minfo=mp` and look for a
line like `Team private (..., pos_s, mass_s) located in CUDA shared
memory`. If you don't see "located in CUDA shared memory" there, the
arrays didn't end up where you think.

`!$omp barrier`, used inside the nested `parallel` region, is OpenMP's
version of CUDA's `__syncthreads()`: it forces every thread in the team to
reach that point before any of them continues. You need it twice per tile —
once after the load (nobody may read the tile until everyone's finished
writing it) and once after the accumulation (nobody may start the next
tile's load until everyone's finished reading this one).

### Starting point

`nbody_tiled_start.f90` is what you edit — it's a copy of the finished
Tasks 1-4 solution with an empty `calc_acc_tiled` stub. `nbody_tiled.f90`
in this same directory is the finished solution to this task; try not to
look at it until you've had a go, the same as `nbody.f90` for the main
exercise.

### What to do

`calc_acc_tiled` in `nbody_tiled_start.f90` has `TODO (Task 5a)` through
`TODO (Task 5h)` comments marking each piece, in order:

- **Task 5a** — declare `TILE` as a compile-time constant (try 128) and the
  two team-private tile arrays it sizes, plus the other locals you'll need.
- **Task 5b** — compute `n`, `epsilon` and `num_teams_needed` (how many
  teams of `TILE` particles each does it take to cover `n` particles?
  round up), then open the `teams`/`distribute`/`parallel` skeleton.
- **Task 5c** — get this thread's index within its team, and its global
  particle index `i` from that plus `team_id` and `TILE`. Zero its
  accumulator.
- **Task 5d** — the outer loop over tiles, with the cooperative load.
- **Task 5e** — the first barrier.
- **Task 5f** — the inner loop over the tile, accumulating from the shared
  arrays.
- **Task 5g** — the second barrier.
- **Task 5h** — write the result back to `acc`, and close the constructs.

Then two more steps, also marked with `TODO` comments:

- **Task 5i**, in `run_sim` — once `calc_acc_tiled` is implemented and its
  test passes, switch `run_sim`'s two `calc_acc` calls over to it.
- **Task 5j** — `test_calc_acc_tiled` currently calls `calc_acc`, so it
  isn't testing your new subroutine at all. Point its calls at
  `calc_acc_tiled`, then add it to the `#ifdef TEST` program block.

**Hint 1 — the self-interaction term.** Don't special-case `j == i`: it's
naturally visited once, inside whichever tile it falls in, with
`dx = dy = 0`, and contributes exactly zero — same as in `calc_acc`.

**Hint 2 — padding, not branching, for the last tile.** `n` won't usually
be a whole multiple of `TILE`, so the last tile is only partly full.
`!$omp barrier` requires *every* thread in the team to reach it. If a
thread with nothing to load skips its write with an early branch that also
jumps past the barrier, the threads that do reach the barrier wait for one
that never arrives. Instead, give every thread the same control-flow path
through both barriers, and make the unused slots harmless by padding them
with `mass = 0` — a particle with zero mass contributes nothing to the sum,
so no extra condition is needed in the inner loop.

**Hint 3 — why `TILE` has to be a compile-time constant.** It sizes the
shared arrays, and it also becomes `thread_limit` — the number of threads
launched per team. Both have to be known when the code is compiled, the
same reason CUDA's tiled kernel needs a `const int` block size rather than
a runtime variable.

### Checking you are right

- `-Minfo=mp` reports `calc_acc_tiled` generating a GPU kernel, **and** the
  `Team private (...) located in CUDA shared memory` line — confirming
  kernel generation alone isn't enough, since the private arrays could in
  principle end up spilled to slower memory instead.
- Unit tests, including your new `test_calc_acc_tiled`, pass. Remember this
  test only exercises 2 particles against a `TILE` of 128, so it runs
  entirely inside one partly-empty tile — it's a real check of the padding
  path from Task 5d, but it cannot catch a bug that only shows up across
  multiple tiles.
- **Diff a full trajectory, not just the unit test.** Run at a particle
  count well above `TILE` (e.g. 50,000) with `calc_acc` and with
  `calc_acc_tiled`, saving `trajectory.csv` from each, and diff them. A
  missing barrier or a padding bug is far more likely to show up here than
  in the 2-particle test. Don't expect a perfectly empty diff, though:
  tiling changes the *order* the pairwise forces are summed in (tile by
  tile, rather than straight through `j = 1..n`), and floating-point
  addition isn't associative, so a handful of particles may differ in the
  last written digit even in a correct implementation. On the reference
  build, 6 out of 50,000 particles differed by 1 in the last decimal place;
  a real bug tends to look very different from that — many particles
  wrong, or wrong by much more than the last digit.
- **Compare timing against plain `calc_acc`.** Don't assume tiling has to
  win — whether it does depends on whether the kernel is bound by
  arithmetic or by memory traffic, and that can change with things you
  wouldn't expect, like floating-point precision. On the reference RTX 3060
  laptop GPU, with this exercise's default single-precision build, tiling
  was a consistent **~30% faster** than plain `calc_acc` — 0.26 s vs 0.38 s
  at N = 50,000, and 3.8 s vs 5.6 s at N = 200,000. If you switch `wp` back
  to double precision (see the commented-out line near the top of the
  module), try the same comparison again: on the same GPU, tiling turned
  out to make close to *no* difference in double precision, because
  consumer GPUs like this one have much lower double-precision throughput,
  which shifts `calc_acc` from being memory-bound towards being bound by
  the `sqrt`/divide arithmetic itself — and tiling only helps with the
  memory side. That's a real, useful result to be able to explain, not
  something to treat as a failed optimisation.

### If you want to go further

Sweep `TILE` (32/64/128/256) and see whether it changes anything, in either
precision. If you have profiler access, compare Compute (SM) Throughput
against memory throughput for `calc_acc` vs `calc_acc_tiled` at both
precisions with `ncu`/`nsys` — that will show you directly which resource
each kernel is limited by, rather than inferring it from timing alone.

---

## If you finish early

- **Try `collapse`.** The pairwise loop in `calc_acc` is a perfect square
  (`i` and `j` both run `1..n`), but you only parallelised the outer `i`
  loop. Look up the `collapse` clause and see whether collapsing both loops
  into a single parallel iteration space changes performance, and why (or
  why not) — is this kernel limited by the number of parallel iterations
  available, or by something else?
- **Try `num_teams` / `thread_limit`.** These clauses let you control the
  GPU launch configuration explicitly instead of leaving it to the
  compiler. Sweep a few values and see whether you can beat the default.
- **Profile it.** `nsys profile -o report ./main_gpu` followed by
  `nsys stats --report cuda_gpu_kern_sum report.nsys-rep` will show you
  which kernel actually dominates runtime, and
  `nsys stats --report cuda_gpu_mem_time_sum report.nsys-rep` will show any
  remaining host/device transfers. Is there anything left to move into the
  `target data` region, or any transfer you didn't expect?
