## Advanced Task (optional) — Tiling for GPU Shared-Memory Reuse

This task is a bigger structural change than Tasks 1-3, and assumes you've
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
Tasks 1-3 solution with an empty `calc_acc_tiled` stub. `nbody_tiled.f90`
in this same directory is the finished solution to this task; try not to
look at it until you've had a go, the same as `nbody.f90` for the main
exercise.

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

<details>
<summary>Hint — the self-interaction term</summary>

Don't special-case `j == i`: it's naturally visited once, inside whichever
tile it falls in, with `dx = dy = 0`, and contributes exactly zero — same
as in `calc_acc`.

</details>

<details>
<summary>Hint — padding, not branching, for the last tile</summary>

`n` won't usually be a whole multiple of `TILE`, so the last tile is only
partly full. `!$omp barrier` requires *every* thread in the team to reach
it. If a thread with nothing to load skips its write with an early branch
that also jumps past the barrier, the threads that do reach the barrier
wait for one that never arrives. Instead, give every thread the same
control-flow path through both barriers, and make the unused slots harmless
by padding them with `mass = 0` — a particle with zero mass contributes
nothing to the sum, so no extra condition is needed in the inner loop.

</details>

<details>
<summary>Hint — why TILE has to be a compile-time constant</summary>

It sizes the shared arrays, and it also becomes `thread_limit` — the
number of threads launched per team. Both have to be known when the code is
compiled, the same reason CUDA's tiled kernel needs a `const int` block
size rather than a runtime variable.

</details>

**Check your work:**

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

**If you want to go further:** sweep `TILE` (32/64/128/256) and see whether
it changes anything, in either precision. If you have profiler access,
compare Compute (SM) Throughput against memory throughput for `calc_acc` vs
`calc_acc_tiled` at both precisions with `ncu`/`nsys` — that will show you
directly which resource each kernel is limited by, rather than inferring it
from timing alone.

### Reflection

Take a moment to note down, or discuss with someone nearby:

- Tiling bought you ~30% in single precision but close to nothing in double
  precision, on the same GPU, on the same code. Before reading that result,
  would you have guessed a data-reuse optimisation could depend on floating
  point precision at all? What does that tell you about deciding whether an
  optimisation is worth the extra complexity, in your own work?
- Everything in this exercise had one particle per thread. What do you
  think changes about the tiling logic — the indexing, the barriers, the
  padding — if a thread instead owned several particles?

---

## Extension tasks

- **Switch between double and single precision with `make ... DOUBLE_PRECISION=true`.** How does the performance change? Is the tiling optimisation still useful?
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

### References

- [OpenMP Application Programming Interface, version 5.2](https://www.openmp.org/spec-html/5.2/openmp.html)
  — the authoritative reference for every directive and map type used here.
- [NVIDIA HPC SDK: OpenMP GPU Programming with the NVIDIA HPC Compilers](https://docs.nvidia.com/hpc-sdk/compilers/openmp-gpu/index.html)
  — `nvfortran`-specific detail on how `target` regions map onto CUDA
  concepts, including shared memory placement for `private` arrays.

---

Getting a loop to run on the GPU (Task 1) turned out to be the easy part.
Everything else in this exercise was really about one idea: a GPU is only
fast if the data it needs is already there. Whether that meant keeping
arrays resident across a whole time-stepping loop (Task 2), not bothering
for a call where the overhead doesn't matter (Task 2c), or restructuring a
kernel so a block's threads share one read from global memory instead of
each paying for their own (the Advanced task) — the question underneath
all of it was the same one: what does the device actually need, and when.
That question is worth carrying into any GPU code you write from here on.
