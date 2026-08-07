#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "timer.hpp"

#include <thrust/copy.h>
#include <thrust/device_vector.h>

using std::vector;

typedef float real;

/// Fill input array `x` with `size` real numbers
void fill_rand_vec(real *x, uint size, real min, real max, int seed = 42) {
  srand(seed);

  // rand() carries global state and is not thread safe, so this cannot be
  // parallelised as written
  for (int i = 0; i < size; ++i) {
    x[i] = min + (static_cast<real>(rand()) /
                  (static_cast<real>(RAND_MAX / (max - min))));
  }
}

struct Vec2 {
  real x;
  real y;
};

__device__ real norm2(const Vec2 &v) { return v.x * v.x + v.y * v.y; }

__device__ Vec2 sub(const Vec2 &v1, const Vec2 &v2) {
  return {v1.x - v2.x, v1.y - v2.y};
}

__device__ Vec2 calc_acc_pair(Vec2 pi, Vec2 pj, real mj, real eps = 0.0) {
  const Vec2 r = sub(pj, pi);
  const real d = norm2(r) + eps * eps;
  const real inv_d = rsqrtf(d); // only when real = float and not double
  // const real inv_d = 1.0 / sqrt(d); // the fallback when real = double
  const real accs = mj * inv_d * inv_d * inv_d;
  return {r.x * accs, r.y * accs};
}

/// Calculate acceleration on each particle due to every other particle
__global__ void calc_acc(Vec2 *acc, const Vec2 *pos, const real *mass, uint N,
                         real eps = 0.0) {
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gtid >= N)
    return;

  const Vec2 pi = pos[gtid];
  Vec2 accl = {0.0, 0.0};

  for (int j = 0; j < N; ++j) {
    const Vec2 accll = calc_acc_pair(pi, pos[j], mass[j], eps);
    accl.x += accll.x;
    accl.y += accll.y;
  }

  acc[gtid] = accl;
}

/// Calculate acceleration on each particle due to every other particle
__global__ void calc_acc_tiled(Vec2 *acc, const Vec2 *pos, const real *mass,
                               uint N, real eps = 0.0) {
  // Task 2 — the numbered steps are in the exercise text

  // TODO 1.2: declare the shared arrays for positions and masses

  // TODO 1.3: global thread index, this thread's particle, zeroed accumulator

  // TODO 1.4: outer loop over tiles, each thread loading one particle

  // TODO 1.5: inner loop over the tile, accumulating from shared memory

  // TODO 1.6: __syncthreads() after the load, and after the inner loop

  // TODO 1.7: write the result back, guarded

  // TODO 1.8: handle a partly-filled final tile
}

/// Calculate next position of every particle from old position and
/// acceleration. The new position is written over pos_prev, which is no longer
/// needed as soon as it has been read.
__global__ void advance_pos(Vec2 *pos_prev, const Vec2 *pos, const Vec2 *acc,
                            uint N, real dt) {
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gtid >= N)
    return;
  pos_prev[gtid].x =
      2.0 * pos[gtid].x - pos_prev[gtid].x + acc[gtid].x * dt * dt;
  pos_prev[gtid].y =
      2.0 * pos[gtid].y - pos_prev[gtid].y + acc[gtid].y * dt * dt;
}

/// Calculate the position and velocity that would give a stable orbit around a
/// huge mass at position (0,0)
void calc_stable_orbit(vector<Vec2> &pos, vector<Vec2> &vel,
                       const vector<real> &r, const vector<real> &theta) {
  for (int i = 0; i < r.size(); ++i) {
    real v_mag = 1. / std::sqrt(r[i]);

    pos[i].x = r[i] * std::sin(theta[i]);
    pos[i].y = r[i] * std::cos(theta[i]);

    vel[i].x = -v_mag * std::cos(theta[i]);
    vel[i].y = v_mag * std::sin(theta[i]);
  }
}

/// Output list of positions as CSV file
void dump_to_file(const std::string &fname, const vector<Vec2> &pos) {
  std::ofstream out(fname);

  if (out.is_open()) {
    for (int i = 0; i < pos.size(); ++i) {
      out << pos[i].x << "," << pos[i].y << "\n";
    }
    out.close();
  } else
    std::cout << "Unable to open file";
}

/// Format integer into CSV filename
std::string format_fname(int count) {
  char buffer[16];
  sprintf(buffer, "%04d.csv", count);
  return std::string(buffer);
}

bool all_tests_pass();

int main() {
  const bool RUN_UNIT_TESTS = true;
  if (RUN_UNIT_TESTS)
    if (!all_tests_pass())
      return -1;

  // Parameters
  int seed = 42;
  uint N_PARTICLES = 221184;
  real dt = 0.01;
  real total_time = 10 * dt;

  // CUDA launch parameters
  const int block_size = 1024;
  const int n_blocks = (N_PARTICLES + block_size - 1) / block_size;

  // Prevents numerical errors when two particles are very close
  const real epsilon = 1.1 * std::pow(real(N_PARTICLES), -0.48);

  // Main variables
  vector<Vec2> pos(N_PARTICLES);
  vector<Vec2> acc(N_PARTICLES);
  vector<Vec2> pos_prev(N_PARTICLES);
  vector<real> mass(N_PARTICLES);

  // Setup initial conditions
  vector<Vec2> vel(N_PARTICLES);
  {
    vector<real> r(N_PARTICLES);
    vector<real> theta(N_PARTICLES);

    fill_rand_vec(r.data(), r.size(), 15, 20.0, seed);
    fill_rand_vec(theta.data(), theta.size(), 0, 2.0 * M_PI, seed + 1);
    fill_rand_vec(mass.data(), mass.size(), 1.0 / 6000000, 1.0 / 1000,
                  seed + 2);

    calc_stable_orbit(pos, vel, r, theta);

    pos[0] = {0, 0};
    vel[0] = {0, 0};
    mass[0] = 1.0;
  }

  thrust::device_vector<Vec2> pos_d(N_PARTICLES);
  thrust::copy(pos.begin(), pos.end(), pos_d.begin());
  thrust::device_vector<Vec2> pos_prev_d(N_PARTICLES);
  thrust::device_vector<real> mass_d(N_PARTICLES);
  thrust::copy(mass.begin(), mass.end(), mass_d.begin());

  // Step 0 to populate pos_prev before the main time-stepping loop
  thrust::device_vector<Vec2> acc_d(N_PARTICLES);
  calc_acc<<<n_blocks, block_size>>>(acc_d.data().get(), pos_d.data().get(),
                                     mass_d.data().get(), N_PARTICLES, epsilon);
  thrust::copy(acc_d.begin(), acc_d.end(), acc.begin());

  for (int i = 0; i < N_PARTICLES; ++i) {
    pos_prev[i].x = pos[i].x - vel[i].x * dt - 0.5 * acc[i].x * dt * dt;
    pos_prev[i].y = pos[i].y - vel[i].y * dt - 0.5 * acc[i].y * dt * dt;
  }
  thrust::copy(pos_prev.begin(), pos_prev.end(), pos_prev_d.begin());

  // Setup timer, reporting and file dump intervals
  Timer<std::chrono::microseconds> timer;
  real time_per_loop = 0.0;
  real total_elapsed_us = 0.0;

  real t_between_stat_prints = 0.1;
  real next_stat_print = t_between_stat_prints;

  real t_between_dump = 0.1;
  real next_dump = t_between_dump;

  real t = 0;
  int loop_counter = 0;
  int dump_counter = 0;

  // Start the timer, discarding setup, allocation and initial host-to-device
  // transfer time
  timer.lap();

  // t and dt are float values, so to ensure the final t value is as close as
  // possible to total_time for a given dt, an offset of dt/2 is set
  while (t < total_time - 0.5 * dt) {

    calc_acc<<<n_blocks, block_size>>>(acc_d.data().get(), pos_d.data().get(),
                                       mass_d.data().get(), N_PARTICLES,
                                       epsilon);
    advance_pos<<<n_blocks, block_size>>>(pos_prev_d.data().get(),
                                          pos_d.data().get(),
                                          acc_d.data().get(), N_PARTICLES, dt);
    pos_d.swap(pos_prev_d);

    loop_counter += 1;
    t += dt;

    if (t + 0.5 * dt >= next_dump) {
      thrust::copy(pos_d.begin(), pos_d.end(), pos.begin());
      dump_to_file(format_fname(dump_counter), pos);
      dump_counter += 1;
      next_dump += t_between_dump;
    }

    time_per_loop = timer.lap();
    total_elapsed_us += time_per_loop;

    if (t + 0.5 * dt >= next_stat_print) {
      const std::string ESC = "\x1b";
      const std::string CLEAR_SCREEN = ESC + "[2J";
      const std::string JUMP_HOME = ESC + "[H";

      std::cout << CLEAR_SCREEN << JUMP_HOME;
      std::cout << "N_PARTICLES: " << N_PARTICLES << "\n";
      std::cout << "Loop time: " << time_per_loop << " us\n";
      std::cout << "Loop count: " << loop_counter << "\n";
      std::cout << "Complete: " << t / total_time * 100 << "%" << "\n";
      next_stat_print += t_between_stat_prints;
    }
  }

  cudaDeviceSynchronize();
  total_elapsed_us += timer.lap();

  std::cout << "Steps: " << loop_counter << "\n";
  std::cout << "Total: " << total_elapsed_us << " us\n";
  std::cout << "Mean per step: " << total_elapsed_us / loop_counter << " us\n";
}

// TESTS
bool assert_nearly_eql(real x, real y, real eps = 1e-6) {
  if (std::fabs(x - y) < eps) {
    return true;
  } else {
    std::cout << x << " != " << y << "\n";
    return false;
  }
}

bool assert_nearly_eql(const Vec2 &v1, const Vec2 &v2, real eps = 1e-6) {
  return assert_nearly_eql(v1.x, v2.x, eps) &&
         assert_nearly_eql(v1.y, v2.y, eps);
}

bool test_calc_acc_x() {
  thrust::device_vector<real> mass({2.0, 0.5});
  thrust::device_vector<Vec2> pos({{0.0, 0.0}, {1.0, 0.0}});
  thrust::device_vector<Vec2> acc(pos.size());

  const real epsilon = 1.1 * std::pow(real(pos.size()), -0.48);
  const real e2 = std::pow(epsilon, 2);
  const real inv_d3 = std::pow(1 + e2, -1.5);

  calc_acc<<<1, 32>>>(acc.data().get(), pos.data().get(), mass.data().get(),
                      pos.size(), epsilon);
  cudaDeviceSynchronize();

  if (!assert_nearly_eql(acc[0], {mass[1] * inv_d3, 0}, 1e-5)) {
    return false;
  }

  if (!assert_nearly_eql(acc[1], {-mass[0] * inv_d3, 0}, 1e-5)) {
    return false;
  }

  return true;
}

bool test_calc_acc_y() {
  thrust::device_vector<real> mass({2.0, 0.5});
  thrust::device_vector<Vec2> pos({{0.0, 0.0}, {0.0, 1.0}});
  thrust::device_vector<Vec2> acc(pos.size());

  const real epsilon = 1.1 * std::pow(real(pos.size()), -0.48);
  const real e2 = std::pow(epsilon, 2);
  const real inv_d3 = std::pow(1 + e2, -1.5);

  calc_acc<<<1, 32>>>(acc.data().get(), pos.data().get(), mass.data().get(),
                      pos.size(), epsilon);
  cudaDeviceSynchronize();

  if (!assert_nearly_eql(acc[0], {0, mass[1] * inv_d3}, 1e-5)) {
    return false;
  }

  if (!assert_nearly_eql(acc[1], {0, -mass[0] * inv_d3}, 1e-5)) {
    return false;
  }

  return true;
}

bool test_advance_pos() {
  const real dt = 0.5;

  thrust::device_vector<Vec2> pos({{1.0, 2.0}});
  thrust::device_vector<Vec2> pos_prev({{0.5, 3.0}});
  thrust::device_vector<Vec2> acc({{0.5, -1.0}});

  advance_pos<<<1, 1>>>(pos_prev.data().get(), pos.data().get(),
                        acc.data().get(), pos.size(), dt);
  cudaDeviceSynchronize();
  pos.swap(pos_prev);

  return assert_nearly_eql(
      pos[0], {2.0 - 0.5 + 0.5 * dt * dt, 4.0 - 3.0 + (-1.0) * dt * dt});
}

bool all_tests_pass() {
  bool all_tests_passed = true;
  if (!test_advance_pos()) {
    std::cout << "test_advance_pos failed!\n";
    all_tests_passed = false;
  }

  if (!test_calc_acc_x()) {
    std::cout << "test_calc_acc_x failed!\n";
    all_tests_passed = false;
  }

  if (!test_calc_acc_y()) {
    std::cout << "test_calc_acc_y failed!\n";
    all_tests_passed = false;
  }

  return all_tests_passed;
}
