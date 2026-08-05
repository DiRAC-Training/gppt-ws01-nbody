#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "timer.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using std::vector;

typedef double real;

const int THREADS_PER_BLOCK = 32;

/// Fill input array `x` with `size` real numbers
void fill_rand_vec(real *x, int size, real min, real max, int seed = 42) {
  srand(seed);

#pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    x[i] = min + (static_cast<real>(rand()) /
                  (static_cast<real>(RAND_MAX / (max - min))));
  }
}

struct Vec2 {
  real x;
  real y;

  // real norm2() const { return x * x + y * y; }
  // Vec2 sub(const Vec2 &v) const { return {x - v.x, y - v.y}; }
};

__device__ __host__ real norm2(const Vec2 &v) {
  return v.x * v.x + v.y * v.y;
}

__device__ __host__ Vec2 sub(const Vec2 &v1, const Vec2 &v2) {
  return {v1.x - v2.x, v1.y - v2.y};
}

__device__ __host__ Vec2 calc_acc_pair(Vec2 pi, Vec2 pj, real mj, real eps = 0.0) {
      const Vec2 r = sub(pj, pi);
      const real accs = mj * pow(norm2(r) + eps * eps, -1.5);
  return {r.x * accs, r.y * accs};
}

void calc_acc_cpu(std::vector<Vec2> &acc, const std::vector<Vec2> &pos,
              const std::vector<real> &mass, real eps = 0.0) {
  for (int i = 0; i < acc.size(); ++i) {
    acc[i] = {0, 0};
    for (int j = 0; j < acc.size(); ++j) {
      const Vec2 accl = calc_acc_pair(pos[i], pos[j], mass[j], eps);
      acc[i].x += accl.x;
      acc[i].y += accl.y;
    }
  }
}

/// Calculate acceleration on each particle due to every other particle
__global__ void calc_acc_tiled(Vec2 *acc, const Vec2 *pos, const real *mass, int N, real eps = 0.0) {
  /// The way this splits the domain, each GPU thread calculates the acceleration on *one* particle - no race conditions!

  __shared__ Vec2 shPosition[THREADS_PER_BLOCK];
  __shared__ real shMass[THREADS_PER_BLOCK];

  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const Vec2 pi = pos[gtid];
  Vec2 accl = {0.0, 0.0};

  int tile = 0;
  // Strided access of N
  for (int i = 0; i < N; i += THREADS_PER_BLOCK) {
    // Each thread caches a position and mass from the current tile
    int idx = tile * blockDim.x + threadIdx.x;
    shPosition[threadIdx.x] = pos[idx];
    shMass[threadIdx.x] = mass[idx];
    // Sync before we use the data
    __syncthreads();

    // Each thread loops across the entire tile
    for (int j = 0; j < blockDim.x; j++) {
      // Add the acceleration from jth particle to this
      const Vec2 accll = calc_acc_pair(pi, shPosition[j], shMass[j], eps);
      accl.x += accll.x;
      accl.y += accll.y;
    }
    __syncthreads();

    tile += 1;
  }

  acc[gtid] = accl;
}

/// Calculate acceleration on each particle due to every other particle
__global__ void calc_acc_naive(Vec2 *acc, const Vec2 *pos, const real *mass, int N, real eps = 0.0) {
  /// The way this splits the domain, each GPU thread calculates the acceleration on *one* particle - no race conditions!
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const Vec2 pi = pos[gtid];
  Vec2 accl = {0.0, 0.0};

  // Strided access of N
  for (int j = 0; j < N; ++j) {
    const Vec2 accll = calc_acc_pair(pi, pos[j], mass[j], eps);
    accl.x += accll.x;
    accl.y += accll.y;
  }

  acc[gtid] = accl;
}

/// Calculate next position of every particle from old position and acceleration
__global__ void advance_pos(Vec2 *pos, const Vec2 *pos_prev, const Vec2 *acc, int N, real dt) {

  int gtid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = gtid; i < N; i += THREADS_PER_BLOCK) {
    pos[i].x = 2.0 * pos[i].x - pos_prev[i].x + acc[i].x * dt * dt;
    pos[i].y = 2.0 * pos[i].y - pos_prev[i].y + acc[i].y * dt * dt;
  }
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
  uint N_PARTICLES = 20000;
  real dt = 0.01;
  real total_time = 10 * dt;

  // This prevents numerical errors when two particles are very close
  const real epsilon = 1.1 * std::pow(real(N_PARTICLES), -0.48);

  // Main variables

  vector<Vec2> pos(N_PARTICLES);
  vector<Vec2> acc(N_PARTICLES);
  vector<Vec2> pos_prev(N_PARTICLES);
  vector<Vec2> pos_temp(N_PARTICLES);
  vector<real> mass(N_PARTICLES);

  // Setup initial conditions
  {
    vector<Vec2> vel(N_PARTICLES);
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

    calc_acc_cpu(acc, pos, mass, epsilon);

    for (int i = 0; i < pos.size(); ++i) {
      pos_prev[i].x = pos[i].x - vel[i].x * dt - 0.5 * acc[i].x * dt * dt;
      pos_prev[i].y = pos[i].y - vel[i].y * dt - 0.5 * acc[i].y * dt * dt;
    }
  }

  thrust::device_vector<Vec2> pos_d(N_PARTICLES);
  thrust::copy(pos.begin(), pos.end(), pos_d.begin());
  thrust::device_vector<Vec2> pos_prev_d(N_PARTICLES);
  thrust::copy(pos_prev.begin(), pos_prev.end(), pos_prev_d.begin());
  thrust::device_vector<real> mass_d(N_PARTICLES);
  thrust::copy(mass.begin(), mass.end(), mass_d.begin());

  thrust::device_vector<Vec2> acc_d(N_PARTICLES);
  thrust::device_vector<Vec2> pos_temp_d(N_PARTICLES);

  Timer<std::chrono::microseconds> timer;
  real time_per_loop = 0.0;

  real next_stat_print = 0.0;
  real t_between_stat_prints = 0.1 * total_time;

  real next_dump = 0.0;
  real t_between_dump = 0.1 * total_time;

  real t = 0;
  int loop_counter = 0;
  int dump_counter = 0;

  const int N_BLOCKS = N_PARTICLES / THREADS_PER_BLOCK + 1;

  while (t < total_time) {
    time_per_loop = timer.lap();
    calc_acc_tiled<<<N_BLOCKS, THREADS_PER_BLOCK>>>(acc_d.data().get(), pos_d.data().get(), mass_d.data().get(), N_PARTICLES, epsilon);
    thrust::copy(pos_d.begin(), pos_d.end(), pos_temp_d.begin());
    advance_pos<<<N_BLOCKS, THREADS_PER_BLOCK>>>(pos_d.data().get(), pos_prev_d.data().get(), acc_d.data().get(), N_PARTICLES, dt);
    pos_temp_d.swap(pos_prev_d);
    if (t >= next_dump ) {
      thrust::copy(pos_d.begin(), pos_d.end(), pos.begin());
      dump_to_file(format_fname(dump_counter), pos);
      dump_counter += 1;
      next_dump += t_between_dump;
    }

    if (t >= next_stat_print) {
      const std::string ESC = "\x1b";
      const std::string CLEAR_SCREEN = ESC + "[2J";
      const std::string JUMP_HOME = ESC + "[H";

      std::cout << CLEAR_SCREEN << JUMP_HOME;
      std::cout << "N_PARTICLES: " << N_PARTICLES << "\n";
      std::cout << "Loop time: " << time_per_loop << " us\n";
      std::cout << "Loop count: " << loop_counter << "\n";
      std::cout << "Complete: " << t / total_time * 100 << " %" << "\n";
      next_stat_print += t_between_stat_prints;
    }

    t += dt;
    loop_counter += 1;
  }
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
  vector<real> mass({2.0, 0.5});
  vector<Vec2> pos({{0.0, 0.0}, {1.0, 0.0}});
  vector<Vec2> acc(pos.size());

  const real epsilon = 1.1 * std::pow(real(pos.size()), -0.48);
  const real e2 = std::pow(epsilon, 2);

  calc_acc_cpu(acc, pos, mass, epsilon);

  if (!assert_nearly_eql(acc[0], {mass[1] * std::pow((1 + e2), -1.5), 0})) {
    return false;
  }

  if (!assert_nearly_eql(acc[1], {-mass[0] * std::pow((1 + e2), -1.5), 0})) {
    return false;
  }

  return true;
}

bool test_calc_acc_y() {
  vector<real> mass({2.0, 0.5});
  vector<Vec2> pos({{0.0, 0.0}, {0.0, 1.0}});
  vector<Vec2> acc(pos.size());

  const real epsilon = 1.1 * std::pow(real(pos.size()), -0.48);
  const real e2 = std::pow(epsilon, 2);

  calc_acc_cpu(acc, pos, mass, epsilon);

  if (!assert_nearly_eql(acc[0], {0, mass[1] * std::pow((1 + e2), -1.5)})) {
    return false;
  }

  if (!assert_nearly_eql(acc[1], {0, -mass[0] * std::pow((1 + e2), -1.5)})) {
    return false;
  }

  return true;
}

bool all_tests_pass() {
  bool all_tests_passed = true;
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
