#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "util.hpp"

using std::vector;

typedef double real;

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

  real norm2() const { return x * x + y * y; }
  Vec2 sub(const Vec2 &v) const { return {x - v.x, y - v.y}; }
};

/// Calculate acceleration on each particle due to every other particle
void calc_acc(vector<Vec2> &acc, const vector<Vec2> &pos,
              const vector<real> &mass, real eps = 0.0) {
  for (int i = 0; i < acc.size(); ++i) {
    acc[i] = {0, 0};
    for (int j = 0; j < acc.size(); ++j) {
      const Vec2 r = pos[j].sub(pos[i]);
      const real accs = mass[j] * std::pow(r.norm2() + eps * eps, -1.5);
      acc[i].x += r.x * accs;
      acc[i].y += r.y * accs;
    }
  }
}

/// Calculate next position of every particle from old position and acceleration
void advance_pos(vector<Vec2> &pos, const vector<Vec2> &pos_prev,
                 const vector<Vec2> &acc, real dt) {
  for (int i = 0; i < pos.size(); ++i) {
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

bool all_tests_pass();

int main(int argc, char* argv[]) {
  // Parameters
  const int seed = get_argval<int>(argv, argv + argc, "--seed", 42);
  const uint N_PARTICLES = get_argval<uint>(argv, argv + argc, "-n", 256);
  const real dt = get_argval<real>(argv, argv + argc, "--dt", 0.1);
  const real total_time = get_argval<real>(argv, argv + argc, "--total_time", 100.0);
  const bool dump_data = get_arg(argv, argv + argc, "--dump");
  const bool quiet = get_arg(argv, argv + argc, "--quiet");
  const bool disable_unit_tests = get_arg(argv, argv + argc, "--disable_unit_tests");
  const real time_between_dumps = total_time / 100;
  const real time_between_reports = total_time / 1000;

  bool tests_passed = true;
  if (!disable_unit_tests) {
    tests_passed = all_tests_pass(); 
    if(!tests_passed) return -1;
  }

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

    calc_acc(acc, pos, mass, epsilon);

    for (int i = 0; i < pos.size(); ++i) {
      pos_prev[i].x = pos[i].x - vel[i].x * dt - 0.5 * acc[i].x * dt * dt;
      pos_prev[i].y = pos[i].y - vel[i].y * dt - 0.5 * acc[i].y * dt * dt;
    }
  }

  real t = 0;
  int loop_counter = 0;
  int dump_counter = 0;
  Timer<std::chrono::microseconds> timer;
  real ms_per_loop = 0.0;
  real time_to_next_dump = 0.0;
  real time_to_next_report = 0.0;

  while (t < total_time) {
    ms_per_loop = timer.lap();

    calc_acc(acc, pos, mass, epsilon);
    std::copy(pos.begin(), pos.end(), pos_temp.begin());
    advance_pos(pos, pos_prev, acc, dt);
    pos_temp.swap(pos_prev);
    if (t > time_to_next_dump and dump_data) {
      time_to_next_dump += time_between_dumps;
      dump_to_file(format_fname(dump_counter), pos);
      dump_counter += 1;
    }

    if(t > time_to_next_report and !quiet) {
      time_to_next_report += time_between_reports;
      std::cout << CLEAR_SCREEN << JUMP_HOME;
      if(tests_passed and !disable_unit_tests) {
        std::cout << "ALL TESTS PASSED\n";
      }
      std::cout << "N_PARTICLES: " << N_PARTICLES << "\n";
      std::cout << "Loop time: " << ms_per_loop << " us\n";
      const real pc_remaining = t / total_time * 100;
      printf("Sim time remaining: %.2f (%.2f \%)\n", total_time - t, pc_remaining);
      const int n_loops_remaining = (total_time - t ) / dt;
      const real realtime_remaining = n_loops_remaining * ms_per_loop;
      printf("Real time remaining: %.2f s\n", realtime_remaining / 1e6);
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

  calc_acc(acc, pos, mass, epsilon);

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

  calc_acc(acc, pos, mass, epsilon);

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
