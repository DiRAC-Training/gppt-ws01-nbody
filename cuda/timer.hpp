#pragma once

#include <chrono>

template<typename TimeUnit>
class Timer {
private:
  // Type aliases to make accessing nested type easier
  using Clock = std::chrono::steady_clock;
  std::chrono::time_point<Clock> m_beg;

public:
  Timer() : m_beg{Clock::now()} {};
  void reset() { m_beg = Clock::now(); }
  double elapsed() const {
    return std::chrono::duration_cast<TimeUnit>(Clock::now() - m_beg)
        .count();
  }
  double lap() {
    auto now = Clock::now();
    auto elapsed =
        std::chrono::duration_cast<TimeUnit>(now - m_beg).count();
    m_beg = now;
    return elapsed;
  }
};

