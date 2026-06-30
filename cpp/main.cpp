#include <vector>
#include <iostream>
#include <cmath>

using std::vector;

typedef double real;

bool nearly_eql(real x, real y, real eps = 1e-6) {
  return std::fabs(x - y) < eps;
}

struct Vec2 {
  real x;
  real y;

  real norm2() const {
    return this->x*this->x + this->y*this->y;
  }

  Vec2 sub(const Vec2 &v) const {
    return {this->x - v.x, this->y - v.y};
  }

  bool nearly_eql(const Vec2 &v, real eps = 1e-6) const {
    return ::nearly_eql(this->x, v.x) && ::nearly_eql(this->y, v.y);
  }
};

void calc_acc(vector<Vec2> &acc, vector<Vec2> &pos, vector<real> &mass, real eps = 0.0) {
  for(int i=0; i<acc.size(); ++i) {
    acc[i] = {0, 0};
    for(int j=0; j<acc.size(); ++j) {
      const Vec2 r = pos[j].sub(pos[i]);
      const real accs = mass[j] * std::pow(r.norm2() + eps*eps, -1.5);
      acc[i].x += r.x * accs;
      acc[i].y += r.y * accs;
    }
  }
}

void test_calc_acc_x() {
  vector<real> mass({2.0, 0.5});
  vector<Vec2> pos({{0.0, 0.0}, {1.0, 0.0}});
  vector<Vec2> acc(pos.size());

  const real epsilon = 1.1*std::pow(real(pos.size()), -0.48);

  calc_acc(acc, pos, mass, epsilon);

  const real eps = 1e-6;
  if(!acc[0].nearly_eql({
    mass[1] * std::pow((1 + std::pow(epsilon, 2)), -1.5),
    0, 
  })) {
    std::cout << "test_calc_acc_x failed!\n";
  }

  if(!acc[1].nearly_eql({
    -mass[0] * std::pow((1 + std::pow(epsilon, 2)), -1.5),
    0, 
  })) {
    std::cout << "test_calc_acc_x failed!\n";
    std::cout << 0 << ", " << -mass[0] * std::pow((1 + std::pow(epsilon, 2)), -1.5) << "\n";
    std::cout << acc[0].x << ", " << acc[0].y << "\n";
  }
}

void test_calc_acc_y() {
  vector<real> mass({2.0, 0.5});
  vector<Vec2> pos({{0.0, 0.0}, {0.0, 1.0}});
  vector<Vec2> acc(pos.size());

  const real epsilon = 1.1*std::pow(real(pos.size()), -0.48);

  calc_acc(acc, pos, mass, epsilon);

  const real eps = 1e-6;
  if(!acc[0].nearly_eql({
    0, 
    mass[1] * std::pow((1 + std::pow(epsilon, 2)), -1.5),
  })) {
    std::cout << "test_calc_acc_y failed!\n";
  }

  if(!acc[1].nearly_eql({
    0, 
    -mass[0] * std::pow((1 + std::pow(epsilon, 2)), -1.5),
  })) {
    std::cout << "test_calc_acc_y failed!\n";
  }
}

int main() {
  test_calc_acc_x();
  test_calc_acc_y();
}
