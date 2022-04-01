#include "matrix.hpp"
#include "stdio.h"
using matrix_v2_type = rdmplmat::matrix<int>;
static int func() { return 0; }
int main() {
  int a;
  a = 1;
  matrix_v2_type m1_v2(5, 8, 4, 2, 3, 4);
  int test_start = 0;
  test_start = 0;
  for (auto &value : m1_v2) {
    value = test_start++;
  }
  m1_v2.permute(2, 1, 4, 6, 5, 3);
  m1_v2.reshape(40, 8, 12);
  m1_v2.permute(2, 1, 3);
  matrix_v2_type m2_v2 = m1_v2({-1}, {-2, 1}, {1, 2, 3});
  m2_v2({1, 4, 5, 6}, {-1}, {-1}) = m2_v2({2, 3, 7, 8}, {-1}, {-1});
  m2_v2.repmat(3);
  m2_v2.repmat(3, 4, 5);
  m2_v2.repmat(2, 2, 6, 6, 1);
  m2_v2(2, 2, 6, 6, 1) = 10288;
  matrix_v2_type m3_v2({1, 2, 4});
  m3_v2.linspace(1, 100, 20);
  m3_v2 *= m3_v2;
  return 0;
}