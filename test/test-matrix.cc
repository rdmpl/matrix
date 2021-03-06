#include <iostream>

#include "matrix.hpp"
#include "stdio.h"
using matrix_type = rdmplmat::matrix<int>;
int print_matrix(const matrix_type &matrix) {
  // print dimension
  const auto &dimension = matrix.dimension();
  printf("dimension: ");
  for (std::size_t i = 0; i < dimension.size(); i++) {
    // printf("%ld", dimension[i]);
    std::cout << dimension[i];
    if (i != dimension.size() - 1) {
      printf(" x ");
    } else {
      printf("\n");
    }
  }
  if (dimension.size() == 1) {
    for (auto i = (decltype(dimension[0]))1; i <= dimension[0]; i++) {
      printf("%d ", matrix(i));
    }
    printf("\n");
  } else if (dimension.size() == 2) {
    for (auto i = (decltype(dimension[0]))1; i <= dimension[0]; i++) {
      for (auto j = (decltype(dimension[0]))1; j <= dimension[1]; j++) {
        printf("%d ", matrix(i, j));
      }
      printf("\n");
    }
  }
  return 0;
}
int main() {
  matrix_type mat1({2, 3});
  print_matrix(mat1);
  matrix_type mat;
  mat.linspace(0, 99, 100).reshape(2, 2, 2);
  print_matrix(mat);
  matrix_type new_mat = mat.sum(2);
  new_mat.squeeze();
  print_matrix(new_mat);
  mat.reshape(4, 2);
  print_matrix(mat);
  mat.permute(2, 1);
  print_matrix(mat);
  mat.repmat(3, 2);
  matrix_type mm = mat({1, 2}, {2, 3});
  print_matrix(mat);
  mm.ones(2, 3);
  print_matrix(mm);
  return 0;
}