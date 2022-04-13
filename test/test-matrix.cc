#include "matrix.hpp"
#include "stdio.h"
using matrix_type = rdmplmat::matrix<int>;
int print_matrix(const matrix_type &matrix) {
  // print dimension
  const auto &dimension = matrix.dimension();
  printf("dimension: ");
  for (std::size_t i = 0; i < dimension.size(); i++) {
    printf("%d", dimension[i]);
    if (i != dimension.size() - 1) {
      printf(" x ");
    } else {
      printf("\n");
    }
  }
  if (dimension.size() == 1) {
    for (int i = 1; i <= dimension[0]; i++) {
      printf("%d ", matrix(i));
    }
    printf("\n");
  } else if (dimension.size() == 2) {
    for (int i = 1; i <= dimension[0]; i++) {
      for (int j = 1; j <= dimension[1]; j++) {
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
  print_matrix(mat);
  return 0;
}