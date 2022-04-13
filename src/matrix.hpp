#ifndef TOOL_MATRIX_HPP__
#define TOOL_MATRIX_HPP__
// MIT License

// Copyright (c) 2022 rdmpl

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <assert.h>
#include <stdio.h>

#include <array>
#include <initializer_list>
#include <vector>

namespace rdmplmat {
namespace {
template <typename T, class... Args>
inline std::vector<T> input_dims(Args... args) {
  return {args...};
}
}  // namespace
template <typename T, typename Size = int>
class matrix {
 public:
  class matrix_fragment {
   public:
    class iterator {
     public:
      iterator& operator++() {
        value_++;
        return *this;
      };
      T& operator*() { return instance_.matrix_.data_[map()]; }
      const T& operator*() const { return instance_.matrix_.data_[map()]; }
      bool operator!=(const iterator& other) const {
        return &instance_ != &other.instance_ || value_ != other.value_;
      }
      iterator(const matrix_fragment& instance, Size value)
          : instance_(instance), value_(value) {}
      iterator(const iterator& others)
          : instance_(others.instance_), value_(others.value_) {}

     private:
      const Size map() const;

     private:
      const matrix_fragment& instance_;
      Size value_;
    };
    typedef const iterator const_iterator;
    typedef typename matrix::value_type value_type;

   public:
    matrix_fragment(const matrix& mat,
                    const std::vector<std::vector<Size>>& fragment);
    matrix_fragment(const matrix& mat,
                    const std::vector<std::initializer_list<Size>>& fragment);
    matrix_fragment(matrix_fragment&& other)
        : matrix_(other.matrix_),
          fragment_(std::move(other.fragment_)),
          weight_(std::move(other.weight_)),
          dimension_(std::move(other.dimension_)),
          size_(other.size_) {
      other.size_ = 0;
    }
    iterator begin() { return iterator(*this, 0); }
    iterator end() { return iterator(*this, size_); }
    const_iterator begin() const { return iterator(*this, 0); }
    const_iterator end() const { return iterator(*this, size_); }
    matrix_fragment& operator=(const value_type&& value) {
      for (auto& v : *this) {
        v = value;
      }
      return *this;
    }
    template <typename container_type>
    matrix_fragment& assign(const container_type& other, bool cycle) {
      auto this_element = begin();
      auto this_element_end = end();
      while (this_element != this_element_end) {
        auto other_element = other.begin();
        auto other_element_end = other.end();
        while (this_element != this_element_end &&
               other_element != other_element_end) {
          *this_element = *other_element;
          ++this_element;
          ++other_element;
        }
        if (!cycle || !(other.begin() != other.end())) {
          break;
        }
      }
      return *this;
    }
    template <typename container_type>
    matrix_fragment& operator=(const container_type& others) {
      return assign(others, false);
    }
    matrix_fragment& operator=(const matrix_fragment&& others) {
      return assign(others, false);
    }
    matrix_fragment& operator=(const matrix_fragment& others) {
      return assign(others, false);
    }

   private:
    const matrix& matrix_;
    std::vector<std::vector<Size>> fragment_;
    std::vector<Size> weight_;
    std::vector<Size> dimension_;
    Size size_;
    friend class matrix;
  };

 public:
  typedef T value_type;
  typedef value_type* iterator;
  typedef const value_type* const_iterator;
  matrix();
  matrix(matrix&& other);
  matrix(const matrix& other);
  matrix(const matrix_fragment& fragment);
  template <class... Args>
  inline matrix(Size first, Args... args);
  matrix(std::vector<Size>& dimension);
  matrix(std::vector<Size>&& dimension);
  matrix(std::initializer_list<Size> list) : matrix(std::vector<Size>(list)) {}
  ~matrix();
  Size size() const { return size_; }
  const std::vector<Size>& dimension() const { return dimension_; }
  iterator begin() { return data_; }
  iterator end() { return data_ + size_; }
  const_iterator begin() const { return data_; }
  const_iterator end() const { return data_ + size_; }
  matrix& clear(void);

 public:
  template <class... Args>
  matrix& reshape(Size first, Args... args);
  template <class... Args>
  matrix& repmat(Size first, Args... args);
  template <class... Args>
  matrix& permute(Size first, Args... args);
  template <typename... Args>
  matrix& ones(Size first, Args... args) {
    *this = matrix(args...);
    for (int i = 0; i < size_; i++) {
      data_[i] = static_cast<T>(1);
    }
    return *this;
  }
  matrix& linspace(const T&& start, const T&& end, Size N = 100) {
    *this = matrix(N);
    for (Size index = 0; index < size_; index++) {
      data_[index] = start + (end - start) * index / (N - 1);
    }
    return *this;
  }
  matrix& squeeze(void);
  matrix sum(Size dimension_at = 0) const;

 public:
  template <class... Args>
  T& operator()(Size first, Args... index);
  template <class... Args>
  const T& operator()(Size first, Args... index) const;
  template <class... Args>
  matrix_fragment operator()(std::initializer_list<Args>... args) const;
  matrix_fragment operator()(std::vector<std::vector<Size>>& fragment) const;
#define impl_matrix_operator(oper)         \
  matrix& operator oper(const T&& value) { \
    for (auto& val : *this) {              \
      val oper value;                      \
    }                                      \
    return *this;                          \
  }
  impl_matrix_operator(=);
  impl_matrix_operator(*=);
  impl_matrix_operator(+=);
  impl_matrix_operator(-=);
  impl_matrix_operator(/=);
  impl_matrix_operator(|=);
  impl_matrix_operator(&=);
  matrix& operator=(const matrix& other) {
    if (data_ != nullptr && size_ != other.size_) {
      delete[] data_;
      data_ = nullptr;
    }
    weight_ = other.weight_;
    dimension_ = other.dimension_;
    size_ = other.size_;
    if (data_ == nullptr && size_ != 0) {
      data_ = new T[size_];
    }
    for (int i = 0; i < size_; i++) {
      data_[i] = other.data_[i];
    }
    return *this;
  }
  matrix& operator=(matrix&& other) {
    dimension_ = std::move(other.dimension_);
    weight_ = std::move(other.weight_);
    if (data_ != other.data_ && data_ != nullptr) {
      delete[] data_;
    }
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
  }
  matrix& operator*=(const matrix& other) {
    if (size_ >= other.size_) {
      for (Size index = 0; index < size_; index++) {
        data_[index] *= other.data_[index % other.size_];
      }
    } else {
      matrix ans(other);
      for (Size index = 0; index < ans.size_; index++) {
        ans.data_[index] *= data_[index % size_];
      }
      *this = std::move(ans);
    }
    return *this;
  }

 public:
  static Size compute_size_and_weight(const std::vector<Size>& dimension,
                                      std::vector<Size>& weight);
  static Size compute_index(const std::vector<Size>& position,
                            const std::vector<Size>& weight);
  static int compute_position(Size index, const std::vector<Size>& weight,
                              std::vector<Size>& position);

 private:
  std::vector<Size> dimension_;
  std::vector<Size> weight_;
  Size size_;
  mutable T* data_;
};
template <typename T, typename Size>
matrix<T, Size>::matrix_fragment::matrix_fragment(
    const matrix& mat, const std::vector<std::vector<Size>>& fragment)
    : matrix_(mat) {
  std::size_t dimension_index = 0;
  fragment_.reserve(matrix_.dimension_.size());
  dimension_.reserve(matrix_.dimension_.size());
  for (const auto& value : fragment) {
    assert(dimension_index < matrix_.dimension_.size() && value.size() > 0);
    fragment_.push_back(value);
    Size frag_size = value.size();
    if (fragment_[dimension_index][0] < 0) {
      Size step = -fragment_[dimension_index][0];
      Size begin = 1;
      Size end = matrix_.dimension_[dimension_index];
      if (fragment_[dimension_index].size() > 1) {
        begin = fragment_[dimension_index][1];
      } else {
        fragment_[dimension_index].push_back(begin);
      }
      if (fragment_[dimension_index].size() > 2) {
        end = fragment_[dimension_index][2];
      } else {
        fragment_[dimension_index].push_back(end);
      }
      frag_size = (end - begin) / step + 1;
    }
    dimension_.push_back(frag_size);
    dimension_index++;
  }
  while (dimension_index < matrix_.dimension_.size()) {
    Size frag_size = matrix_.dimension_[dimension_index];
    fragment_.push_back({-1, 1, frag_size});
    dimension_.push_back(frag_size);
    ++dimension_index;
  }
  size_ = matrix::compute_size_and_weight(dimension_, weight_);
  assert(fragment_.size() == matrix_.weight_.size());
}
template <typename T, typename Size>
matrix<T, Size>::matrix_fragment::matrix_fragment(
    const matrix<T, Size>& mat,
    const std::vector<std::initializer_list<Size>>& fragment)
    : matrix_(mat) {
  std::vector<std::vector<Size>> vfrag;
  for (const auto& value : fragment) {
    vfrag.push_back(value);
  }
  new (this) matrix_fragment(mat, vfrag);
}
template <typename T, typename Size>
Size matrix<T, Size>::compute_size_and_weight(
    const std::vector<Size>& dimension, std::vector<Size>& weight) {
  Size size = 0;
  weight.clear();
  if (dimension.size() != 0) {
    size = 1;
    weight.reserve(dimension.size());
    for (auto value : dimension) {
      assert(value > 0);
      weight.push_back(size);
      size *= value;
    }
  }
  return size;
}
template <typename T, typename Size>
Size matrix<T, Size>::compute_index(const std::vector<Size>& position,
                                    const std::vector<Size>& weight) {
  Size index = 0;
  auto position_length = position.size();
  auto weight_length = weight.size();
  for (std::size_t i = 0; i < position_length && i < weight_length; i++) {
    index = index + (position[i] - 1) * weight[i];
  }
  return index;
}
template <typename T, typename Size>
int matrix<T, Size>::compute_position(Size index,
                                      const std::vector<Size>& weight,
                                      std::vector<Size>& position) {
  int length = weight.size();
  position.resize(length);
  for (int i = length - 1; i >= 0; --i) {
    position[i] = 1 + index / weight[i];
    index %= weight[i];
  }
  return 0;
}

template <typename T, typename Size>
matrix<T, Size>::matrix(std::vector<Size>&& dimension) : matrix(dimension) {}
template <typename T, typename Size>
matrix<T, Size>::matrix(std::vector<Size>& dimension) {
  dimension_ = dimension;
  size_ = compute_size_and_weight(dimension_, weight_);
  if (size_ > 0) {
    data_ = new T[size_];
  } else {
    data_ = nullptr;
  }
}
template <typename T, typename Size>
matrix<T, Size>::matrix() {
  size_ = 0;
  data_ = nullptr;
}
template <typename T, typename Size>
template <class... Args>
matrix<T, Size>::matrix(Size first, Args... args) {
  auto dimension = input_dims<Size>(first, args...);
  new (this) matrix(dimension);
}
template <typename T, typename Size>
matrix<T, Size>::matrix(matrix&& other) : matrix() {
  *this = std::move(other);
}
template <typename T, typename Size>
matrix<T, Size>::matrix(const matrix& other) {
  dimension_ = other.dimension_;
  weight_ = other.weight_;
  size_ = other.size_;
  data_ = new T[size_];
  for (int i = 0; i < size_; i++) {
    data_[i] = other.data_[i];
  }
}

template <typename T, typename Size>
matrix<T, Size>::matrix(const matrix_fragment& fragment) {
  dimension_ = fragment.dimension_;
  weight_ = fragment.weight_;
  size_ = fragment.size_;
  data_ = new T[size_];
  int index = 0;
  for (const auto& value : fragment) {
    data_[index++] = value;
  }
}
template <typename T, typename Size>
matrix<T, Size>& matrix<T, Size>::clear(void) {
  if (data_) {
    delete[] data_;
    data_ = nullptr;
  }
  size_ = 0;
  dimension_.clear();
  weight_.clear();
  return *this;
}
template <typename T, typename Size>
matrix<T, Size>::~matrix() {
  clear();
}

template <typename T, typename Size>
template <class... Args>
T& matrix<T, Size>::operator()(Size first, Args... index) {
  auto position = input_dims<Size>(first, index...);
  Size idx = compute_index(position, weight_);
  assert(idx < size_);
  return data_[idx];
}
template <typename T, typename Size>
template <class... Args>
const T& matrix<T, Size>::operator()(Size first, Args... index) const {
  auto position = input_dims<Size>(first, index...);
  Size idx = compute_index(position, weight_);
  assert(idx < size_);
  return data_[idx];
}
template <typename T, typename Size>
template <class... Args>
matrix<T, Size>& matrix<T, Size>::reshape(Size first, Args... args) {
  dimension_ = input_dims<Size>(first, args...);
  bool auto_complete_flag = false;
  assert(dimension_.size() > 0);
  if (dimension_[dimension_.size() - 1] < 0) {
    dimension_.resize(dimension_.size() - 1);
    auto_complete_flag = true;
  }
  auto new_size = compute_size_and_weight(dimension_, weight_);
  if (auto_complete_flag) {
    assert(size_ % new_size == 0);
    dimension_.push_back(size_ / new_size);
    weight_.push_back(new_size);
  } else {
    size_ = new_size;
  }
  return *this;
}

template <typename T, typename Size>
template <class... Args>
matrix<T, Size>& matrix<T, Size>::repmat(Size first, Args... args) {
  auto ans_dimension = input_dims<Size>(first, args...);
  std::vector<Size> rep_dimension = ans_dimension;
  std::size_t i = 0;
  while (i < ans_dimension.size() && i < dimension_.size()) {
    ans_dimension[i] *= dimension_[i];
    ++i;
  }
  if (i < dimension_.size()) {
    rep_dimension.resize(dimension_.size(), 1);
    ans_dimension.insert(ans_dimension.end(), dimension_.begin() + i,
                         dimension_.end());
  }
  matrix ans_matrix(ans_dimension);
  std::vector<Size> rep_weight;
  std::vector<Size> rep_position;
  Size rep_size = compute_size_and_weight(rep_dimension, rep_weight);
  std::vector<std::vector<Size>> fragment(rep_dimension.size());
  for (Size index = 0; index < rep_size; index++) {
    compute_position(index, rep_weight, rep_position);
    for (std::size_t j = 0; j < rep_position.size(); j++) {
      if (j < dimension_.size()) {
        auto begin = 1 + (rep_position[j] - 1) * dimension_[j];
        auto end = rep_position[j] * dimension_[j];
        fragment[j] = {-1, begin, end};
      } else {
        fragment[j] = {rep_position[j]};
      }
    }
    ans_matrix(fragment) = *this;
  }
  *this = std::move(ans_matrix);
  return *this;
}
template <typename T, typename Size>
template <class... Args>
matrix<T, Size>& matrix<T, Size>::permute(Size first, Args... args) {
  std::vector<Size> ans_dimension;
  auto vsort = input_dims<Size>(first, args...);
  ans_dimension.reserve(dimension_.size());
  for (std::size_t i = 0; i < dimension_.size(); i++) {
    ans_dimension.push_back(dimension_[vsort[i] - 1]);
  }
  matrix ans(ans_dimension);
  std::vector<Size> position;
  std::vector<Size> ans_position;
  ans_position.resize(ans_dimension.size());
  for (int i = 0; i < size_; i++) {
    compute_position(i, weight_, position);
    for (std::size_t j = 0; j < ans_dimension.size(); j++) {
      ans_position[j] = position[vsort[j] - 1];
    }
    auto index = compute_index(ans_position, ans.weight_);
    ans.data_[index] = data_[i];
  }
  *this = std::move(ans);
  return *this;
}
template <typename T, typename Size>
matrix<T, Size>& matrix<T, Size>::squeeze(void) {
  std::size_t i = 0, j = 0;
  while (i < dimension_.size()) {
    if (dimension_[i] != 1) {
      dimension_[j++] = dimension_[i];
    }
    ++i;
  }
  if (j == 0 && size_ > 0) {
    dimension_[j++] = size_;
  }
  dimension_.resize(j);
  size_ = compute_size_and_weight(dimension_, weight_);
  return *this;
}
template <typename T, typename Size>
matrix<T, Size> matrix<T, Size>::sum(Size dimension_at) const {
  if (dimension_at == 0) {
    while ((std::size_t)dimension_at < dimension_.size() &&
           dimension_[dimension_at] == 1) {
      dimension_at++;
    }
  } else {
    dimension_at -= 1;
  }
  std::vector<std::vector<Size>> fragment;
  for (std::size_t i = 0; i < dimension_.size(); i++) {
    fragment.push_back({-1});
  }
  fragment[dimension_at][0] = 1;
  matrix sum = (*this)(fragment);
  for (Size i = 2; i <= dimension_[dimension_at]; i++) {
    fragment[dimension_at][0] = i;
    auto matrix_frag = (*this)(fragment);
    Size index = 0;
    for (auto& value : matrix_frag) {
      if (index < sum.size_) {
        sum.data_[index++] += value;
      }
    }
  }
  return sum;
}
template <typename T, typename Size>
template <class... Args>
typename matrix<T, Size>::matrix_fragment matrix<T, Size>::operator()(
    std::initializer_list<Args>... args) const {
  auto slice = input_dims<std::initializer_list<Size>>(args...);
  return matrix_fragment(*this, slice);
}
template <typename T, typename Size>
typename matrix<T, Size>::matrix_fragment matrix<T, Size>::operator()(
    std::vector<std::vector<Size>>& fragment) const {
  return matrix_fragment(*this, fragment);
}
template <typename T, typename Size>
const Size matrix<T, Size>::matrix_fragment::iterator::map() const {
  Size pos = 0;
  Size value = value_ % instance_.size_;
  int index = instance_.dimension_.size() - 1;
  while (value >= 0 && index >= 0) {
    Size weight = instance_.weight_[index];
    Size current = value / weight;
    value %= weight;
    Size dv = instance_.fragment_[index][0];
    if (dv > 0) {
      dv = instance_.fragment_[index][current];
    } else {
      dv = instance_.fragment_[index][1] -
           instance_.fragment_[index][0] * current;
    }
    pos = pos + (dv - 1) * instance_.matrix_.weight_[index];
    --index;
  }
  assert(pos < instance_.matrix_.size_);
  return pos;
}
}  // namespace rdmplmat
#endif