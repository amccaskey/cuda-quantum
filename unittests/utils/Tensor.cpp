/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/utils/tensor.h"
#include "cudaq/tensor.h"

#include "gtest/gtest.h"

TEST(Tensor, initialization) {
  {
    cudaq::matrix_2 m0;
    EXPECT_EQ(m0.dump(), "{}");
  }
  {
    cudaq::matrix_2 m1({1., 0., 0., 1.});
    EXPECT_EQ(m1.dump(), "{  { (1,0)  (0,0) }\n   { (0,0)  (1,0) }\n }");
  }
  {
    cudaq::matrix_2 m1({1., 2., 3., 4., 5., 6.}, {2, 3});
    EXPECT_EQ(m1.dump(),
              "{  { (1,0)  (2,0)  (3,0) }\n   { (4,0)  (5,0)  (6,0) }\n }");
  }
}

TEST(Tensor, initializationError) {
  {
    EXPECT_THROW(cudaq::matrix_2 m1({1., 2., 3., 4., 5.}, {2, 3}),
                 std::runtime_error);
  }
}

TEST(Tensor, access) {
  {
    cudaq::matrix_2 m1({1., 0., 0., 1.});

    EXPECT_EQ((m1[{0, 1}]), 0.);

    m1[{0, 1}] = 4.;
    m1[{1, 0}] = m1[{0, 1}];

    EXPECT_EQ((m1[{0, 1}]), 4.);
    EXPECT_EQ((m1[{1, 0}]), 4.);
  }
  {
    cudaq::matrix_2 m1({1., 2., 3., 4., 5., 6.}, {2, 3});

    EXPECT_EQ((m1[{0, 2}]), 3.);

    m1[{0, 2}] = 9.;
    m1[{1, 0}] = m1[{0, 2}];

    EXPECT_EQ((m1[{0, 2}]), 9.);
    EXPECT_EQ((m1[{1, 0}]), 9.);
  }
}

TEST(Tensor, accessError) {
  {
    cudaq::matrix_2 m0;

    EXPECT_THROW((m0[{0}]), std::runtime_error);
    EXPECT_THROW((m0[{0, 1}]), std::runtime_error);
    EXPECT_THROW((m0[{0, 0, 0}]), std::runtime_error);
  }
  {
    cudaq::matrix_2 m1({1., 0., 0., 1.});

    EXPECT_THROW((m1[{0, 2}]), std::runtime_error);
    EXPECT_THROW((m1[{0, 0, 0}]), std::runtime_error);
  }
  {
    cudaq::matrix_2 m1({1., 2., 3., 4., 5., 6.}, {2, 3});

    EXPECT_THROW((m1[{0, 3}]), std::runtime_error);
    EXPECT_THROW((m1[{2, 1}]), std::runtime_error);
    EXPECT_THROW((m1[{0, 2, 3}]), std::runtime_error);
  }
}

TEST(Tensor, product) {
  {
    cudaq::matrix_2 m2({2., 1., 3., 4.});
    cudaq::matrix_2 m3({3., 2., 1., 4.});
    cudaq::matrix_2 m4 = m2 * m3;
    EXPECT_EQ(m4.dump(), "{  { (7,0)  (8,0) }\n   { (13,0)  (22,0) }\n }");
  }
  {
    cudaq::matrix_2 m2({1., 2., 3., 4., 5., 6.}, {3, 2});
    cudaq::matrix_2 m3({1., 2., 3., 4., 5., 6.}, {2, 3});
    cudaq::matrix_2 m4 = m2 * m3;
    EXPECT_EQ(m4.dump(), "{  { (9,0)  (12,0) }\n   { (15,0)  (19,0) }\n   { "
                         "(26,0)  (33,0) }\n }");
  }
}

TEST(Tensor, productError) {
  {
    cudaq::matrix_2 m2({2., 1., 3., 4.});
    cudaq::matrix_2 m3({1., 2., 3., 4., 5., 6.}, {3, 2});
    EXPECT_THROW(m2 * m3, std::runtime_error);
  }
}

TEST(Tensor, addition) {
  {
    cudaq::matrix_2 m5({2., 11., 3., 4.2});
    cudaq::matrix_2 m6({3., 42., 1.4, 4.});
    cudaq::matrix_2 m7 = m5 + m6;
    EXPECT_EQ(m7.dump(), "{  { (5,0)  (53,0) }\n   { (4.4,0)  (8.2,0) }\n }");
  }
}

TEST(Tensor, additionError) {
  {
    cudaq::matrix_2 m5({2., 1., 3., 4.});
    cudaq::matrix_2 m6({1., 2., 3., 4., 5., 6.}, {3, 2});
    EXPECT_THROW(m5 + m6, std::runtime_error);
  }
}

TEST(Tensor, subtraction) {
  {
    cudaq::matrix_2 m8({12.1, 1., 3., 14.});
    cudaq::matrix_2 m9({3., 22., 31., 4.});
    cudaq::matrix_2 ma = m8 - m9;
    EXPECT_EQ(ma.dump(), "{  { (9.1,0)  (-21,0) }\n   { (-28,0)  (10,0) }\n }");
  }
}
TEST(Tensor, subtractionError) {
  {
    cudaq::matrix_2 m8({2., 1., 3., 4.});
    cudaq::matrix_2 m9({1., 2., 3., 4., 5., 6.}, {3, 2});
    EXPECT_THROW(m8 - m9, std::runtime_error);
  }
}

TEST(Tensor, kroneckerProduct) {
  {
    cudaq::matrix_2 mb({6.1, 1.5, 3., 14.});
    cudaq::matrix_2 mc({7.4, 8., 9., 4.2});
    cudaq::matrix_2 md = cudaq::kronecker(mb, mc);
    EXPECT_EQ(
        md.dump(),
        "{  { (45.14,0)  (48.8,0)  (11.1,0)  (12,0) }\n   { (54.9,0)  "
        "(25.62,0)  (13.5,0)  (6.3,0) }\n   { (22.2,0)  (24,0)  (103.6,0)  "
        "(112,0) }\n   { (27,0)  (12.6,0)  (126,0)  (58.8,0) }\n }");
  }
}

TEST(Tensor, kroneckerOnList) {
  {
    cudaq::matrix_2 me({{1., 1.}}, {1, 1});
    cudaq::matrix_2 mf({1., 2.}, {1, 2});
    cudaq::matrix_2 mg({3., 4., 5.}, {3, 1});
    std::vector<cudaq::matrix_2> v{me, mf, mg};
    cudaq::matrix_2 mh = cudaq::kronecker(v.begin(), v.end());
    EXPECT_EQ(
        mh.dump(),
        "{  { (3,3)  (6,6) }\n   { (4,4)  (8,8) }\n   { (5,5)  (10,10) }\n }");
  }
}

TEST(CoreTester, checkTensorSimple) {

  {
    cudaq::tensor t({1, 2, 1});
    EXPECT_EQ(t.get_rank(), 3);
    EXPECT_EQ(t.get_num_elements(), 2);
    for (std::size_t i = 0; i < 1; i++)
      for (std::size_t j = 0; j < 2; j++)
        for (std::size_t k = 0; k < 1; k++)
          EXPECT_NEAR(t.at({i, j, k}).real(), 0.0, 1e-8);

    t.at({0, 1, 0}) = 2.2;
    EXPECT_NEAR(t.at({0, 1, 0}).real(), 2.2, 1e-8);

    EXPECT_ANY_THROW({ t.at({2, 2, 2}); });
  }

  {
    std::vector<std::complex<double>> data{1, 2, 3, 4};
    cudaq::tensor t(data, {2, 2});
    EXPECT_EQ(t.get_rank(), 2);
    EXPECT_EQ(t.get_num_elements(), 4);
    EXPECT_NEAR(t.at({0, 0}).real(), 1., 1e-8);
    EXPECT_NEAR(t.at({0, 1}).real(), 2., 1e-8);
    EXPECT_NEAR(t.at({1, 0}).real(), 3., 1e-8);
    EXPECT_NEAR(t.at({1, 1}).real(), 4., 1e-8);
  }
  {
    std::vector<std::complex<double>> data{1, 2, 3, 4};
    cudaq::tensor t(data, {2, 2});
    EXPECT_EQ(t.get_rank(), 2);
    EXPECT_EQ(t.get_num_elements(), 4);
    EXPECT_NEAR(t.at({0, 0}).real(), 1., 1e-8);
    EXPECT_NEAR(t.at({0, 1}).real(), 2., 1e-8);
    EXPECT_NEAR(t.at({1, 0}).real(), 3., 1e-8);
    EXPECT_NEAR(t.at({1, 1}).real(), 4., 1e-8);
  }

  {
    cudaq::tensor<int> t({1, 2, 1});
    EXPECT_EQ(t.get_rank(), 3);
    EXPECT_EQ(t.get_num_elements(), 2);
    for (std::size_t i = 0; i < 1; i++)
      for (std::size_t j = 0; j < 2; j++)
        for (std::size_t k = 0; k < 1; k++)
          EXPECT_NEAR(t.at({i, j, k}), 0.0, 1e-8);

    t.at({0, 1, 0}) = 2;
    EXPECT_EQ(t.at({0, 1, 0}), 2);

    EXPECT_ANY_THROW({ t.at({2, 2, 2}); });
  }
}

// Test elementwise operations
TEST(TensorTest, ElementwiseAddition) {

  // Initialize test data
  std::vector<double> data_a{1.0, 2.0, 3.0, 4.0};
  std::vector<double> data_b{5.0, 6.0, 7.0, 8.0};
  cudaq::tensor<double> a(data_a, {2, 2});
  cudaq::tensor<double> b(data_b, {2, 2});

  auto result = a + b;

  // Check result dimensions
  EXPECT_EQ(result.get_rank(), 2);
  EXPECT_EQ(result.shape()[0], 2);
  EXPECT_EQ(result.shape()[1], 2);

  // Check elementwise addition results
  EXPECT_DOUBLE_EQ(result.at({0, 0}), 6.0);  // 1 + 5
  EXPECT_DOUBLE_EQ(result.at({0, 1}), 8.0);  // 2 + 6
  EXPECT_DOUBLE_EQ(result.at({1, 0}), 10.0); // 3 + 7
  EXPECT_DOUBLE_EQ(result.at({1, 1}), 12.0); // 4 + 8
}

TEST(TensorTest, ElementwiseModulo) {

  std::vector<int> data_a{7, 8, 9, 10};
  std::vector<int> data_b{4, 3, 5, 2};
  cudaq::tensor<int> a(data_a, {2, 2});
  cudaq::tensor<int> b(data_b, {2, 2});
  auto result = a % b;

  EXPECT_EQ(result.get_rank(), 2);
  EXPECT_EQ(result.shape()[0], 2);
  EXPECT_EQ(result.shape()[1], 2);

  EXPECT_EQ(result.at({0, 0}), 3); // 7 % 4
  EXPECT_EQ(result.at({0, 1}), 2); // 8 % 3
  EXPECT_EQ(result.at({1, 0}), 4); // 9 % 5
  EXPECT_EQ(result.at({1, 1}), 0); // 10 % 2
}

TEST(TensorTest, Any) {
  {
    cudaq::tensor<uint8_t> a({7, 8, 9, 10}, {2, 2});

    uint8_t result = a.any();

    EXPECT_TRUE(result);
  }
  {
    cudaq::tensor<uint8_t> a({0, 0, 1, 0}, {2, 2});

    uint8_t result = a.any();

    EXPECT_TRUE(result);
  }
  {
    cudaq::tensor<uint8_t> a({2, 2});

    uint8_t result = a.any();

    EXPECT_FALSE(result);
  }
}

TEST(TensorTest, SumAll) {
  {
    // test int
    cudaq::tensor<int> a({7, 8, 9, 10}, {2, 2});

    int result = a.sum_all();

    EXPECT_EQ(result, 34);
  }
  {
    // test uint8_t
    cudaq::tensor<uint8_t> a({7, 8, 9, 10}, {2, 2});

    uint8_t result = a.sum_all();

    EXPECT_EQ(result, 34);
  }

  {
    // test float
    cudaq::tensor<float> a({7.1, 8.2, 9.1, 10.3}, {2, 2});

    float result = a.sum_all();

    float tolerance = 1.e-5;
    EXPECT_FLOAT_EQ(result, 34.7);
  }
}

TEST(TensorTest, ScalarModulo) {
  {
    // test int
    cudaq::tensor<int> a({7, 8, 9, 10}, {2, 2});

    auto result = a % 2;

    EXPECT_EQ(result.get_rank(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 2);

    EXPECT_EQ(result.at({0, 0}), 1); // 7 % 2
    EXPECT_EQ(result.at({0, 1}), 0); // 8 % 2
    EXPECT_EQ(result.at({1, 0}), 1); // 9 % 2
    EXPECT_EQ(result.at({1, 1}), 0); // 10 % 2
  }
}

TEST(TensorTest, MatrixDotProduct) {
  cudaq::tensor<double> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3});
  cudaq::tensor<double> b({7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, {3, 2});

  {
    auto result = a.dot(b);

    EXPECT_EQ(result.get_rank(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 2);

    // Matrix multiplication results
    EXPECT_DOUBLE_EQ(result.at({0, 0}), 58.0);  // 1*7 + 2*9 + 3*11
    EXPECT_DOUBLE_EQ(result.at({0, 1}), 64.0);  // 1*8 + 2*10 + 3*12
    EXPECT_DOUBLE_EQ(result.at({1, 0}), 139.0); // 4*7 + 5*9 + 6*11
    EXPECT_DOUBLE_EQ(result.at({1, 1}), 154.0); // 4*8 + 5*10 + 6*12
  }
  {
    auto result = a * b;

    EXPECT_EQ(result.get_rank(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 2);

    // Matrix multiplication results
    EXPECT_DOUBLE_EQ(result.at({0, 0}), 58.0);  // 1*7 + 2*9 + 3*11
    EXPECT_DOUBLE_EQ(result.at({0, 1}), 64.0);  // 1*8 + 2*10 + 3*12
    EXPECT_DOUBLE_EQ(result.at({1, 0}), 139.0); // 4*7 + 5*9 + 6*11
    EXPECT_DOUBLE_EQ(result.at({1, 1}), 154.0); // 4*8 + 5*10 + 6*12
  }
}

TEST(TensorTest, MatrixVectorProduct) {
  cudaq::tensor<double> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3});
  cudaq::tensor<double> v({7., 8., 9.}, {3});

  auto result = a.dot(v);

  EXPECT_EQ(result.get_rank(), 1);
  EXPECT_EQ(result.shape()[0], 2);

  EXPECT_DOUBLE_EQ(result.at({0}), 50.0);  // 1*7 + 2*8 + 3*9
  EXPECT_DOUBLE_EQ(result.at({1}), 122.0); // 4*7 + 5*8 + 6*9
}

TEST(TensorTest, MatrixTranspose) {
  cudaq::tensor<double> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3});

  auto result = a.transpose();

  EXPECT_EQ(a.get_rank(), 2);
  EXPECT_EQ(a.shape()[0], 2);
  EXPECT_EQ(a.shape()[1], 3);

  EXPECT_EQ(result.get_rank(), 2);
  EXPECT_EQ(result.shape()[0], 3);
  EXPECT_EQ(result.shape()[1], 2);

  EXPECT_DOUBLE_EQ(a.at({0, 0}), 1.0);
  EXPECT_DOUBLE_EQ(a.at({0, 1}), 2.0);
  EXPECT_DOUBLE_EQ(a.at({0, 2}), 3.0);
  EXPECT_DOUBLE_EQ(a.at({1, 0}), 4.0);
  EXPECT_DOUBLE_EQ(a.at({1, 1}), 5.0);
  EXPECT_DOUBLE_EQ(a.at({1, 2}), 6.0);

  EXPECT_DOUBLE_EQ(result.at({0, 0}), 1.0);
  EXPECT_DOUBLE_EQ(result.at({0, 1}), 4.0);
  EXPECT_DOUBLE_EQ(result.at({1, 0}), 2.0);
  EXPECT_DOUBLE_EQ(result.at({1, 1}), 5.0);
  EXPECT_DOUBLE_EQ(result.at({2, 0}), 3.0);
  EXPECT_DOUBLE_EQ(result.at({2, 1}), 6.0);
}

// Test error conditions
TEST(TensorTest, MismatchedShapeAddition) {
  cudaq::tensor<double> a({2, 2});
  cudaq::tensor<double> b({2, 3});

  EXPECT_THROW(a + b, std::runtime_error);
}

TEST(TensorTest, InvalidDotProductDimensions) {
  cudaq::tensor<double> a({2, 3});
  cudaq::tensor<double> b({2, 2});

  EXPECT_THROW(a.dot(b), std::runtime_error);
}

TEST(TensorTest, InvalidMatrixVectorDimensions) {
  cudaq::tensor<double> a({2, 3});
  cudaq::tensor<double> v({2});

  EXPECT_THROW(a.dot(v), std::runtime_error);
}

TEST(TensorTest, ConstructorWithShape) {
  std::vector<std::size_t> shape = {2, 3, 4};
  cudaq::tensor t(shape);

  EXPECT_EQ(t.get_rank(), 3);
  EXPECT_EQ(t.get_num_elements(), 24);
  EXPECT_EQ(t.shape(), shape);
}

TEST(TensorTest, ConstructorWithDataAndShape) {
  std::vector<std::size_t> shape = {2, 2};
  std::vector<std::complex<double>> data(4);
  data[0] = {1.0, 0.0};
  data[1] = {0.0, 1.0};
  data[2] = {0.0, -1.0};
  data[3] = {1.0, 0.0};

  cudaq::tensor t(data, shape);

  EXPECT_EQ(t.get_rank(), 2);
  EXPECT_EQ(t.get_num_elements(), 4);
  EXPECT_EQ(t.shape(), shape);

  // Check if data is correctly stored
  EXPECT_EQ(t.at({0, 0}), std::complex<double>(1.0, 0.0));
  EXPECT_EQ(t.at({0, 1}), std::complex<double>(0.0, 1.0));
  EXPECT_EQ(t.at({1, 0}), std::complex<double>(0.0, -1.0));
  EXPECT_EQ(t.at({1, 1}), std::complex<double>(1.0, 0.0));
}

TEST(TensorTest, AccessElements) {
  std::vector<std::size_t> shape = {2, 3};
  cudaq::tensor t(shape);

  // Set values
  t.at({0, 0}) = {1.0, 0.0};
  t.at({0, 1}) = {0.0, 1.0};
  t.at({1, 2}) = {-1.0, 0.0};

  // Check values
  EXPECT_EQ(t.at({0, 0}), std::complex<double>(1.0, 0.0));
  EXPECT_EQ(t.at({0, 1}), std::complex<double>(0.0, 1.0));
  EXPECT_EQ(t.at({1, 2}), std::complex<double>(-1.0, 0.0));
}

TEST(TensorTest, InvalidAccess) {
  std::vector<std::size_t> shape = {2, 2};
  cudaq::tensor t(shape);

  EXPECT_THROW(t.at({2, 0}), std::runtime_error);
  EXPECT_THROW(t.at({0, 2}), std::runtime_error);
  EXPECT_THROW(t.at({0, 0, 0}), std::runtime_error);
}
