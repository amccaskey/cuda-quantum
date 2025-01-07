
#include "cudaq/operator.h"
#include "cudaq/utils/cudaq_utils.h"

#include <cassert>

using namespace cudaq::experimental;

int main() {

  {
    // Can build composite operations as usual
    auto H = 5.907 - 2.1433 * spin::x(0) * spin::x(1) -
             2.1433 * spin::y(0) * spin::y(1) + .21829 * spin::z(0) -
             6.125 * spin::z(1);
    H.dump();
    // Can be mapped to matrices
    // FIXME not implemented yet.
    // auto matrix = H.to_matrix();
  }

  {

    // Define a coefficient functor
    auto f_t = [](const parameter_map &parameters) {
      return 2. * parameters.at("t") + 3. * parameters.at("omega");
    };

    // Can multiply spin ops by coefficient functors
    auto H_t = f_t * spin::x(2);
    H_t.dump();

    // These ops are considered "templates"
    assert(H_t.is_template());

    // Concretize this template from constant parameters.
    for (auto t : cudaq::linspace(-M_PI, M_PI, 10)) {
      auto concrete_H_t = H_t({{"t", t}, {"omega", t / 3.}});
      assert(!concrete_H_t.is_template());
      assert(H_t.is_template());

      concrete_H_t.dump();
      assert(std::fabs(2. * t + 3. * (t / 3.) -
                       concrete_H_t.get_coefficient().real()) < 1e-3);
    }
  }

  {
    auto n1 = particle::create(1) * particle::annihilate(1);
    assert(1 == n1.num_terms());

    auto n2 = particle::annihilate(1) * particle::create(1);
    n2.dump(); // Look, is it normal ordered?
    // should get a -1 sign flip
    assert(n2.get_coefficient() == -1.0);

    // Build hopping term a†_i a_j
    auto hop = particle::create(1) * particle::annihilate(2);
    assert(1 == hop.num_terms());

    // Combine terms
    auto hamiltonian = n1 + hop;
    assert(2 == hamiltonian.num_terms());
  }

  {
    // Define a coefficient functor
    auto f_t = [](const parameter_map &parameters) {
      return 2. * parameters.at("t") + 3. * parameters.at("omega");
    };

    // Can multiply spin ops by coefficient functors
    auto H_t = f_t * particle::create(2) * particle::annihilate(2);
    H_t.dump();

    // These ops are considered "templates"
    assert(H_t.is_template());

    // Concretize this template from constant parameters.
    for (auto t : cudaq::linspace(-M_PI, M_PI, 10)) {
      auto concrete_H_t = H_t({{"t", t}, {"omega", t / 3.}});
      assert(!concrete_H_t.is_template());
      assert(H_t.is_template());

      concrete_H_t.dump();
      assert(std::fabs(2. * t + 3. * (t / 3.) -
                       concrete_H_t.get_coefficient().real()) < 1e-3);
    }
  }

  {
    // Operators from user-defined matrices

    cudaq::matrix_2 x_mat({0., 1., 1., 0.});
    operator_matrix z_mat({1.0, 0.0, 0.0, -1.0}, {2, 2});

    auto xop = from_matrix(x_mat, {0});
    auto zop = from_matrix(z_mat, {1});
    auto mult = xop * zop;

    // Each term in the matrix operator is a product of operators on unique
    // sites We can get all the operator matrices for each site in a term. Note
    // this will throw if the operator has multiple terms (summed).
    auto support_matrices = mult.get_elementary_operators();
    assert(support_matrices.size() == 2);
    for (auto &m : support_matrices)
      printf("M = %s\n", m.dump().c_str());
  }

  {
    // Can multiply different types, coming soon
    auto n1 = particle::create(1) * particle::annihilate(1);
    auto z = spin::z(2);
    auto H = n1 + spin::z(2);
    // H.dump();

    // printf("%s\n", H.to_matrix().dump().c_str());
  }

  {
    // Elementary operators...
    // auto p2 = fermion::position(2);
    // p2.dump();
    // auto m3 = fermion::momentum(3);
    // m3.dump();
  }
}