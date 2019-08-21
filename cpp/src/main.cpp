#include <iostream>
#include <numeric>
#include "/Users/miguel/Documents/python apps/py3.6/dlgork/Metal_py/metal/cpp/tensor/functions.cpp"
#include <pybind11/pybind11.h>
#include <xtensor/xmath.hpp>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>
#include <xtensor-blas/xlinalg.hpp>

using namespace std;
namespace py = pybind11;


int add(int i, int j) {
    std::cout << "/* message */" << '\n';
    return i + j;
}


double sum_of_sines(xt::pyarray<double>& m)
{
    auto sines = xt::sin(m);  // sines does not actually hold values.
    return std::accumulate(sines.cbegin(), sines.cend(), 0.0);
}


auto matmul(py::array_t<double>& m,py::array_t<double>& k)
{
  auto p = __pythran_functions::F_mat_mul()(m,k);
  return p;
}


PYBIND11_MODULE(_mod1, m)
{
    xt::import_numpy();
    m.doc() = "Test module for xtensor python bindings";

    m.def("sum_of_sines", sum_of_sines, "Sum the sines of the input values");

    m.def("matmul", matmul, "matrixmul");


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}


int main(int argc, char const *argv[]) {
 add(3,4);
 return 0;
}
