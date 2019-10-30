#include <iostream>
#include <numeric>                        // Standard library import for std::accumulate
#include <pybind11/pybind11.h>            // Pybind11 import to define Python bindings
#include <xtensor/xmath.hpp>              // xtensor import for the C++ universal functions
#define FORCE_IMPORT_ARRAY                // numpy C api loading
#include <xtensor-python/pyarray.hpp>     // Numpy bindings
#include <xtensor-blas/xlinalg.hpp>
using namespace std;
namespace py = pybind11;


int sub(int i, int j) {
    std::cout << "/* message */" << '\n';
    return i - j;
}

auto mul(xt::pyarray<double> m, xt::pyarray<double> k)
{
    xt::pyarray<double> o =  m*k;
    return o;

}


double sum_of_sines(xt::pyarray<double>& m)
{
    auto sines = xt::sin(m);  // sines does not actually hold values.
    return std::accumulate(sines.cbegin(), sines.cend(), 0.0);
}

auto matmul(xt::pyarray<double>& m,xt::pyarray<double>& k)
{

    return xt::linalg::dot(m,k);
}

PYBIND11_MODULE(_mod2, m)
{
    xt::import_numpy();
    m.doc() = "Test module for xtensor python bindings";

    m.def("sum_of_sines", sum_of_sines, "Sum the sines of the input values");

    m.def("matmul", matmul, "matrixmul");

    m.def("sub", sub, "subtract");

    m.def("mul", mul, "m");


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
