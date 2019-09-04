#include <iostream>
#include <numeric>
#include <pybind11/pybind11.h>
#include <xtensor/xmath.hpp>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;

#include "/Users/miguel/Documents/python apps/py3.6/dlgork/Metal_py/metal/cpp/tensor/tensor.cpp"

int add(int i, int j) {
    std::cout << "/* message */" << '\n';
    return i + j;
}


double sum_of_sines(xt::pyarray<double>& m)
{
    auto sines = xt::sin(m);  // sines does not actually hold values.
    return std::accumulate(sines.cbegin(), sines.cend(), 0.0);
}

Eigen::MatrixXd  scale_by_2(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>v, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>c) {
    Eigen::MatrixXd p = v*c;
    return p;
}

//auto matmul(py::array_t<double>& m,py::array_t<double>& k)
//{
  //auto p = __pythran_functions::F_mat_mul()(m,k);
  //return p;
//}


PYBIND11_MODULE(_mod1, m)
{



  py::class_<Tensor>(m, "Tensor")
  .def(py::init< py::array_t<double> &, const py::bool_ &, const  py::list &, const  py::int_ &>(),
       py::arg("_data"),
       py::arg("_requires_grad")=py::bool_(false),
       py::arg("_depends_on")=py::list{},
       py::arg("_id")=py::int_()
       )
  .def_readwrite("data", &Tensor::data)
  .def_readwrite("requires_grad", &Tensor::requires_grad)
  .def_readwrite("depends_on", &Tensor::depends_on)
  .def_readwrite("id", &Tensor::id);

  m.def("scale_by_2", scale_by_2, "scale_by_2");


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
