#include <iostream>
#include <numeric>
#include <pybind11/pybind11.h>

#define FORCE_IMPORT_ARRAY

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;


int add(int i, int j) {
    std::cout << "/* message */" << '\n';
    return i + j;
}



std::vector<Eigen::MatrixXd>  scale_by_2(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>v, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>c) {
  std::vector<Eigen::MatrixXd> matrices;

    Eigen::MatrixXd p = v *c;
     matrices.push_back(p);
    return matrices;
}

//auto matmul(py::array_t<double>& m,py::array_t<double>& k)
//{
  //auto p = __pythran_functions::F_mat_mul()(m,k);
  //return p;
//}


PYBIND11_MODULE(_mod1, m)
{







  m.def("scale_by_2", scale_by_2, "scale_by_2");


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
