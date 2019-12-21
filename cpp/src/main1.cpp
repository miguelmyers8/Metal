#include <iostream>
#include <numeric>                        // Standard library import for std::accumulate
#include <pybind11/pybind11.h>            // Pybind11 import to define Python bindings
#define FORCE_IMPORT_ARRAY                // numpy C api loading

using namespace std;
namespace py = pybind11;


int sub(int i, int j) {
    std::cout << "/* message */" << '\n';
    return i - j;
}




PYBIND11_MODULE(_mod2, m)
{


    m.def("sub", sub, "subtract");


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
