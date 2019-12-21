//#include "functions1.cpp"
#include "tensor.hpp"

Tensor::Tensor(py::array_t<double>& _data,const  py::bool_ & _requires_grad,const py::list & _depends_on,const py::int_ & _id){
    this->data = _data;
    this->requires_grad = _requires_grad;
    this->depends_on = _depends_on;
    this->id = _id;
}
