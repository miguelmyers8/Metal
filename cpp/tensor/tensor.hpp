
//  Tensor.hpp
//  metal
//
//  Created by miguel myers on 5/15/19.
//  Copyright © 2019 miguel myers. All rights reserved.
//

#ifndef Tensor_hpp
#define Tensor_hpp

#include <stdio.h>
#include <iostream>
#include <boost/variant/variant.hpp>
#include <tuple>
#include <vector>
#include <optional>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

using namespace std;
namespace py = pybind11;


class Tensor;
class Dependancies;


class Tensor {
    
public:
    //var
    py::array_t<double> data;
    py::bool_ requires_grad;
    py::list depends_on;
    py::int_ id;

   //functions
    Tensor(py::array_t<double>& _data,
           const py::bool_ & _requires_grad = py::bool_(false),
           const py::list & _depends_on = py::list{},
           const py::int_ & _id = py::none()
           );
    
    

};


class Dependancies{
    
public:
        
    string name;

};

#endif /* Tensor_hpp */
