#define PY_MAJOR_VERSION 3
#undef ENABLE_PYTHON_MODULE
#include <pythonic/core.hpp>
#include <pythonic/python/core.hpp>
#include <pythonic/types/bool.hpp>
#include <pythonic/types/int.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pythonic/include/types/dict.hpp>
#include <pythonic/include/types/float32.hpp>
#include <pythonic/include/types/float64.hpp>
#include <pythonic/include/types/ndarray.hpp>
#include <pythonic/include/types/str.hpp>
#include <pythonic/include/types/numpy_texpr.hpp>
#include <pythonic/types/numpy_texpr.hpp>
#include <pythonic/types/float32.hpp>
#include <pythonic/types/dict.hpp>
#include <pythonic/types/float64.hpp>
#include <pythonic/types/ndarray.hpp>
#include <pythonic/types/str.hpp>
namespace __pythran_functions
{
  struct D1_mat_mul
  {
    typedef void callable;
    typedef void pure;
    template <typename argument_type0 >
    struct type
    {
      typedef typename pythonic::returnable<typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type>::type result_type;
    }  
    ;
    template <typename argument_type0 >
    typename type<argument_type0>::result_type operator()(argument_type0&& a) const
    ;
  }  ;
  struct F_mat_mul
  {
    typedef void callable;
    typedef void pure;
    template <typename argument_type0 , typename argument_type1 >
    struct type
    {
      typedef typename pythonic::returnable<typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type>::type result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 >
    typename type<argument_type0, argument_type1>::result_type operator()(argument_type0&& a, argument_type1&& b) const
    ;
  }  ;
  template <typename argument_type0 >
  typename D1_mat_mul::type<argument_type0>::result_type D1_mat_mul::operator()(argument_type0&& a) const
  {
    return a;
  }
  template <typename argument_type0 , typename argument_type1 >
  typename F_mat_mul::type<argument_type0, argument_type1>::result_type F_mat_mul::operator()(argument_type0&& a, argument_type1&& b) const
  {
    return b;
  }
}
#include <pythonic/python/exception_handler.hpp>
#ifdef ENABLE_PYTHON_MODULE
typename __pythran_functions::D1_mat_mul::type<pythonic::types::dict<pythonic::types::str,pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>>>::result_type D1_mat_mul0(pythonic::types::dict<pythonic::types::str,pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>>&& a) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_functions::D1_mat_mul()(a);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran_functions::D1_mat_mul::type<pythonic::types::dict<pythonic::types::str,pythonic::types::ndarray<float,pythonic::types::pshape<long,long,long>>>>::result_type D1_mat_mul1(pythonic::types::dict<pythonic::types::str,pythonic::types::ndarray<float,pythonic::types::pshape<long,long,long>>>&& a) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_functions::D1_mat_mul()(a);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran_functions::F_mat_mul::type<pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>>::result_type F_mat_mul0(pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>&& a, pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>&& b) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_functions::F_mat_mul()(a, b);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran_functions::F_mat_mul::type<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>::result_type F_mat_mul1(pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>&& a, pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>&& b) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_functions::F_mat_mul()(a, b);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran_functions::F_mat_mul::type<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>::result_type F_mat_mul2(pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>&& a, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>&& b) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_functions::F_mat_mul()(a, b);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran_functions::F_mat_mul::type<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>, pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>::result_type F_mat_mul3(pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>&& a, pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>&& b) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_functions::F_mat_mul()(a, b);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran_functions::F_mat_mul::type<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>::result_type F_mat_mul4(pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>&& a, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>&& b) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_functions::F_mat_mul()(a, b);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran_functions::F_mat_mul::type<pythonic::types::ndarray<float,pythonic::types::pshape<long,long,long>>, pythonic::types::ndarray<float,pythonic::types::pshape<long,long,long>>>::result_type F_mat_mul5(pythonic::types::ndarray<float,pythonic::types::pshape<long,long,long>>&& a, pythonic::types::ndarray<float,pythonic::types::pshape<long,long,long>>&& b) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_functions::F_mat_mul()(a, b);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran_functions::F_mat_mul::type<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>, pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>::result_type F_mat_mul6(pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>&& a, pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>&& b) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_functions::F_mat_mul()(a, b);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran_functions::F_mat_mul::type<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>, pythonic::types::numpy_texpr<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>>::result_type F_mat_mul7(pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>&& a, pythonic::types::numpy_texpr<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>&& b) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_functions::F_mat_mul()(a, b);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran_functions::F_mat_mul::type<pythonic::types::numpy_texpr<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>, pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>::result_type F_mat_mul8(pythonic::types::numpy_texpr<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>&& a, pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>&& b) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_functions::F_mat_mul()(a, b);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran_functions::F_mat_mul::type<pythonic::types::numpy_texpr<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>, pythonic::types::numpy_texpr<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>>::result_type F_mat_mul9(pythonic::types::numpy_texpr<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>&& a, pythonic::types::numpy_texpr<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>&& b) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_functions::F_mat_mul()(a, b);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}

static PyObject *
__pythran_wrap_D1_mat_mul0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[1+1];
    char const* keywords[] = {"a", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "O",
                                     (char**)keywords, &args_obj[0]))
        return nullptr;
    if(is_convertible<pythonic::types::dict<pythonic::types::str,pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>>>(args_obj[0]))
        return to_python(D1_mat_mul0(from_python<pythonic::types::dict<pythonic::types::str,pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>>>(args_obj[0])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_D1_mat_mul1(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[1+1];
    char const* keywords[] = {"a", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "O",
                                     (char**)keywords, &args_obj[0]))
        return nullptr;
    if(is_convertible<pythonic::types::dict<pythonic::types::str,pythonic::types::ndarray<float,pythonic::types::pshape<long,long,long>>>>(args_obj[0]))
        return to_python(D1_mat_mul1(from_python<pythonic::types::dict<pythonic::types::str,pythonic::types::ndarray<float,pythonic::types::pshape<long,long,long>>>>(args_obj[0])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_F_mat_mul0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[2+1];
    char const* keywords[] = {"a","b", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OO",
                                     (char**)keywords, &args_obj[0], &args_obj[1]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>>(args_obj[1]))
        return to_python(F_mat_mul0(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>>(args_obj[0]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>>(args_obj[1])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_F_mat_mul1(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[2+1];
    char const* keywords[] = {"a","b", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OO",
                                     (char**)keywords, &args_obj[0], &args_obj[1]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[1]))
        return to_python(F_mat_mul1(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[0]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[1])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_F_mat_mul2(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[2+1];
    char const* keywords[] = {"a","b", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OO",
                                     (char**)keywords, &args_obj[0], &args_obj[1]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[0]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[1]))
        return to_python(F_mat_mul2(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[0]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[1])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_F_mat_mul3(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[2+1];
    char const* keywords[] = {"a","b", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OO",
                                     (char**)keywords, &args_obj[0], &args_obj[1]))
        return nullptr;
    if(is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[1]))
        return to_python(F_mat_mul3(from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[0]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[1])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_F_mat_mul4(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[2+1];
    char const* keywords[] = {"a","b", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OO",
                                     (char**)keywords, &args_obj[0], &args_obj[1]))
        return nullptr;
    if(is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[0]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[1]))
        return to_python(F_mat_mul4(from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[0]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[1])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_F_mat_mul5(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[2+1];
    char const* keywords[] = {"a","b", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OO",
                                     (char**)keywords, &args_obj[0], &args_obj[1]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<float,pythonic::types::pshape<long,long,long>>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<float,pythonic::types::pshape<long,long,long>>>(args_obj[1]))
        return to_python(F_mat_mul5(from_python<pythonic::types::ndarray<float,pythonic::types::pshape<long,long,long>>>(args_obj[0]), from_python<pythonic::types::ndarray<float,pythonic::types::pshape<long,long,long>>>(args_obj[1])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_F_mat_mul6(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[2+1];
    char const* keywords[] = {"a","b", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OO",
                                     (char**)keywords, &args_obj[0], &args_obj[1]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>(args_obj[1]))
        return to_python(F_mat_mul6(from_python<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>(args_obj[0]), from_python<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>(args_obj[1])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_F_mat_mul7(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[2+1];
    char const* keywords[] = {"a","b", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OO",
                                     (char**)keywords, &args_obj[0], &args_obj[1]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>(args_obj[0]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>>(args_obj[1]))
        return to_python(F_mat_mul7(from_python<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>(args_obj[0]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>>(args_obj[1])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_F_mat_mul8(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[2+1];
    char const* keywords[] = {"a","b", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OO",
                                     (char**)keywords, &args_obj[0], &args_obj[1]))
        return nullptr;
    if(is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>(args_obj[1]))
        return to_python(F_mat_mul8(from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>>(args_obj[0]), from_python<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>(args_obj[1])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_F_mat_mul9(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[2+1];
    char const* keywords[] = {"a","b", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OO",
                                     (char**)keywords, &args_obj[0], &args_obj[1]))
        return nullptr;
    if(is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>>(args_obj[0]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>>(args_obj[1]))
        return to_python(F_mat_mul9(from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>>(args_obj[0]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<float,pythonic::types::pshape<long,long>>>>(args_obj[1])));
    else {
        return nullptr;
    }
}

            static PyObject *
            __pythran_wrapall_D1_mat_mul(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_D1_mat_mul0(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_D1_mat_mul1(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "D1_mat_mul", "\n    - D1_mat_mul(str:float64[:,:,:] dict)\n    - D1_mat_mul(str:float32[:,:,:] dict)", args, kw);
                });
            }


            static PyObject *
            __pythran_wrapall_F_mat_mul(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_F_mat_mul0(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_F_mat_mul1(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_F_mat_mul2(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_F_mat_mul3(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_F_mat_mul4(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_F_mat_mul5(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_F_mat_mul6(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_F_mat_mul7(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_F_mat_mul8(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_F_mat_mul9(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "F_mat_mul", "\n    - F_mat_mul(float64[:,:,:], float64[:,:,:])\n    - F_mat_mul(float64[:,:], float64[:,:])\n    - F_mat_mul(float32[:,:,:], float32[:,:,:])\n    - F_mat_mul(float32[:,:], float32[:,:])", args, kw);
                });
            }


static PyMethodDef Methods[] = {
    {
    "D1_mat_mul",
    (PyCFunction)__pythran_wrapall_D1_mat_mul,
    METH_VARARGS | METH_KEYWORDS,
    "Supported prototypes:\n\n    - D1_mat_mul(str:float64[:,:,:] dict)\n    - D1_mat_mul(str:float32[:,:,:] dict)"},{
    "F_mat_mul",
    (PyCFunction)__pythran_wrapall_F_mat_mul,
    METH_VARARGS | METH_KEYWORDS,
    "Supported prototypes:\n\n    - F_mat_mul(float64[:,:,:], float64[:,:,:])\n    - F_mat_mul(float64[:,:], float64[:,:])\n    - F_mat_mul(float32[:,:,:], float32[:,:,:])\n    - F_mat_mul(float32[:,:], float32[:,:])"},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "functions",            /* m_name */
    "",         /* m_doc */
    -1,                  /* m_size */
    Methods,             /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
  };
#define PYTHRAN_RETURN return theModule
#define PYTHRAN_MODULE_INIT(s) PyInit_##s
#else
#define PYTHRAN_RETURN return
#define PYTHRAN_MODULE_INIT(s) init##s
#endif
PyMODINIT_FUNC
PYTHRAN_MODULE_INIT(functions)(void)
#ifndef _WIN32
__attribute__ ((visibility("default")))
__attribute__ ((externally_visible))
#endif
;
PyMODINIT_FUNC
PYTHRAN_MODULE_INIT(functions)(void) {
    import_array()
    #if PY_MAJOR_VERSION >= 3
    PyObject* theModule = PyModule_Create(&moduledef);
    #else
    PyObject* theModule = Py_InitModule3("functions",
                                         Methods,
                                         ""
    );
    #endif
    if(! theModule)
        PYTHRAN_RETURN;
    PyObject * theDoc = Py_BuildValue("(sss)",
                                      "0.9.2",
                                      "2019-08-18 00:10:22.397841",
                                      "ef9346ef8afcb5903d6152e8b89ffad4fb28d84bd0cc11eaad29f3aee0861c62");
    if(! theDoc)
        PYTHRAN_RETURN;
    PyModule_AddObject(theModule,
                       "__pythran__",
                       theDoc);


    PYTHRAN_RETURN;
}

#endif