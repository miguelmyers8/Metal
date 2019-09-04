cmake_minimum_required(VERSION 2.8.12)
project(cpp)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

add_definitions(-DHAVE_CBLAS=1)

set (CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cpp/cmake")


if (WIN32)
    find_package(OpenBLAS REQUIRED)
    set(BLAS_LIBRARIES ${CMAKE_INSTALL_PREFIX}${OpenBLAS_LIBRARIES})
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})

find_package(xtl REQUIRED)
message(STATUS "Found xtl: ${xtl_INCLUDE_DIRS}/xtl")
find_package(xtensor-python)
message(STATUS "Found xtensor-python: ${xtensor-python_INCLUDE_DIRS}/xtensor-python")
find_package(xtensor REQUIRED)
message(STATUS "Found xtensor: ${xtensor_INCLUDE_DIRS}/xtensor")
find_package(pybind11 REQUIRED)
message(STATUS "Found pybind11: ${pybind11_INCLUDE_DIRS}/pybind11")
find_package(NumPy REQUIRED)
message(STATUS "Found numpy: ${NUMPY_INCLUDE_DIRS}")
find_package(Pythran REQUIRED)
message(STATUS "Found Pythran: ${Pythran_INCLUDE_DIRS}")

find_package(Eigen3 REQUIRED)
message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")

pybind11_add_module(_mod1 cpp/src/main.cpp)
pybind11_add_module(_mod2 cpp/src/main1.cpp)

include_directories(
${EIGEN3_INCLUDE_DIR}
${xtl_INCLUDE_DIRS}
${xtensor_INCLUDE_DIRS}
${xtensor-python_INCLUDE_DIRS}
${NUMPY_INCLUDE_DIRS}
${Pythran_INCLUDE_DIRS}
${pybind11_INCLUDE_DIRS}
)
set(LIBS ${LIBS} PUBLIC ${xtensor-python_LIBRARIES} )
set(LIBS ${LIBS} ${BLAS_LIBRARIES}  )
set(LIBS ${LIBS} ${LAPACK_LIBRARIES}  )


target_link_libraries(_mod1 ${LIBS})
target_link_libraries(_mod2 ${LIBS})

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()


message(${CMAKE_CXX_FLAGS_RELEASE})
