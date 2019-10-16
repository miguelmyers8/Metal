# cpp
THIS PART OF THE PACKAGE NEEDS CLEANING. IGNORE FOR NOW.

## CMakeLists
Cmake must know about each one of your pybind module.<br>
You can have multiple of these with different names.<br>

pybind11_add_module(<target name here> path/to/pybind11_module)
pybind11_add_module(<target name here> path/to/pybind11_module)

Then link any library you want to use.<bar>
target_link_libraries(<target name here> ${LIBS})<bar>
target_link_libraries(<target name here> ${LIBS})<bar>

## How to add c++ code
Add c++ code in cpp/lib, and include it in src/main.cpp or src/main1.cpp or create on yourself with a PYBIND11_MODULE() function.<br>
Your code will be import under this name in python.<br>
It must be the same name you used in your CMakeLists.txt file.

pybind will call your code under the name you give it.<br>
for example in main.cpp PYBIND11_MODULE(<name here>, m)<br>

lib/<br>
  foo.cpp<br>
  bar.cpp<br>
src/<br>
  main.cpp  <--import foo.cpp in here<br>
  main1.cpp <--import bar.cpp in here<br>


##setup.py
In setup this is how cmake tells python to import your code<br>
CMakeExtension('cpp.<name here>'),CMakeExtension('cpp.<name here>')<br>
