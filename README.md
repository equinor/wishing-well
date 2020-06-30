# wishing-well
Team repo for Wishing Well team - Virtual summer internship 2020


This is a C++ library for computing shortest paths with higher-order properties like curvature and torsion taken into account. It implements the algorithms of our ICCV 2013 paper [1] and PAMI 2015 paper [2].

[![Build Status](https://travis-ci.org/PetterS/curve_extraction.png)](https://travis-ci.org/PetterS/curve_extraction)

Compilation
====
The following is required to compile the library:
* C++ compiler.
* CMake
* Spii installed; see https://github.com/PetterS/spii 

All tests pass with the following compilers:
* Visual Studio 2013
* GCC 4.8 (Cygwin)
* GCC 4.7 (Ubuntu)

Earlier compilers might not work.

### Linux ###


* Spii
```
git clone https://github.com/PetterS/spii
cd spii
mkdir build
cd build
cmake ..
make && make test
sudo make install
```

* Curve_extraction
```
git clone https://github.com/PetterS/curve_extraction
cd curve_extraction
mkdir build
cd build
cmake ..
make && make test
sudo make install
```

Matlab
------

Paths can be changed in compile.m.

### Linux ###
1. Install curve_extraction and spii using instruction above.
2. Install Eigen3 in /usr/local/include/eigen3.

You can now run /matlab/examples/simple_3D.m all files compiles on demand.

### Windows ###
1. Use CMake to build SPII and curve_extraction. (if building curve_extraction throws error for sqrt, add the following line  in the header curve_extraction/source/mesh.cpp and curve_extraction/source/grid_mesh.cpp "#include \<cmath\>")
2. Copy spii\include to C:\Program Files\SPII\include
3. Copy \<spii-build-path\>\bin\spii.dll to C:\Program Files\SPII\lib
4. Copy \<spii-build-path\>\lib\Release\\* to C:\Program Files\SPII\lib
5. Copy curve_extraction\include\ to  C:\Program Files\curve_extraction\include
6. Copy \<curve_extraction-build-path\>\lib\ to C:\Program Files\curve_extraction\lib
7. Download and put all [Eigen 3](http://eigen.tuxfamily.org/) headers in C:\Program Files\Eigen
8. If your version of MATLAB does not support Visual Studio 2013 follow the instructions [http://www.mathworks.com/matlabcentral/fileexchange/44408-matlab-mex-support-for-visual-studio-2013-and-mbuild](http://www.mathworks.com/matlabcentral/fileexchange/44408-matlab-mex-support-for-visual-studio-2013-and-mbuild).
9. Add C:\Program Files\SPII\lib to your system path.

You can now run /matlab/examples/simple_3D.m all files compiles on demand.

Video
====
 * http://www.youtube.com/watch?v=9qjQj3I5pBc

References
====
1. Petter Strandmark, Johannes Ulén, Fredrik Kahl, Leo Grady. [Shortest Paths with Curvature and Torsion](http://www2.maths.lth.se/vision/publications/publications/view_paper.php?paper_id=582). International Conference on Computer Vision. 2013.

2. Johannes Ulén, Petter Strandmark, Fredrik Kahl [Shortest Paths with Higher-Order Regularization](http://www2.maths.lth.se/vision/publications/publications/view_paper.php?paper_id=623). IEEE Transactions on Pattern Analysis and Machine Intelligence. 2015
