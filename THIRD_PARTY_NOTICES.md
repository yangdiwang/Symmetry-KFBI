# Third-party dependencies

## Eigen 3.4.0

Eigen is not copied into this repository. CMake first searches for an installed
Eigen 3.4 package and otherwise downloads the fixed `3.4.0` release from the
upstream Eigen repository. Eigen's own source distribution contains its full
license notices.

## CGAL 5.6 or newer

CGAL is not copied into this repository. It is a required system dependency
when `KFBIM_BUILD_3D=ON` (the default), together with the Boost, GMP, and MPFR
packages selected by CGAL. CGAL's package includes its license information;
the applicable license can depend on which CGAL components are used.

## zFFT

The zFFT sources used by the KFBI bulk solver are included under
`third_party/zfft`. The source headers identify Han Zhou as the creator. No
separate license file was present alongside the imported zFFT snapshot; users
who redistribute it should confirm the applicable terms with its author.
