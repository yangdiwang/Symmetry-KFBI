# Third-party dependencies

## Eigen 3.4.0

Eigen is not copied into this repository. CMake first searches for an installed
Eigen 3.4 package and otherwise downloads the fixed `3.4.0` release from the
upstream Eigen repository. Eigen's own source distribution contains its full
license notices.

## zFFT

The zFFT sources used by the KFBI bulk solver are included under
`third_party/zfft`. The source headers identify Han Zhou as the creator. No
separate license file was present alongside the imported zFFT snapshot; users
who redistribute it should confirm the applicable terms with its author.
