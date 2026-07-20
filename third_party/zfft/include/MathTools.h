#pragma once
// Minimal MathTools shim: only Log2 is used by zfft.cc.
inline int Log2(int m)
{
    int k = 0;
    while (m > 1) { m >>= 1; k++; }
    return k;
}
