#ifndef HEADER_H_
#define HEADER_H_

#define NN (100 * 24 * 24)
#define MM (30)
#define KK (25)

#define EL(x, y, Width) ((y) * (Width) + (x))
#define SIZE(Width, Height) (Width * Height)

#define checkError(x) \
 { \
   cudaError_t et = (x); \
   if (et != cudaSuccess) { \
     std::cout << __FILE__ << ":(" << __LINE__ << "): " << cudaGetErrorString(et) << std::endl; \
     exit(1); \
   } \
 }
void calc_fma(float c[], half a[], half b[], int n, int m, int k, bool bTensorCore);

#endif // HEADER_H_

