CSRC=main.cc
CUSRC=wmma.cu
OBJS=$(CSRC:.cc=.o) $(CUSRC:.cu=.o)
TARGET=wmma
CXXFLAGS=-fopenmp -g -O2 -I/usr/local/cuda/include -DGL_GLEXT_PROTOTYPES -march=native -mavx2 -ftree-vectorize
NVFLAGS=-g -O2 -arch=compute_70 -code=sm_70 --compiler-options="-fopenmp -march=native -mavx2 -ftree-vectorize"
NVCC=/usr/local/cuda/bin/nvcc
LIBS=-L/usr/lib/nvidia-396 -lGL -lGLU -lX11 -lglut

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVFLAGS) -cudart=shared -o $(TARGET) $(OBJS) $(LIBS)

main.o: main.cc header.h
	$(CXX) $(CXXFLAGS) -c main.cc

wmma.o: wmma.cu header.h
	$(NVCC) $(NVFLAGS) -c wmma.cu

clean:
	-rm $(TARGET) $(OBJS)

