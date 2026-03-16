CXX = hipcc

ARCH ?= gfx90a

CXXFLAGS = --std=c++20 --offload-arch=$(ARCH) \
           -I/opt/rocm-7.2.0/include \
           -I/opt/rocm-7.2.0/include/hipsparse \
           -I/opt/rocm-7.2.0/include/hipfft

LDFLAGS = -L/opt/rocm-7.2.0/lib
LIBS = -lhipblas -lhipsparse -lrocsparse

.PHONY: gemm clean

gemm: bin/gemm.out 

bin/gemm.out: gemm.cpp utils.cpp
	@mkdir -p bin
	$(CXX) $< -o $@ $(CXXFLAGS) $(LDFLAGS) $(LIBS)

