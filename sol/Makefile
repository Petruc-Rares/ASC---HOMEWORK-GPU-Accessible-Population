all: gpu_sol

gpu_sol: gpu_sol.cu
	nvcc -arch=sm_35 -rdc=true gpu_sol.cu -o gpu_sol -O0

clean:
	rm -f gpu_sol *.o
