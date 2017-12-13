all : lfe_stack_cpu lfe_stack_gpu

lfe_stack_cpu: lfe_stacking.c
	gcc -o lfe_stack_cpu lfe_stacking.c -lfftw3 -lm -m64

lfe_stack_gpu: lfe_stacking_gpu.cu
	module load cuda; nvcc -o lfe_stack_gpu lfe_stacking_gpu.cu -lfftw3 -lcufft -m64

.PHONY: clean
clean:
	rm -f lfe_stack_cpu lfe_stack_gpu
