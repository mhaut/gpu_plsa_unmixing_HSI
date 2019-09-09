default: base

base:
	g++ plsa_omp.cpp -o plsa_omp.bin -fopenmp

optimized:
	g++ plsa_omp.cpp -o plsa_omp_optimized.bin -O3 -fopenmp -ffast-math

clean:
	-rm -f  *.bin
