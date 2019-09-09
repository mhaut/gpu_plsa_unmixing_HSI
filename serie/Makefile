default: base

base:
	g++ plsa.cpp -o plsa.bin -O1 -xAVX -fopenmp

optimized:
	g++ plsa.cpp -o plsa_optimized.bin -O3 -xAVX -fopenmp

clean:
	-rm -f  *.bin
