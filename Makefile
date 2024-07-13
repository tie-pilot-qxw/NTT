
build:
	nvcc -I ./include -arch=sm_80 ./src/GZKP-NTT.cu -o NTT.out

clean:
	rm -f NTT.out



