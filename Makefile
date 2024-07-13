
big-num.out: ./src/big-num.cu
	nvcc -I ./include -arch=sm_80 ./src/big-num.cu -o big-num.out -lgmp
	
GZKP-NTT.out: ./src/GZKP-NTT.cu
	nvcc -I ./include -arch=sm_80 ./src/GZKP-NTT.cu -o NTT.out -lgmp


clean:
	rm -f NTT.out



