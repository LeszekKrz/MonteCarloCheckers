all: MonteCarloCheckers 

MonteCarloCheckers: MonteCarloCheckers.cu
	nvcc -gencode=arch=compute_75,code=\"sm_75,compute_75\" MonteCarloCheckers.cu -o MonteCarloCheckers

clean:
	rm MonteCarloCheckers

.PHONY: clean
