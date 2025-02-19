all: page hmm uvm


page: page.cu
	nvcc -G -g $(DEFS) -Xcompiler "-fopenmp" $(DEFS) -Xptxas -v   -arch=sm_90   page.cu -o page
	#nvcc -Xcompiler $(DEFS) -Xptxas -v  -Xptxas -O0 -Xcicc -O0 -arch=sm_70 -O0  page.cu -o page
	#nvcc -Xcompiler "-fopenmp" $(DEFS) -Xptxas -v  -Xptxas -O0 -Xcicc -O0 -arch=sm_70 -O0  page.cu -o page

hmm: hmm.cu
	nvcc -G -g $(DEFS) -Xptxas -v -arch=sm_90 hmm.cu -o hmm

uvm: uvm.cu
	nvcc -G -g $(DEFS) -Xptxas -v -arch=sm_90 uvm.cu -o uvm

clean:
	rm -f page
	rm -f hmm
	rm -f uvm


#nvcc $(DEFS) -G -g -Xptxas -v -arch=sm_70 -O0  band.cu -o band
