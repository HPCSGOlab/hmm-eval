#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h> 
#include <cuda.h>
#include "mmio.h"
#include "spmv.cuh"
#include "utils.cu"

int main(int argc, char ** argv){
    	char *mFile; 
    	float *SPvec, *SPout;
        double *DPvec, *DPout;
   	double time_taken;
		
    	MatrixInfo<float> * SPnewMat = (MatrixInfo<float> *) malloc(sizeof(MatrixInfo<float>));
        MatrixInfo<double> * DPnewMat = (MatrixInfo<double> *) malloc(sizeof(MatrixInfo<double>));
    	mFile = argv[1];
	//printf("Reading matrix from %s\n", mFile);
	
    	cudaEvent_t start, stop;
    	cudaEventCreate(&start);
    	cudaEventCreate(&stop);
	
	//capture the data movement
    	cudaEventRecord(start);


	// Load matrices as both single and double precision
	MatrixInfo<float> * SPmatrix = read_file<float>(mFile);
        MatrixInfo<double> * DPmatrix = read_file<double>(mFile);
	if(SPmatrix == NULL){
		printf("Error regarding matrix file.");
		return 1;
	}
        else
        {
          	//printf("Number of rows is %d\n",SPmatrix->M);
      		//printf("Number of columns is %d\n",SPmatrix->N);
		//printf("Number of non zeros is %d\n",SPmatrix->nz);
    		//printf("Average number of non zeros per row is %d\n\n",SPmatrix->nz/SPmatrix->M);
        }


	double gflop = 2 * (double) DPmatrix->nz / 1e9;
    	float milliseconds = 0;


	// Change from COO format to CSR format
        //printf("Changing sparse matrix format to CSR...\n");
	SPnewMat = transferMat<float>(SPmatrix);
        DPnewMat = transferMat<double>(DPmatrix);
	
        convert2CSR<float>(SPnewMat);
        convert2CSR<double>(DPnewMat);

	// Create dense vector in both single and double precision
        SPvec = write_vector<float>(SPnewMat->N);
     	DPvec = write_vector<double>(DPnewMat->N);
 
	// Run kernels and print results
	//printf("\nSingle Precision Results\n\n");
	/*
        SPout = (float *)malloc(SPnewMat->M*sizeof(float));
        spmv_csr_scalar<float>(SPnewMat, SPvec, SPout);
        verify<float>(SPmatrix->nz,SPmatrix->M,SPmatrix->rIndex,SPmatrix->cIndex,SPmatrix->val,SPvec,SPout);
        free(SPout);

        SPout = (float *)malloc(SPnewMat->M*sizeof(float));
        spmv_csr_vector(SPnewMat, SPvec, SPout);
	verify<float>(SPmatrix->nz,SPmatrix->M,SPmatrix->rIndex,SPmatrix->cIndex,SPmatrix->val,SPvec,SPout);
        free(SPout);        

	SPout = (float *)malloc(SPnewMat->M*sizeof(float));
        spmv_csr_adaptive(SPnewMat, SPvec, SPout);
        verify<float>(SPmatrix->nz,SPmatrix->M,SPmatrix->rIndex,SPmatrix->cIndex,SPmatrix->val,SPvec,SPout);
        free(SPout);
        
        SPout = (float *)malloc(SPnewMat->M*sizeof(float));
        spmv_pcsr(SPnewMat,SPvec,SPout);
        verify<float>(SPmatrix->nz,SPmatrix->M,SPmatrix->rIndex,SPmatrix->cIndex,SPmatrix->val,SPvec,SPout);
        free(SPout);
*/

	/* SINGLE PERCISION
	SPout = (float *)malloc(SPnewMat->M*sizeof(float));
        spmv_light(SPnewMat,SPvec,SPout);
        verify<float>(SPmatrix->nz,SPmatrix->M,SPmatrix->rIndex,SPmatrix->cIndex,SPmatrix->val,SPvec,SPout);
        free(SPout);
	*/
/*
        SPout = (float *)malloc(SPnewMat->M*sizeof(float));
        
        start = clock();
        spmv_cpu<float>(SPnewMat->val, SPvec, SPnewMat->cIndex,SPnewMat->rIndex,SPnewMat->M,SPout);
        end = clock();
        
        time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
        printf("Time taken for %s is %lf\n", "SpMV by CPU CSR Algorithm", time_taken);
        verify(SPmatrix->nz,SPmatrix->M,SPmatrix->rIndex,SPmatrix->cIndex,SPmatrix->val,SPvec,SPout);      	
	free(SPout);
	*/

	//printf("\nDouble Precision Results\n\n");       
	/*
	DPout = (double *)malloc(SPnewMat->M*sizeof(double));
        spmv_csr_scalar<double>(DPnewMat, DPvec, DPout);
        verify<double>(DPmatrix->nz,DPmatrix->M,DPmatrix->rIndex,DPmatrix->cIndex,DPmatrix->val,DPvec,DPout);
        free(DPout);

	DPout = (double *)malloc(SPnewMat->M*sizeof(double));
        spmv_csr_vector<double>(DPnewMat, DPvec, DPout);
        verify<double>(DPmatrix->nz,DPmatrix->M,DPmatrix->rIndex,DPmatrix->cIndex,DPmatrix->val,DPvec,DPout);
        free(DPout);

	DPout = (double *)malloc(SPnewMat->M*sizeof(double));
        spmv_csr_adaptive<double>(DPnewMat, DPvec, DPout);
        verify<double>(DPmatrix->nz,DPmatrix->M,DPmatrix->rIndex,DPmatrix->cIndex,DPmatrix->val,DPvec,DPout);
        free(DPout);
	
	DPout = (double *)malloc(SPnewMat->M*sizeof(double));
        spmv_pcsr<double>(DPnewMat, DPvec, DPout);
        verify<double>(DPmatrix->nz,DPmatrix->M,DPmatrix->rIndex,DPmatrix->cIndex,DPmatrix->val,DPvec,DPout);
        free(DPout);
	*/

	DPout = (double *)malloc(SPnewMat->M*sizeof(double));
        spmv_light<double>(DPnewMat, DPvec, DPout);
        verify<double>(DPmatrix->nz,DPmatrix->M,DPmatrix->rIndex,DPmatrix->cIndex,DPmatrix->val,DPvec,DPout);
        free(DPout);

	cudaEventRecord(stop);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&milliseconds, start, stop);

	// Calculate and print out GFLOPs and GB/s
	double gbs = ((DPmatrix->N * sizeof(double)) + (DPmatrix->nz*sizeof(double)) + (DPmatrix->M*sizeof(int)) + (DPmatrix->nz*sizeof(int)) + (DPmatrix->M*sizeof(double))) / (milliseconds/ITER) / 1e6;
    	time_taken = (milliseconds/ITER)/1000.0;
    	//printf("Average time taken for %s is %f\n", "SpMV by GPU CSR LightSpMV Algorithm",time_taken);
    	//printf("Average GFLOP/s is %lf\n",gflop/time_taken);
	//printf("Average GB/s is %lf\n\n",gbs);
	//Type,Size(KB),s,GB/s
	double size = ((DPmatrix->N * sizeof(double)) + (DPmatrix->nz*sizeof(double)) + (DPmatrix->M*sizeof(int)) + (DPmatrix->nz*sizeof(int)) + (DPmatrix->M*sizeof(double))) * 1.0E-6;
	printf("GPU,%f,%f,%f\n", size, time_taken, (gflop/time_taken));

	/*
	DPout = (double *)malloc(SPnewMat->M*sizeof(double));
	start = clock();
        spmv_cpu<double>(DPnewMat->val, DPvec, DPnewMat->cIndex,DPnewMat->rIndex,DPnewMat->M,DPout);
	end = clock();
	
	time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
        printf("Time taken for %s is %lf\n", "SpMV by CPU CSR Algorithm", time_taken);
        verify<double>(DPmatrix->nz,DPmatrix->M,DPmatrix->rIndex,DPmatrix->cIndex,DPmatrix->val,DPvec,DPout);
        free(DPout); 
	*/

	// Free up memory
        freeMatrixInfo<float>(SPmatrix);
	freeMatrixInfo<float>(SPnewMat);
	free(SPvec);

	freeMatrixInfo<double>(DPmatrix);
        freeMatrixInfo<double>(DPnewMat);
        free(DPvec);
}
