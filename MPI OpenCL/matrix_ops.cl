__kernel void square_magnitude(const int size,
                      __global int* v) {
    
    // Thread identifiers
    const int globalIndex = get_global_id(0); // Row ID of C (0..M)   
 
    //uncomment to see the index each PE works on
    //printf("Kernel process index :(%d)\n ", globalIndex);

    v[globalIndex] = v[globalIndex] * v[globalIndex];
}


__kernel void add_vector(const int size,
                      const __global int* v1,const __global int* v2,__global int* v3) {
    
    // Thread identifiers
    const int globalIndex = get_global_id(0);    
 
    //uncomment to see the index each PE works on
    //printf("Kernel process index :(%d)\n ", globalIndex);

    v3[globalIndex] = v2[globalIndex] + v1[globalIndex];
}


__kernel void add_matrix(const int size,
                      const __global int* v1,const __global int* v2,__global int* v3) {
    
    // Thread identifiers
    const int i = get_global_id(0);    
    const int j = get_global_id(1);    

    const int index = (i * size) + j;
 
    //uncomment to see the index each PE works on
    //printf("Kernel process index :(%d,%d) and 1d index is %d\n", i, j, index);

    v3[index] = v2[index] + v1[index];
}

__kernel void multi_matrix(const int size,
                      const __global int* A,const __global int* B,__global int* C,const int chunksize) {
    
    // Thread identifiers
    const int i = get_global_id(0);    
    const int j = get_global_id(1);    
	
	int k;
	float c= 0;
	
	if(i > size) return;
	if(j > size) return;
	
	
	for (k =0; k <  size; k++)
	{
	
		c =  c + A[i* chunksize + k] *  B[k* size + j]; 
	
	}   

    C[i* chunksize + j] = c;
}
