#include <iostream>
#include<stdio.h>
#include<time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <thread>
#include <mpi.h>
#include <omp.h>
#include <CL/cl.h>

using namespace std;
using namespace std::chrono;

#define BILLION  1000000000L;
int NUM_THREADS = 1;

int SZ = 8;
int chunk_size;
int **A, **B, **C;
int *A_dim , *B_dim, *C_dim;
cl_mem bufA, bufB, bufC;


cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue queue;
cl_event event = NULL;

int err;
cl_device_id create_device();
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname);
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename);
void setup_kernel_memory(int chunk_rows);
void copy_kernel_args();
void free_memory();

void init(int** &matrix, int rows, int cols, bool initialise);
void print( int** matrix, int rows, int cols);
void* add(void* block_id);
void* multiply(void* args);
void head(int num_processes,int process_rank);
void node(int process_rank, int num_processes);
void convertToOneD(int** matrix, int rows, int cols, int* oneDarray);
void convertToTwoD(int** matrix, int rows, int cols, int* oneDarray, int process_rank);
void emptymatrixmapping(int** &A, int rows, int cols);

int main(int argc, char** argv) {
    if(argc > 1) SZ = atoi(argv[1]);


    MPI_Init(NULL, NULL);

    int num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    if(process_rank == 0) //head
        head(num_processes,process_rank);
    else
        node(process_rank, num_processes);
    
    MPI_Finalize();
	
	
}


void head(int num_processes,int process_rank)
{
	auto start = high_resolution_clock::now();
    init(A, SZ, SZ, true), init(B, SZ, SZ, true);
    
    print(A, SZ, SZ);
    print(B, SZ, SZ);

    //my plan is to scatter A based on number of processes and broadcast B to all nodes
    int num_rows_per_process_from_A = SZ / num_processes;
    int num_elements_to_bcast = (SZ * SZ);
    int num_elements_to_scatter_or_gather = (SZ * SZ) / num_processes;

	chunk_size = num_rows_per_process_from_A;
	 size_t global[] = {(size_t)SZ, (size_t)SZ};
	 emptymatrixmapping(C,SZ,SZ);

    MPI_Scatter(&A[0][0], num_elements_to_scatter_or_gather ,  MPI_INT , &A , 0, MPI_INT, 0 , MPI_COMM_WORLD);
    MPI_Bcast(&B[0][0], num_elements_to_bcast , MPI_INT , 0 , MPI_COMM_WORLD);
	A_dim = (int *)malloc(sizeof(int) * num_rows_per_process_from_A * SZ);
	B_dim =	(int *)malloc(sizeof(int) * SZ * SZ);
	C_dim = (int *)malloc(sizeof(int) * num_rows_per_process_from_A * SZ);
    convertToOneD(A,num_rows_per_process_from_A,SZ,A_dim);
	convertToOneD(B,SZ,SZ,B_dim);
    
   setup_openCL_device_context_queue_kernel((char *)"./matrix_ops.cl", (char *)"multi_matrix");

   //this function is used to load/copy memory and link arguments -- you will need to change this
   //for your program as this varies from one implementation to another
   setup_kernel_memory(num_rows_per_process_from_A);
   copy_kernel_args();

   //submit the kernel for execution
   clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, &event);
   clWaitForEvents(1, &event);

   //copying data from the device back to host c matrix
   //The resultant matrix should be the same size as the chunk of matrix A that was used in the multiplication. if A was 2x8 matrix and was multplied with B which was 8x8 matrix you will get a 2x8 matrix as the resultant chunk of matrix C.
   clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,  num_rows_per_process_from_A * SZ * sizeof(int), &C_dim[0], 0, NULL, NULL);



	convertToTwoD(C,num_rows_per_process_from_A,SZ,C_dim,process_rank);
   //frees memory for device, kernel, queue, etc.
   //you will need to modify this to free your own buffers
   free_memory();
 
 
  
 //send the results back to the head node for merging and printing
    MPI_Gather(MPI_IN_PLACE, num_elements_to_scatter_or_gather , MPI_INT, &C[0][0] , num_elements_to_scatter_or_gather , MPI_INT, 0 , MPI_COMM_WORLD);
   

 auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << "Time taken by function: "
        << duration.count() << " milliseconds"        
        <<endl;



    print(C, SZ, SZ);


}
void node(int process_rank, int num_processes)
{
    int num_rows_per_process_from_A = SZ / num_processes;
    int num_elements_to_bcast = (SZ * SZ);
    int num_elements_to_scatter_or_gather = (SZ * SZ) / num_processes;

	chunk_size = num_rows_per_process_from_A;
    //receive my rows of matrix A, and all B
    init(A, num_rows_per_process_from_A , SZ, true), init(B, SZ, SZ, false), init(C, num_rows_per_process_from_A, SZ, false);
	emptymatrixmapping(C,SZ,SZ);
	
	 size_t global[] = {(size_t)SZ, (size_t)SZ};

    MPI_Scatter(NULL, num_elements_to_scatter_or_gather , MPI_INT , &A[0][0], num_elements_to_scatter_or_gather, MPI_INT, 0 , MPI_COMM_WORLD);
    MPI_Bcast(&B[0][0], num_elements_to_bcast , MPI_INT , 0 , MPI_COMM_WORLD);
	A_dim = (int *)malloc(sizeof(int) * num_rows_per_process_from_A * SZ);
	B_dim =	(int *)malloc(sizeof(int) * SZ * SZ);
	C_dim = (int *)malloc(sizeof(int) * num_rows_per_process_from_A * SZ);
	convertToOneD(A,num_rows_per_process_from_A,SZ,A_dim);
	convertToOneD(B,SZ,SZ,B_dim);
    
	setup_openCL_device_context_queue_kernel((char *)"./matrix_ops.cl", (char *)"multi_matrix");

   //this function is used to load/copy memory and link arguments -- you will need to change this
   //for your program as this varies from one implementation to another
   setup_kernel_memory(num_rows_per_process_from_A);
   copy_kernel_args();

   //submit the kernel for execution
   clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, &event);
   clWaitForEvents(1, &event);

   //copying data from the device back to host c matrix
    //The resultant matrix should be the same size as the chunk of matrix A that was used in the multiplication. if A was 2x8 matrix and was multplied with B which was 8x8 matrix you will get a 2x8 matrix as the resultant chunk of matrix C.
   clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,  num_rows_per_process_from_A * SZ * sizeof(int), &C_dim[0], 0, NULL, NULL);
   

	convertToTwoD(C,num_rows_per_process_from_A,SZ,C_dim,process_rank);

   //frees memory for device, kernel, queue, etc.
   //you will need to modify this to free your own buffers
   free_memory();
	

    MPI_Gather(&C[0][0], num_elements_to_scatter_or_gather , MPI_INT, NULL, num_elements_to_scatter_or_gather , MPI_INT, 0 , MPI_COMM_WORLD);

}



void emptymatrixmapping(int** &A, int rows, int cols)
{
	A = (int **) malloc(sizeof(int*) * rows * cols);  // number of rows * size of int* address in the memory
    int* tmp = (int *) malloc(sizeof(int) * cols * rows); 

    for(int i = 0 ; i < rows ; i++) {
        A[i] = &tmp[i * cols];
    }
	
	for(long i = 0 ; i < rows; i++) {
        for(long j = 0 ; j < cols; j++) {
            A[i][j] = 0;
        }
    }

	
	
	
}


void init(int** &A, int rows, int cols, bool initialise) {
    A = (int **) malloc(sizeof(int*) * rows * cols);  // number of rows * size of int* address in the memory
    int* tmp = (int *) malloc(sizeof(int) * cols * rows); 

    for(int i = 0 ; i < SZ ; i++) {
        A[i] = &tmp[i * cols];
    }
  

    if(!initialise) return;

    for(long i = 0 ; i < rows; i++) {
        for(long j = 0 ; j < cols; j++) {
            A[i][j] = rand() % 100; // any number less than 100
        }
    }
}

void print( int** A, int rows, int cols) {
  for(long i = 0 ; i < rows; i++) { //rows
        for(long j = 0 ; j < cols; j++) {  //cols
            printf("%d ",  A[i][j]); // print the cell value
        }
        printf("\n"); //at the end of the row, print a new line
    }
    printf("----------------------------\n");
}

void printf(int *A, int size)
{
  
   for (long i = 0; i < size; i++)
   {      
      for (long j = 0; j < size; j++)
      {                     
         printf("%d ", A[i*size + j]); // print the cell value
      }                
      printf("\n"); // print the cell value
   }
   printf("\n----------------------------\n");
}

void convertToOneD(int** matrix, int rows, int cols, int* oneDarray)
{
	
	for(int  i =0;i < rows;i++)
	{
		for(int j=0; j < cols;j++)
		{
			
			oneDarray[i* cols +j] = matrix[i][j];
			
			
		}
		
	}
	
	
}


void convertToTwoD(int** matrix, int rows, int cols, int* oneDarray, int process_rank)
{
	for( int i = process_rank * rows;i < (process_rank + 1) * rows;i++)
	{
		 int x =0;
		 
		for(int j=0; j < cols;j++)
		{
		   int indexrow = (i * cols + j) / cols;
		   int indexcol = (i * cols + j) % cols;
		    //int value = *(oneDarray+ x* cols + j);
			// cout << "The value is" + value << endl;
		  //translate for the positioning for the one dimensional array which will have the same starting positioning regardless of the process rank. 	
		  //matrix[indexrow][indexcol]= oneDarray[x* cols + j];
		 matrix[indexrow][indexcol]= *(oneDarray+ x* cols + j);
			
		}
		
		x++;
		
	}
		
	
}



void free_memory()
{
   //free the buffers
   clReleaseMemObject(bufA);
   clReleaseMemObject(bufB);
   clReleaseMemObject(bufC);

   //free opencl objects
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);

   free(A_dim);
   free(B_dim);
   free(C_dim);
}

void copy_kernel_args()
{
   clSetKernelArg(kernel, 0, sizeof(int), (void *)&SZ);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufA);
   clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufB);
   clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufC);
   clSetKernelArg(kernel, 4, sizeof(int), (void *)&chunk_size);

   if (err < 0)
   {
      perror("Couldn't create a kernel argument");
      printf("error = %d", err);
      exit(1);
   }
}


void setup_kernel_memory(int chunk_rows)
{
   bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, chunk_rows * SZ * sizeof(int), NULL, NULL);
   bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, SZ * SZ * sizeof(int), NULL, NULL);
   bufC=  clCreateBuffer(context, CL_MEM_WRITE_ONLY, chunk_rows * SZ * sizeof(int), NULL, NULL);

   // Copy matrices to the GPU
   clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, chunk_rows * SZ * sizeof(int), &A_dim[0], 0, NULL, NULL);
   clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, SZ * SZ * sizeof(int), &B_dim[0], 0, NULL, NULL);      
   clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0,  chunk_rows * SZ * sizeof(int), &C_dim[0], 0, NULL, NULL);
}

void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname)
{
   device_id = create_device();
   cl_int err;
   context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
   if (err < 0)
   {
      perror("Couldn't create a context");
      exit(1);
   }

   program = build_program(context, device_id, filename);
   queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
   if (err < 0)
   {
      perror("Couldn't create a command queue");
      exit(1);
   };

   kernel = clCreateKernel(program, kernelname, &err);
   if (err < 0)
   {
      perror("Couldn't create a kernel");
      printf("error =%d", err);
      exit(1);
   };
}


cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename)
{

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
   if (program_handle == NULL)
   {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char *)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   /* Create program from file 

   Creates a program from the source code in the add_numbers.cl file. 
   Specifically, the code reads the file's content into a char array 
   called program_buffer, and then calls clCreateProgramWithSource.
   */
   program = clCreateProgramWithSource(ctx, 1,
                                       (const char **)&program_buffer, &program_size, &err);
   if (err < 0)
   {
      perror("Couldn't create the program");
      exit(1);
   }
   free(program_buffer);

   /* Build program 

   The fourth parameter accepts options that configure the compilation. 
   These are similar to the flags used by gcc. For example, you can 
   define a macro with the option -DMACRO=VALUE and turn off optimization 
   with -cl-opt-disable.
   */
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if (err < 0)
   {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                            0, NULL, &log_size);
      program_log = (char *)malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}


cl_device_id create_device()
{

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if (err < 0)
   {
      perror("Couldn't identify a platform");
      exit(1);
   }

   // Access a device
   // GPU
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if (err == CL_DEVICE_NOT_FOUND)
   {
      // CPU
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if (err < 0)
   {
      perror("Couldn't access any devices");
      exit(1);
   }

   return dev;
}










