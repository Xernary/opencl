#define CL_TARGET_OPENCL_VERSION 120
#include <stdlib.h>
#include <stdio.h>
#include "ocl_boiler.h"

void error(const char*);
void verify(const int*, int, int);
cl_event init_array(cl_command_queue, cl_kernel, cl_mem, int, int);

int main(int argn, char* args[]){
  
 if(argn <= 2){
    perror("Must insert matrix rows and cols");
    exit(1);
  }
  
  int rows = atoi(args[1]);
  int cols = atoi(args[2]);
  int n = rows*cols;
  
  // boiler 
  cl_platform_id p = select_platform();
  cl_device_id d = select_device(p);
  cl_context ctx = create_context(p, d);
  cl_command_queue que = create_queue(ctx, d);
  cl_program prog = create_program("vecinit.ocl", ctx, d); // compiles kernel and puts it into a prog object 

  cl_int err;
  // create kernel obect from the program (compiled kernel) given the name of the kernel funciton (there could be multiples kernels function into compiled kernel file)
  cl_kernel init_kernel = clCreateKernel(prog, "mat_init_kernel", &err);
  ocl_check(err, "create init_kernel");
  
  // allocate memory on device and get a kinda-pointer (d_array) to it
  cl_mem d_array = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(int)*n, NULL, &err);
  ocl_check(err, "create d_buffer");
  
  // start the kernel and get an event refering to kernel execution completion
  cl_event init_evt = init_array(que, init_kernel, d_array, rows, cols);

  int* h_array = calloc(n, sizeof(int));
  if(h_array == NULL) error("failed to allocate host array");

  cl_event read_evt;

  err = clEnqueueReadBuffer(que, d_array, CL_TRUE, 0, sizeof(int)*n, h_array, 1, &init_evt, &read_evt);
  ocl_check(err, "read device buffer");

  verify(h_array, n, rows);

  double init_runtime = runtime_ms(init_evt);
  double read_runtime = runtime_ms(read_evt);
  
  printf("init: %gms\n", init_runtime);
  printf("read: %gms\n", read_runtime);

  free(h_array);

  clReleaseKernel(init_kernel);
  clReleaseMemObject(d_array);
  clReleaseProgram(prog);
  clReleaseCommandQueue(que);
  clReleaseContext(ctx);

  return 0;
}

void verify(const int* array, int n, int rows){
  for(int i = 0; i < n; i++){
    int r = i%rows;
    int c = i/rows;
    int expected = r-c;
    if(array[c*rows + r] != expected){
      fprintf(stderr, "mismatch @ [%d][%d]: %d != %d\n", r, c, 
              array[c*rows+r], expected);
    }
  }
}

void error(const char* err){
  fprintf(stderr, "%s\n", err);
  exit(1);
}

cl_event init_array(cl_command_queue que, cl_kernel init_kernel, cl_mem d_array, 
                    int rows, int cols){
  size_t gws[] = { rows, cols };
  cl_int err;
  cl_event ret;

  err = clSetKernelArg(init_kernel, 0, sizeof(d_array), &d_array);
  ocl_check(err, "set kernel arg");

  err = clSetKernelArg(init_kernel, 1, sizeof(int), &rows);
  ocl_check(err, "set kernel arg");

  err = clSetKernelArg(init_kernel, 2, sizeof(int), &cols);
  ocl_check(err, "set kernel arg");

  err = clEnqueueNDRangeKernel(que, init_kernel, 2, NULL, gws, NULL, 0, NULL, &ret);
  ocl_check(err, "enqueue kernel execution");

  return ret;
}



