#include <stdlib.h>
#include "ocl_boiler.h"

void verify(int* array, int n){
  for(int i = 0; i < n; i++)
    if(array[i] != n){
      perror("result array values are not correct\n");
      exit(-1);
    }
}

cl_evt init_arrays(cl_kernel kernel, cl_command_queue que, int n, int lws){
  if(lws <= 0) lws = 256;

  cl_evt evt;
  cl_int err;
  err = clEnqueueNDRangeKernel(que, kernel, 1, 0, n, lws, 0, NULL, evt);
  ocl_check(err, "starting init kernel");
  return evt;
}

cl_evt sum_arrays(cl_kernel kernel, cl_command_queue que, int n, int lws){
  if(lws <= 0) lws = 256;

  cl_evt evt;
  cl_int err;
  err = clEnqueueNDRangeKernel(que, kernel, 1, 0, n, lws, 0, NULL, evt);
  ocl_check(err, "starting sum kernel");
  return evt;
}

int main(int argn, char* args){

  if(argn < 3) {
    printf("Must specify array size and local work size\n");
    exit(1);    
  }
  
  int n = atoi(args[1]);
  int lws = atoi(args[2]);

  // boiler
  cl_platform p = select_platform();
  cl_device_id d = select_device(p);
  cl_context ctx = create_context(p, d);
  cl_command_queue q = create_queue(ctx, d);
  cl_program p = create_program("vecinit.ocl", ctx, d);

  cl_err err;

  // create kernel object
  cl_kernel init_kernel = clCreateKernel(prog, "vecinit.ocl", err);
  ocl_check(err, "creating kernel");

  // create kernel object
  cl_kernel sum_kernel = clCreateKernel(prog, "vecsum.ocl", err);
  ocl_check(err, "creating kernel");

  // create memory buffer on device
  cl_mem d_array1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(int)*n, NULL, &err);
  ocl_check(err, "creating input 1 buffer");

  cl_mem d_array2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(int)*n, NULL, &err);
  ocl_check(err, "creating input 2 buffer");

  cl_mem d_result = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(int)*n, NULL, &err);
  ocl_check(err, "creating output buffer");

  // map device output buffer to memory
  int* h_array = clEnqueueMapBuffer(que, d_result, TRUE, CL_MAP_READ, 0, n, 0, NULL, NULL, err);

  // execute init kernels (via wrapped kernel function)
  cl_evt init_evt = init_arrays(init_kernel, que, n, lws); 

  // wait for init kernel event completed and execute sum kernel (via wrapped kernel function)
  cl_evt sum_evt = sum_arrays(sum_kernel, que, n, lws);

  // read the result from host array pointer (mapped memory) 
  verify_sum(h_array, n); 

  // unmap memory buffer
  clEnqueueUnmapBuffer(que, d_result, h_array, 0, NULL, NULL);

  // release boiler
  clReleaseKernel(init_kernel);
  clReleaseKernel(sum_kernel);
  clReleaseMemObject(d_result);
  clReleaseMemObject(d_array1);
  clReleaseMemObject(d_array2);
  clReleaseProgram(p);
  clReleaseCommandQueue(que);
  clReleaseContext(ctx);

  return 0;
}
