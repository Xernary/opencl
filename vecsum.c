#define CL_TARGET_OPENCL_VERSION 120
#include <stdlib.h>
#include <stdio.h>
#include "ocl_boiler.h"

void verify_sum(int* array, int n){
  for(int i = 0; i < n; i++)
    if(array[i] != n){
      perror("result array values are not correct\n");
      printf("array[%d]: %d\n", i, array[i]);
      exit(-1);
    }
}

cl_event init_arrays(cl_kernel kernel, cl_command_queue que, int n, int lws, 
                     cl_mem d_array1, cl_mem d_array2){
  if(lws <= 0) lws = 16;
  
  size_t gws[] = { n };
  size_t lws_[] = { (size_t) lws };
  cl_event evt;
  cl_int err;
  err = clSetKernelArg(kernel, 0, sizeof(d_array1), &d_array1);
  ocl_check(err, "setting kernel arg");
  err = clSetKernelArg(kernel, 1, sizeof(d_array2), &d_array2);
  ocl_check(err, "setting kernel arg");
  err = clSetKernelArg(kernel, 2, sizeof(int), &n);
  ocl_check(err, "setting kernel arg");
  err = clEnqueueNDRangeKernel(que, kernel, 1, NULL, gws, lws_, 0, NULL, &evt);
  printf("lws: %d\n", lws);
  ocl_check(err, "starting init kernel");
  return evt;
}

cl_event sum_arrays(cl_kernel kernel, cl_command_queue que, int n, int lws, 
                    cl_mem d_array1, cl_mem d_array2, cl_mem d_result,
                    cl_event evt_wait){
  if(lws <= 0) lws = 16;
  
  size_t gws[] = { n };
  size_t lws_[] = { (size_t) lws };

  cl_event evt;
  cl_int err;
  err = clSetKernelArg(kernel, 0, sizeof(d_array1), &d_array1);
  ocl_check(err, "setting kernel arg");
  err = clSetKernelArg(kernel, 1, sizeof(d_array2), &d_array2);
  ocl_check(err, "setting kernel arg");
  err = clSetKernelArg(kernel, 2, sizeof(d_result), &d_result);
  ocl_check(err, "setting kernel arg");
  err = clSetKernelArg(kernel, 3, sizeof(int), &n);
  ocl_check(err, "setting kernel arg");
  err = clEnqueueNDRangeKernel(que, kernel, 1, NULL, gws, lws_, 1, &evt_wait, &evt);
  ocl_check(err, "starting sum kernel");
  return evt;
}

int main(int argn, char* args[]){

  if(argn < 3) {
    printf("Must specify array size and local work size\n");
    exit(1);    
  }
  
  int n = atoi(args[1]);
  int lws = atoi(args[2]);
  printf("n: %d\n", n);
  // boiler
  cl_platform_id p = select_platform();
  cl_device_id d = select_device(p);
  cl_context ctx = create_context(p, d);
  cl_command_queue que = create_queue(ctx, d);
  cl_program prog = create_program("vecsum.ocl", ctx, d);

  cl_int err;

  // create kernel object
  cl_kernel init_kernel = clCreateKernel(prog, "init_kernel", &err);
  ocl_check(err, "creating kernel");

  // create kernel object
  cl_kernel sum_kernel = clCreateKernel(prog, "sum_kernel", &err);
  ocl_check(err, "creating kernel");

  // create memory buffer on device
  cl_mem d_array1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(int)*n, NULL, &err);
  ocl_check(err, "creating input 1 buffer");

  cl_mem d_array2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(int)*n, NULL, &err);
  ocl_check(err, "creating input 2 buffer");

  cl_mem d_result = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(int)*n, NULL, &err);
  ocl_check(err, "creating output buffer");


  // execute init kernels (via wrapped kernel function)
  cl_event init_evt = init_arrays(init_kernel, que, n, lws, 
                                  d_array1, d_array2); 

  // wait for init kernel event completed and execute sum kernel (via wrapped kernel function)
  cl_event sum_evt = sum_arrays(sum_kernel, que, n, lws, 
                                d_array1, d_array2, d_result, init_evt);
  
  // map device output buffer to memory
  cl_event map_evt;
  int* h_array = clEnqueueMapBuffer(que, d_result, CL_TRUE, CL_MAP_READ, 0, 
                                    sizeof(cl_int)*n, 1,
                                    &sum_evt, &map_evt, &err);
  ocl_check(err, "map buffer");

  // read the result from host array pointer (mapped memory) 
  verify_sum(h_array, n); 

  // unmap memory buffer
  clEnqueueUnmapMemObject(que, d_result, h_array, 0, NULL, NULL);
  ocl_check(err, "unmap buffer");

  double init_runtime = runtime_ms(init_evt);
  double sum_runtime = runtime_ms(sum_evt);
  double map_runtime = runtime_ms(map_evt);

  printf("init time: %gms\n", init_runtime);
  printf("sum time: %gms\n", sum_runtime);
  printf("map time: %gms\n", map_runtime);

  // release boiler
  clReleaseKernel(init_kernel);
  clReleaseKernel(sum_kernel);
  clReleaseMemObject(d_result);
  clReleaseMemObject(d_array1);
  clReleaseMemObject(d_array2);
  clReleaseProgram(prog);
  clReleaseCommandQueue(que);
  clReleaseContext(ctx);

  return 0;
}
