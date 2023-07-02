#include <stdlib.h>
#include <stdio.h>
#define OCL_TARGET_VERSION 120
#include "ocl_boiler.h"


void verify(int nels, int result){
  int sum = 0;
  for(int i = 0; i < nels, i++)
      sum += i;
  if(sum != result) printf("mismatch: %d != %d", result, sum);
}

int main(int argn, char* args[]){

  if(argn <= 2){
    fprintf(stderr, "insert number of elements and lws");
    exit(1);
  }

  int nels = atoi(args[1]);
  int lws = atoi(args[2]);

  if(nels & (nels - 1)){
    fprintf(stderr, "number of elements must be power of 2");
    exit(1);
  }

  // boiler 
  cl_platform_id p = select_platform();
  cl_device_id d = select_device(d);
  cl_context ctx = create_context(p, d);
  cl_command_queue que = create_command_queue(ctx, d);
  cl_program prog = create_program("reduce.ocl", ctx, d);

  // kernels
  cl_int err;
  cl_kernel init_kernel = clCreateKernel("init_array", prog, &err);
  ocl_check(err, "create init kernel");

  cl_kernel reduce_kernel_v1 = clCreateKernel("reduce_v1", prog, &err);
  ocl_check(err, "create reduce kernel");

  // allocate input/output device buffers
  size_t memsize = sizeof(cl_int) * nels;
  cl_mem d_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY, memsize, 
                                  NULL, &err);
  ocl_check(err, "allocate input buffer");

  cl_mem d_output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(cl_int), 
                                  NULL, &err);
  ocl_check(err, "allocate output buffer");

  // get preffered lws for each kernels
  cl_uint preferred_init_lws;
  err = clGetKernelWorkGroupInfo(init_kernel, d, 
                          CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                 sizeof(preferred_init_lws) 
                                 &preferred_init_lws, NULL);
  ocl_check(err, "get preferred init lws");

  cl_uint preferred_reduce_v1_lws;
  err = clGetKernelWorkGroupInfo(reduce_kernel_v1, d, 
                          CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                 sizeof(preferred_reduce_v1_lws) 
                                 &preferred_reduce_v1_lws, NULL);
  ocl_check(err, "get preferred reduce lws");

  // launch kernels
  cl_event init_evt = init_array(init_kernel, que, nels, preferred_init_lws);

  cl_event reduce_evt = reduce(reduce_kernel_v1, que, nels, preferred_reduce_v1_lws,
                           init_evt);


  int* h_result;
  cl_event read_evt;
  err = clEnqueueReadBuffer(que, d_output, CL_TRUE, 0, 
                                      sizeof(d_output), h_result, 1, 
                                      &reduce_evt, &read_evt);

  verify(nels, h_result);



}
