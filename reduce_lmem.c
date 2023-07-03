#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"

void verify(int nels, const cl_int result){
  int sum = 0;
  for(int i = 0; i < nels; i++)
      sum += i;
  if(sum != result) printf("mismatch: %d != %d\n", result, sum);
}

cl_event init_array(cl_kernel kernel, cl_command_queue que, int nels,
                    int lws_arg, int preferred_lws, 
                    cl_mem d_input){

  size_t lws[] = { lws_arg>0 ? lws_arg : preferred_lws };
  size_t gws[] = { round_mul_up(nels, lws[0]) };

  cl_int err;
  cl_event evt;

  err = clSetKernelArg(kernel, 0, sizeof(d_input), &d_input);
  ocl_check(err, "set init kernel arg");

  err = clSetKernelArg(kernel, 1, sizeof(int), &nels);
  ocl_check(err, "set init kernel arg");

  err = clEnqueueNDRangeKernel(que, kernel, 1, NULL, gws, lws, 0, 
                               NULL, &evt);
  ocl_check(err, "start init kernel");

  return evt;
}

cl_event reduce_lmem(cl_kernel kernel, cl_command_queue que, int nels,
                    int lws_arg, int preferred_lws, 
                    cl_mem d_input, cl_mem d_output, cl_event init_evt){

  size_t lws[] = { lws_arg>0 ? lws_arg : preferred_lws };
  size_t gws[] = { round_mul_up(nels, lws[0]) };

  cl_int err;
  cl_event evt;

  err = clSetKernelArg(kernel, 0, sizeof(d_input), &d_input);
  ocl_check(err, "set reduce kernel arg");

  err = clSetKernelArg(kernel, 1, sizeof(int), &nels);
  ocl_check(err, "set reduce kernel arg");

  err = clSetKernelArg(kernel, 2, sizeof(d_output), &d_output);
  ocl_check(err, "set reduce kernel arg");

  err = clSetKernelArg(kernel, 3, sizeof(cl_int) * lws[0], NULL);
  ocl_check(err, "set lmem kernel arg");

  err = clEnqueueNDRangeKernel(que, kernel, 1, NULL, gws, lws, 1, 
                               &init_evt, &evt);
  ocl_check(err, "start reduce kernel");

  return evt;
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
  
  float res = nels;
  for(int i = 0; i < log(nels)/log(4); i++){
    res /= 4;
  }
  
  if(res != 1){
    fprintf(stderr, "number of elements must be power of 4");
    exit(1);
  }

  // boiler 
  cl_platform_id p = select_platform();
  cl_device_id d = select_device(p);
  cl_context ctx = create_context(p, d);
  cl_command_queue que = create_queue(ctx, d);
  cl_program prog = create_program("reduce.ocl", ctx, d);

  // kernels
  cl_int err;
  cl_kernel init_kernel = clCreateKernel(prog, "init_array", &err);
  ocl_check(err, "create init kernel");

  cl_kernel reduce_kernel_lmem = clCreateKernel(prog, "reduce_lmem", &err);
  ocl_check(err, "create reduce kernel");

  // allocate input/output device buffers
  size_t memsize = sizeof(cl_int) * nels;
  cl_mem d_input = clCreateBuffer(ctx, CL_MEM_READ_WRITE, memsize, 
                                  NULL, &err);
  ocl_check(err, "allocate input buffer");

  cl_mem d_output = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
                                   sizeof(cl_int), 
                                  NULL, &err);
  ocl_check(err, "allocate output buffer");

  // get preffered lws for each kernels
  size_t preferred_init_lws;
  err = clGetKernelWorkGroupInfo(init_kernel, d, 
                          CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                 sizeof(preferred_init_lws), 
                                 &preferred_init_lws, NULL);
  ocl_check(err, "get preferred init lws");

  size_t preferred_reduce_lmem_lws;
  err = clGetKernelWorkGroupInfo(reduce_kernel_lmem, d, 
                          CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                 sizeof(preferred_reduce_lmem_lws), 
                                 &preferred_reduce_lmem_lws, NULL);
  ocl_check(err, "get preferred reduce lws");

  // launch kernels
  cl_event init_evt = init_array(init_kernel, que, nels, lws, 
                                 preferred_init_lws, d_input);

  cl_event reduce_evt = reduce_lmem(reduce_kernel_lmem, que, nels/4, 
                                    lws, preferred_reduce_lmem_lws, 
                                    d_input, d_output, init_evt);

  cl_int h_result;
  cl_event read_evt;
  err = clEnqueueReadBuffer(que, d_output, CL_TRUE, 0, 
                                      sizeof(cl_int), &h_result, 1, 
                                      &reduce_evt, &read_evt);
  ocl_check(err, "read buff");
  printf("RESULT: %d\n", h_result);
  verify(nels, h_result);

  err = clFlush(que);
  ocl_check(err, "flush queue");

  // benchmarks
  double init_time = runtime_ms(init_evt);
  double reduce_time = 0;
  double read_time = runtime_ms(read_evt);

  printf("init time: %lf\n", init_time);
//  printf("reduce time: %lf\n", reduce_time);
  printf("read time: %lf\n", read_time);
  
  // illegal


  // release boiler
  clReleaseMemObject(d_input);
  clReleaseMemObject(d_output);
  clReleaseKernel(init_kernel);
  clReleaseKernel(reduce_kernel_lmem);
  clReleaseProgram(prog);
  clReleaseCommandQueue(que);
  clReleaseContext(ctx);

  return 0;


}
