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

cl_event reduce(cl_kernel kernel, cl_command_queue que, int nels,
                    int lws_arg, int ngroups, 
                    cl_mem d_input, cl_mem d_output, cl_event init_evt){

  size_t lws[] = { lws_arg };
  size_t gws[] = { ngroups * lws[0] };

  cl_int err;
  cl_event evt;

  err = clSetKernelArg(kernel, 0, sizeof(d_input), &d_input);
  ocl_check(err, "set reduce kernel arg");

  err = clSetKernelArg(kernel, 1, sizeof(int), &nels);
  ocl_check(err, "set reduce kernel arg");

  err = clSetKernelArg(kernel, 2, sizeof(d_output), &d_output);
  ocl_check(err, "set reduce kernel arg");

  err = clSetKernelArg(kernel, 3, sizeof(cl_int) * lws[0], NULL);
  ocl_check(err, "set reduce kernel arg");

  err = clEnqueueNDRangeKernel(que, kernel, 1, NULL, gws, lws, 1, 
                               &init_evt, &evt);
  ocl_check(err, "start reduce kernel");

  return evt;
}


int main(int argn, char* args[]){

  if(argn <= 2){
    fprintf(stderr, "insert number of elements, lws and number of groups");
    exit(1);
  }

  int nels = atoi(args[1]);
  int lws = atoi(args[2]);
  int ngroups = atoi(args[3]);
  
  if(ngroups & 3 && (ngroups != 1)) {
    fprintf(stderr, "number of groups must be multiple of 4\n");
    exit(1);
  }
  
  if(nels & 3){
    fprintf(stderr, "number of elements must be a multiple of 4\n");
    exit(1);
  } 
  
  // cuz of local memory splitting in half in kernel
  if(lws & (lws - 1)){
    fprintf(stderr, "local work size must be a power of 2\n");
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

  cl_kernel reduce_kernel_sw = clCreateKernel(prog, "reduce_sw", &err);
  ocl_check(err, "create reduce kernel");

  // allocate input/output device buffers
  size_t memsize = sizeof(cl_int) * nels;
  cl_mem d_input = clCreateBuffer(ctx, CL_MEM_READ_WRITE, memsize, 
                                  NULL, &err);
  ocl_check(err, "allocate input buffer");

  cl_mem d_output = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
                                   sizeof(cl_int)*ngroups, 
                                  NULL, &err);
  ocl_check(err, "allocate output buffer");

  // launch kernels
  cl_event init_evt = init_array(init_kernel, que, nels, lws, ngroups, 
                                 d_input);

  cl_event reduce_evt = reduce(reduce_kernel_sw, que, nels/4, lws, ngroups,
                            d_input, d_output, init_evt);
  cl_event reduce_evt_2;
  if(ngroups > 1)
    reduce_evt_2 = reduce(reduce_kernel_sw, que, nels, lws, 1, d_output, 
                          d_output, reduce_evt);
  
  cl_int h_result;
  cl_event read_evt;
  err = clEnqueueReadBuffer(que, d_output, CL_TRUE, 0, 
                            sizeof(cl_int), &h_result, 1, 
                            (ngroups==1 ? &reduce_evt : &reduce_evt_2), 
                            &read_evt);
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
  printf("reduce time: %lf\n", reduce_time);
  printf("read time: %lf\n", read_time);

  // release boiler
  clReleaseMemObject(d_input);
  clReleaseMemObject(d_output);
  clReleaseKernel(init_kernel);
  clReleaseKernel(reduce_kernel_sw);
  clReleaseProgram(prog);
  clReleaseCommandQueue(que);
  clReleaseContext(ctx);

  return 0;


}
