#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"

void verify(int nels, int* result){
  int sum = 0;
  for(int i = 0; i < nels; i++)
      sum += i;
  if(sum != *result) printf("mismatch: %d != %d", *result, sum);
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
  /*
  if(nels & 4)){
    fprintf(stderr, "number of elements must be multiple of 4");
    exit(1);
  }*/

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

  cl_kernel reduce_kernel_v1 = clCreateKernel(prog, "reduce_v1", &err);
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
  cl_uint preferred_init_lws;
  err = clGetKernelWorkGroupInfo(init_kernel, d, 
                          CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                 sizeof(preferred_init_lws), 
                                 &preferred_init_lws, NULL);
  ocl_check(err, "get preferred init lws");

  cl_uint preferred_reduce_v1_lws;
  err = clGetKernelWorkGroupInfo(reduce_kernel_v1, d, 
                          CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                 sizeof(preferred_reduce_v1_lws), 
                                 &preferred_reduce_v1_lws, NULL);
  ocl_check(err, "get preferred reduce lws");

  // launch kernels
  cl_event init_evt = init_array(init_kernel, que, nels, lws, 
                                 preferred_init_lws, d_input);
  int steps = log(nels)/log(4);
  cl_event reduce_evts[steps + 1];
  int current_nels = nels;
  reduce_evts[0] = init_evt;
  for(int i = 1; i <= steps; i++){

    reduce_evts[i] = reduce(reduce_kernel_v1, que, current_nels/4, lws,
                            preferred_reduce_v1_lws, d_input,
                            d_output, reduce_evts[i-1]);
    cl_mem temp = d_input;
    d_input = d_output;
    d_output = temp;
    current_nels /= 4;
  }

  int* h_result;
  cl_event read_evt;
  err = clEnqueueReadBuffer(que, d_output, CL_TRUE, 0, 
                                      sizeof(d_output), h_result, 1, 
                                      &reduce_evts[steps], &read_evt);

  verify(nels, h_result);

  // benchmarks
  double init_time = runtime_ms(init_evt);
  double reduce_time = 0;
  double read_time = runtime_ms(read_evt);

  for(int i = 1; i <= steps; i++){
    reduce_time += runtime_ms(reduce_evts[i]); 
  }

  printf("init time: %lf\n", init_time);
  printf("reduce time: %lf\n", reduce_time);
  printf("read time: %lf\n", read_time);

  // release boiler
  clReleaseMemObject(d_input);
  clReleaseMemObject(d_output);
  clReleaseKernel(init_kernel);
  clReleaseKernel(reduce_kernel_v1);
  clReleaseProgram(prog);
  clReleaseCommandQueue(que);
  clReleaseContext(ctx);

  return 0;


}
