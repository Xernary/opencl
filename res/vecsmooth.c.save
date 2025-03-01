#include <stdlib.h>
#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"

void verify_init(const int* array, int n){
  for(int i = 0; i < n; i++){
    if(array[i] != i) 
      fprintf(stderr, "mismatch init #%d : %d != %d\n", i, array[i], i);
  }
}

void verify(const int* array, int n){
  for(int i = 0; i < n; i++){
    int expected = (i - !!(i == n - 1));
    if(array[i] != expected) 
      fprintf(stderr, "mismatch #%d : %d != %d\n", i, array[i], expected);
  }
}

cl_event init_array(cl_kernel kernel, cl_mem d_input,
                     int n, int lws_arg,
                     int preferred_lws, cl_command_queue que){
  cl_int err;
  size_t lws[] = { (size_t) (lws_arg>0 ? lws_arg : preferred_lws) };
  size_t gws[] = { (size_t) (round_mul_up(n, lws[0])) };

  // set kernel args
  err = clSetKernelArg(kernel, 0, sizeof(d_input), &d_input);
  ocl_check(err, "set init kernel arg");

  err = clSetKernelArg(kernel, 1, sizeof(int), &n);
  ocl_check(err, "set init kernel arg");

  cl_event evt;
  err = clEnqueueNDRangeKernel(que, kernel, 1, NULL, gws, lws, 
                               0, NULL, &evt);
  ocl_check(err, "execute init kernel");

  return evt;
}

cl_event smooth_array(cl_kernel kernel, cl_mem d_input,
                      cl_mem d_result, int n, int lws_arg,
                      int preferred_lws, cl_command_queue que, 
                      cl_event init_evt){
  cl_int err;
  size_t lws[] = { lws_arg>0 ? lws_arg : preferred_lws };
  size_t gws[] = { round_mul_up(n, lws[0]) };

  // set kernel args
  err = clSetKernelArg(kernel, 0, sizeof(d_input), &d_input);
  ocl_check(err, "set smooth kernel arg");

  err = clSetKernelArg(kernel, 1, sizeof(d_result), &d_result);
  ocl_check(err, "set smooth kernel arg");

  err = clSetKernelArg(kernel, 2, sizeof(cl_int)*(lws[0]+2), NULL);
  ocl_check(err, "set smooth kernel lmem arg");

  err = clSetKernelArg(kernel, 3, sizeof(int), &n);
  ocl_check(err, "set smooth kernel arg");

  cl_event evt;
  err = clEnqueueNDRangeKernel(que, kernel, 1, 0, gws, lws, 
                               1, &init_evt, &evt);
  ocl_check(err, "execute smooth kernel");

  return evt;
}


int main (int argn, char* args[]){
  
  if(argn <= 2){
    printf("must specify number of elements and lws\n");
    exit(1);
  }

  int n = atoi(args[1]);
  int lws = atoi(args[2]);

  cl_int err;

  // boiler
  cl_platform_id p = select_platform();
  cl_device_id d = select_device(p);
  cl_context ctx = create_context(p, d);
  cl_command_queue que = create_queue(ctx, d);
  cl_program prog = create_program("vecsmooth.ocl", ctx, d);

  cl_kernel init_kernel = clCreateKernel(prog, "init_kernel", &err);
  ocl_check(err, "creating init kernel object");

  cl_kernel smooth_kernel = clCreateKernel(prog, "smooth_kernel", &err);
  ocl_check(err, "creating kernel object");

  // allocate memory
  cl_mem d_input = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
                                  sizeof(cl_int)*n, NULL, &err);
  ocl_check(err, "allocate input buffer");

  cl_mem d_result = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
                                  sizeof(cl_int)*n, NULL, &err);
  ocl_check(err, "allocate result buffer");

  // execute init kernel
  cl_event init_evt = init_array(init_kernel, d_input, n, lws, 32, que);
 
  // execute smooth kernel
  cl_event smooth_evt = smooth_array(smooth_kernel, d_input, d_result, 
                                     n, lws, 32, que, init_evt);
  
  cl_event map_evt;
  int* h_result = clEnqueueMapBuffer(que, d_result, CL_TRUE, 
                                     CL_MAP_READ, 0, sizeof(cl_int)*n, 
                                     1, &init_evt, &map_evt, &err); 
  ocl_check(err, "map buffer");
  
  verify(h_result, n);

  clEnqueueUnmapMemObject(que, d_result, h_result, 0, NULL, NULL);

  // benchmarks
  double init_time = runtime_ms(init_evt);
  double smooth_time = runtime_ms(smooth_evt);
  double map_time = runtime_ms(map_evt);

  printf("init time: %lf\n", init_time);
  printf("smooth time: %lf\n", smooth_time);
  printf("map time: %lf\n", map_time);

  // release boiler
  clReleaseMemObject(d_input);
  clReleaseMemObject(d_result);
  clReleaseKernel(init_kernel);
  clReleaseKernel(smooth_kernel);
  clReleaseProgram(prog);
  clReleaseCommandQueue(que);
  clReleaseContext(ctx);

  return 0;
}
