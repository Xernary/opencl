#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"

int show(const cl_int* h_result, int nrows, int ncols, int pitch_el){
  int el;
  int sum = 0;
  for(int r = 0; r < nrows; r++){
    for(int c = 0; c < ncols; c++){
      el = h_result[r * pitch_el + c];
      if(el == 1) sum += 1;
      printf(" %d ", el);
    } 
    printf("\n");
  }
  printf("\n");
  return sum;
}

void verify(int expected, cl_int obtained){
  printf((expected != obtained ? "mismatch %d != %d\n" : ""), 
         obtained, expected);
}

cl_event init_array(cl_command_queue que, cl_kernel kernel,  
                    cl_mem d_input, int n,
                    int pitch_el, int ngroups, int lws_arg){

  cl_int err;
  cl_event evt;
  size_t lws[] = { lws_arg, lws_arg };
  size_t gws[] = { round_mul_up(n, lws[0]), round_mul_up(n, lws[1]) };

  // set kernel args
  err = clSetKernelArg(kernel, 0, sizeof(d_input), &d_input);
  ocl_check(err, "set init kernel arg");

  err = clSetKernelArg(kernel, 1, sizeof(cl_int), &n);
  ocl_check(err, "set init kernel arg");

  err = clSetKernelArg(kernel, 2, sizeof(cl_int), &pitch_el);
  ocl_check(err, "set init kernel arg");

  err = clEnqueueNDRangeKernel(que, kernel, 2, NULL, gws, lws, 
                               0, NULL, &evt);
  ocl_check(err, "enqueue init kernel");

  return evt;
}

cl_event reduce(cl_command_queue que, cl_kernel kernel,  
                cl_mem d_input, cl_mem d_output, int nquads,
                int pitch_el, int ngroups, int lws_arg, 
                cl_event init_evt){

  cl_int err;
  cl_event evt;
  size_t lws[] = { lws_arg, lws_arg };
  size_t gws[] = { sqrt(ngroups)*lws[0], sqrt(ngroups)*lws[1] };

  err = clSetKernelArg(kernel, 0, sizeof(d_input), &d_input);
  ocl_check(err, "set reduce kernel 0 arg");
  
  err = clSetKernelArg(kernel, 1, sizeof(d_output), &d_output);
  ocl_check(err, "set reduce kernel 1 arg");

  err = clSetKernelArg(kernel, 2, sizeof(cl_int)*lws[0], NULL);
  ocl_check(err, "set reduce kernel 2 arg");

  err = clSetKernelArg(kernel, 3, sizeof(cl_int), &nquads);
  ocl_check(err, "set reduce kernel 3 arg");

  err = clSetKernelArg(kernel, 4, sizeof(cl_int), &pitch_el);
  ocl_check(err, "set reduce kernel 4 arg");

  err = clEnqueueNDRangeKernel(que, kernel, 2, NULL, gws, lws, 
                               0, NULL, &evt);
  ocl_check(err, "enqueue init kernel");

  return evt;
}

int main(int argn, char* args[]){

  if(argn != 4){
    fprintf(stderr, "specify n, lws_side and ngroups\n");
    exit(1);
  }

  int n = atoi(args[1]);
  int lws = atoi(args[2]); // lws side
  int ngroups = atoi(args[3]);

  int nels = n*n;

  //sliding window requirements
  
  if(nels & 3){
    fprintf(stderr, "n*n must be multiple of 4");
    exit(1);
  }

  if(ngroups & 3 && (ngroups != 1)){
    fprintf(stderr, "ngroups must be multiple of 4");
    exit(1);
  }

  if(ngroups & (ngroups - 1) && (ngroups != 1)){
    fprintf(stderr, "ngroups must be power of 2");
    exit(1);
  }

  if(lws*lws & (lws*lws - 1)){
    fprintf(stderr, "lws_side*lws_side must be a power of 2");
    exit(1);
  }


  // boiler 
  cl_platform_id p = select_platform();
  cl_device_id d = select_device(p);
  cl_context ctx = create_context(p, d);
  cl_command_queue que = create_queue(ctx, d);
  cl_program prog = create_program("pigreco.ocl", ctx, d);

  cl_int err;

  // create kernel objects
  cl_kernel init_kernel = clCreateKernel(prog, "init_array", &err);
  ocl_check(err, "creating init kernel object");

  cl_kernel reduce_kernel = clCreateKernel(prog, "reduce_array", &err);
  ocl_check(err, "creating reduce kernel object");

  // pitch

  // get device memory base address
  cl_uint pitch_align;
  err = clGetDeviceInfo(d, CL_DEVICE_MEM_BASE_ADDR_ALIGN, 
                      sizeof(pitch_align), &pitch_align, NULL);
  ocl_check(err, "get pitch align");

  size_t pitch_byte = round_mul_up(n*sizeof(cl_int), pitch_align/8);
  size_t pitch_el = pitch_byte/sizeof(cl_int);

  printf("pitch: %d -> %zu\n", n, pitch_el);

  // allocate device memory buffers
  size_t memsize = sizeof(cl_int) * pitch_el * n;
  cl_mem d_input, d_output;
  d_input = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, memsize, NULL, &err);
  ocl_check(err, "allocating device input buffer");

  d_output = clCreateBuffer(ctx, CL_MEM_READ_WRITE, ngroups*sizeof(cl_int),
                       NULL, &err);
  ocl_check(err, "allocating device output buffer");

  // execute kernels
  cl_event init_evt, reduce_evt;
  init_evt = init_array(que, init_kernel, d_input, n, pitch_el, 
                        ngroups, lws);

  //reduce_evt = reduce(que, reduce_kernel, d_input, d_output, n/4, 
  //                    pitch_el, ngroups, lws, init_evt);
  
  cl_event map_evt;
  int *h_result = clEnqueueMapBuffer(que, d_input, CL_TRUE,
		CL_MAP_READ, 0, memsize,
		1, &init_evt, &map_evt, &err);

  ocl_check(err, "map input buffer");

  clFlush(que);
  ocl_check(err, "queue flush");

  int val = show(h_result, n, n, pitch_el);

  err = clEnqueueUnmapMemObject(que, d_input, h_result,
		0, NULL, NULL);
	ocl_check(err, "unmap buffer");

  reduce_evt = reduce(que, reduce_kernel, d_input, d_output, 
                               n/4, pitch_el, ngroups, lws, init_evt);
  cl_event reduce_evt_2;
  if(ngroups > 1)
    reduce_evt_2 = reduce(que, reduce_kernel, d_input, d_output, 
                          1, pitch_el, ngroups, lws, reduce_evt);

  cl_int h_reduce_result;
  err = clEnqueueReadBuffer(que, d_output, CL_TRUE, 0,
                            sizeof(cl_int), &h_reduce_result, 1,
                            (ngroups==1 ? &reduce_evt : &reduce_evt_2),
                            NULL);
  ocl_check(err, "read buff");
  printf("RESULT: %d\n", h_reduce_result);
  verify(val, h_reduce_result);

  err = clFlush(que);
  ocl_check(err, "flush queue");
  
  // release boiler
  clReleaseKernel(init_kernel);
<<<<<<< HEAD
=======
  clReleaseKernel(reduce_kernel);
>>>>>>> 237adc0baf8f27ff2192e377357819402c19ff45
	clReleaseMemObject(d_input);
	clReleaseMemObject(d_output);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);

  return 0;

}

