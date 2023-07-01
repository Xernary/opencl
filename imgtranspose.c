#include <stdlib.h>
#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"

void verify_init(const int* array, int n){
  for(int i = 0; i < n; i++){
    if(array[i] != i) 
      fprintf(stderr, "mismatch init #%d : %d != %d\n", i, array[i], i);
  }
}

void verify(const int* array, int nrows, int ncols, int pitch_el){
  for(int r = 0; r < nrows; r++)
    for(int c = 0; c < ncols; c++){
      int val = array[r*pitch_el + c];
      int expected = c-r;
      if(expected != val)
        fprintf(stderr, "mismatch #[%d][%d] : %d != %d\n", r, c,
                val, expected);
    }
}

cl_event init_array(cl_kernel kernel, cl_mem d_input,
                     int nrows, int ncols, int lws_arg,
                     cl_command_queue que){
  cl_int err;
  size_t lws[] = { (size_t) lws_arg, (size_t) lws_arg };
  size_t gws[] = { (size_t) (round_mul_up(nrows, lws[0])), (size_t) (round_mul_up(ncols, lws[1])) };

  // set kernel args
  err = clSetKernelArg(kernel, 0, sizeof(d_input), &d_input);
  ocl_check(err, "set init kernel arg");

  cl_event evt;
  err = clEnqueueNDRangeKernel(que, kernel, 2, NULL, gws, lws, 
                               0, NULL, &evt);
  ocl_check(err, "execute init kernel");

  return evt;
}

cl_event transpose(cl_kernel kernel, cl_mem d_input,
                      cl_mem d_output, int nrows, int ncols, int pitch_el,
                      int lws_arg,
                      cl_command_queue que, 
                      cl_event init_evt){
  cl_int err;
  size_t lws[] = { (size_t) lws_arg, (size_t) lws_arg };
  size_t gws[] = { (size_t) (round_mul_up(nrows, lws[0])), (size_t) (round_mul_up(ncols, lws[1])) };

  // set kernel args
  err = clSetKernelArg(kernel, 0, sizeof(d_input), &d_input);
  ocl_check(err, "set transpose kernel arg");

  err = clSetKernelArg(kernel, 1, sizeof(d_output), &d_output);
  ocl_check(err, "set transpose kernel arg");

  err = clSetKernelArg(kernel, 2, sizeof(int), &pitch_el);
  ocl_check(err, "set transpose kernel arg");

  cl_event evt;
  err = clEnqueueNDRangeKernel(que, kernel, 1, 0, gws, lws, 
                               1, &init_evt, &evt);
  ocl_check(err, "execute transpose kernel");

  return evt;
}


int main (int argn, char* args[]){
  
  if(argn <= 3){
    printf("must specify rows, cols and lws\n");
    exit(1);
  }

  int nrows = atoi(args[1]);
  int ncols = atoi(args[2]);
  int lws = atoi(args[3]);

  cl_int err;

  // boiler
  cl_platform_id p = select_platform();
  cl_device_id d = select_device(p);
  cl_context ctx = create_context(p, d);
  cl_command_queue que = create_queue(ctx, d);
  cl_program prog = create_program("imgtranspose.ocl", ctx, d);

  cl_kernel init_kernel = clCreateKernel(prog, "init_array", &err);
  ocl_check(err, "creating init kernel object");

  cl_kernel transpose_kernel = clCreateKernel(prog, "transpose_array", &err);
  ocl_check(err, "creating kernel object");
  
  size_t memsize = sizeof(cl_int) * nrows * ncols;

  cl_image_format array_format = { .image_channel_order = CL_R, 
                  .image_channel_data_type = CL_SIGNED_INT32 };

  cl_image_desc in_desc, out_desc;
  memset(&in_desc, 0, sizeof(in_desc));
  memset(&out_desc, 0, sizeof(out_desc));

  in_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  in_desc.image_width = ncols;
  in_desc.image_height = nrows;

  // pitch
  cl_uint pitch_align;
  err = clGetDeviceInfo(d, CL_DEVICE_MEM_BASE_ADDR_ALIGN, 
                        sizeof(pitch_align), &pitch_align, NULL);
  ocl_check(err, "get pitch align");
  
  size_t pitch_byte = round_mul_up(nrows*sizeof(cl_int), pitch_align/8);
  size_t pitch_el = pitch_byte/sizeof(cl_int);


	printf("pitch: %d => %zu\n", nrows, pitch_el);
  
  // allocate memory
  cl_mem d_input = clCreateImage(ctx, CL_MEM_READ_WRITE, 
                                  &array_format, &in_desc, NULL, &err);
  ocl_check(err, "allocate input image");

  size_t memsize_pitch = pitch_byte * ncols;
  cl_mem d_result = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
                                  memsize, NULL, &err);
  ocl_check(err, "allocate result buffer");

  // execute init kernel
  cl_event init_evt = init_array(init_kernel, d_input, nrows, ncols, 
                                 lws, que);
 
  // execute transpose kernel
  cl_event transpose_evt = transpose(transpose_kernel, d_input,
                                           d_result, nrows, ncols, pitch_el,
                                           lws, que, init_evt);
  
  cl_event map_evt;
  int* h_result = clEnqueueMapBuffer(que, d_result, CL_TRUE, 
                                     CL_MAP_READ, 0, memsize_pitch, 
                                     1, &init_evt, &map_evt, &err); 
  ocl_check(err, "map buffer");
  
  verify(h_result, ncols, nrows, pitch_el);

  clEnqueueUnmapMemObject(que, d_result, h_result, 0, NULL, NULL);

  // benchmarks
  double init_time = runtime_ms(init_evt);
  double transpose_time = runtime_ms(transpose_evt);
  double map_time = runtime_ms(map_evt);

  printf("init time: %lf\n", init_time);
  printf("transpose time: %lf\n", transpose_time);
  printf("map time: %lf\n", map_time);

  // release boiler
  clReleaseMemObject(d_input);
  clReleaseMemObject(d_result);
  clReleaseKernel(init_kernel);
  clReleaseKernel(transpose_kernel);
  clReleaseProgram(prog);
  clReleaseCommandQueue(que);
  clReleaseContext(ctx);

  return 0;
}
