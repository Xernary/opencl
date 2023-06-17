#include <stdlib.h>
#include "ocl_boiler.h"
int main(int argn, char* args){

  if(argn < 2) {
    printf("Must specifu global work size\n");
    exit(1);    
  }

  int gws = atoi(args[1]);

  // boiler
  cl_platform p = select_platform();
  cl_device_id d = select_device(p);
  cl_context ctx = create_context(p, d);
  cl_command_queue q = create_queue(ctx, d);
  cl_program p = create_program("vecinit.ocl", ctx, d);

  cl_err err;

  // create kernel object
  cl_program p = create_program(ctx, d)
  cl_kernel init_kernel = clCreateKernel(prog, "vecinit.ocl", err);
  ocl_check(err, "creating kernel");

  // create memory buffer on device 
  return 0;
}
