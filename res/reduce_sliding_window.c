#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"
#include <stdio.h>
#include <stdlib.h>

void error(const char *err)
{
	fprintf(stderr, "%s\n", err);
	exit(1);
}

cl_event init_array(cl_command_queue que, cl_kernel init_kernel,
	cl_mem d_in, int nels, size_t preferred_rounding_init, int lws_arg)
{
	size_t lws[] = { lws_arg > 0 ? (size_t)lws_arg : preferred_rounding_init };
	size_t gws[] = { round_mul_up(nels, lws[0]) };

	printf("init: %u | %zu = %zu\n", nels, lws[0], gws[0]);
	cl_int err;
	cl_event ret;

	int arg = 0;
	err = clSetKernelArg(init_kernel, arg, sizeof(d_in), &d_in);
	ocl_check(err, "set init_array arg %d", arg++);
	err = clSetKernelArg(init_kernel, arg, sizeof(nels), &nels);
	ocl_check(err, "set init_array arg %d", arg++);

	err = clEnqueueNDRangeKernel(que, init_kernel, 1,
		NULL, gws, (lws_arg > 0 ? lws : NULL),
		0, NULL,  &ret);
	ocl_check(err, "enqueue init");

	return ret;
}

cl_event reduce(cl_command_queue que, cl_kernel reduce_kernel, cl_event init_evt,
	cl_mem d_out, cl_mem d_in, int nels, int lws_arg, int ngroups)
{
	size_t lws[] = { lws_arg };
	size_t gws[] = { ngroups*lws[0] };

	cl_int err;
	cl_event ret;

	int arg = 0;
	err = clSetKernelArg(reduce_kernel, arg, sizeof(d_out), &d_out);
	ocl_check(err, "set reduce_array arg %d", arg++);
	err = clSetKernelArg(reduce_kernel, arg, sizeof(d_in), &d_in);
	ocl_check(err, "set reduce_array arg %d", arg++);
	err = clSetKernelArg(reduce_kernel, arg, lws[0]*sizeof(cl_int), NULL);
	ocl_check(err, "set reduce_array arg %d", arg++);
	err = clSetKernelArg(reduce_kernel, arg, sizeof(nels), &nels);
	ocl_check(err, "set reduce_array arg %d", arg++);

	err = clEnqueueNDRangeKernel(que, reduce_kernel, 1,
		NULL, gws, lws,
		1, &init_evt,  &ret);
	ocl_check(err, "enqueue reduce");

	return ret;
}


void verify(const cl_int sum, int nels)
{
	int expected = (nels - 1)*(nels/2);
	if (expected != sum)
		fprintf(stderr, "mismatch: %d != %d\n", sum, expected);
}

int main(int argc, char *argv[])
{
	if (argc < 4) error("please specify number of elements, lws, ngroups");

	int nels = atoi(argv[1]);

	if (nels <= 0) error("please specify a positive integer");
	if (nels & 3) error("please specify a multiple of 4");

	int lws = atoi(argv[2]);
	if (lws <= 0) error("lws must be > 0");
	if (lws & (lws - 1)) error("lws must be a power of 2");

	int ngroups = atoi(argv[3]);
	if (ngroups <= 0) error("please specify a positive integer");
	if (ngroups > 1 && (ngroups & 3)) error("please specify a multiple of 4 (groups)");

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("reduce.ocl", ctx, d);

	cl_int err;
	cl_kernel init_kernel = clCreateKernel(prog, "init_kernel", &err);
	ocl_check(err, "create init_kernel");
	cl_kernel reduce_kernel = clCreateKernel(prog, "reduce_lmem_sliding_window", &err);
	ocl_check(err, "create reduce_kernel");

	size_t memsize = nels*sizeof(cl_int);

	cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_WRITE, memsize, NULL, &err);
	ocl_check(err, "create d_in1 failed");
	cl_mem d_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, ngroups*sizeof(cl_int),
		NULL, &err);
	ocl_check(err, "create d_out failed");

	size_t preferred_rounding_init;

	err = clGetKernelWorkGroupInfo(init_kernel, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(preferred_rounding_init), &preferred_rounding_init, NULL);
	ocl_check(err, "get preferred work-group size multiple");

	cl_event init_evt = init_array(que, init_kernel, d_in, nels, preferred_rounding_init, lws);

	int reduction_steps = 1 + (ngroups > 1);
	cl_event reduce_evt[2];

	reduce_evt[0] = reduce(que, reduce_kernel, init_evt, d_out, d_in, nels/4, lws, ngroups);
	if (ngroups > 1) {
		reduce_evt[1] = reduce(que, reduce_kernel, reduce_evt[0], d_out, d_out, ngroups/4, lws, 1);
	}

	cl_int r;

	cl_event read_evt;
	err = clEnqueueReadBuffer(que, d_out, CL_TRUE, 0, sizeof(cl_int),
		&r, 1, reduce_evt + reduction_steps - 1, &read_evt);
	ocl_check(err, "read value");

	verify(r, nels);

	err = clFinish(que);
	ocl_check(err, "finish");

	double init_runtime = runtime_ms(init_evt);
	double reduce_runtime = total_runtime_ms(reduce_evt[0], reduce_evt[reduction_steps-1]);
	double reduce0_runtime = runtime_ms(reduce_evt[0]);
	double reduce1_runtime = reduction_steps > 1 ? runtime_ms(reduce_evt[1]) : 0.0f;
	double read_runtime = runtime_ms(read_evt);

	printf("init: %gms, %gGE/s, %gGB/s\n", init_runtime, nels/init_runtime/1.0e6, memsize/init_runtime/1.e6);
	printf("reduce[0]: %gms, %gGE/s, %gGB/s\n",
		reduce0_runtime, nels/reduce0_runtime/1.0e6,
		(memsize + ngroups*sizeof(cl_int))/reduce0_runtime/1.0e6);
	if (reduction_steps > 1)
		printf("reduce[1]: %gms, %gGE/s, %gGB/s\n",
			reduce1_runtime, nels/reduce1_runtime/1.0e6,
			(ngroups + 1)*sizeof(cl_int)/reduce1_runtime/1.0e6);

	printf("reduce: %gms, %gGE/s\n", reduce_runtime, nels/reduce_runtime/1.0e6);
	printf("read: %gms, %gGB/s\n", read_runtime, sizeof(cl_int)/read_runtime/1.0e6);

	clReleaseKernel(init_kernel);
	clReleaseMemObject(d_out);
	clReleaseMemObject(d_in);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);
}
