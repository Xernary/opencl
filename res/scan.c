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

size_t scan_workgroups(cl_int nels, cl_int lws)
{
	cl_int nquarts = nels/4;
	size_t ngroups = round_div_up(nquarts, lws);
	if (ngroups > 1)
		ngroups = round_mul_up(ngroups,4);
	return ngroups;
}

cl_event scan(cl_command_queue que, cl_kernel scan_kernel, cl_event init_evt,
	cl_mem d_out, cl_mem d_tails, cl_mem d_in, int nels, int lws_arg)
{
	cl_int nquarts = nels/4;
	size_t lws[] = { lws_arg };
	size_t gws[] = { round_mul_up(nquarts, lws[0]) };

	printf("scan: %u => %u | %zu = %zu\n", nels, nquarts, lws[0], gws[0]);
	cl_int err;
	cl_event ret;

	int arg = 0;
	err = clSetKernelArg(scan_kernel, arg, sizeof(d_out), &d_out);
	ocl_check(err, "set scan_array arg %d", arg++);
	err = clSetKernelArg(scan_kernel, arg, sizeof(d_tails), &d_tails);
	ocl_check(err, "set scan_array arg %d", arg++);
	err = clSetKernelArg(scan_kernel, arg, sizeof(d_in), &d_in);
	ocl_check(err, "set scan_array arg %d", arg++);
	err = clSetKernelArg(scan_kernel, arg, sizeof(nquarts), &nquarts);
	ocl_check(err, "set scan_array arg %d", arg++);
	err = clSetKernelArg(scan_kernel, arg, lws[0]*sizeof(cl_int), NULL);
	ocl_check(err, "set scan_array arg %d", arg++);

	err = clEnqueueNDRangeKernel(que, scan_kernel, 1,
		NULL, gws, lws,
		1, &init_evt,  &ret);
	ocl_check(err, "enqueue scan");

	return ret;
}

cl_event fixup(cl_command_queue que, cl_kernel fixup_kernel, cl_event init_evt,
	cl_mem d_out, cl_mem d_tails, int nels, int lws_arg)
{
	cl_int nquarts = nels/4;
	size_t lws[] = { lws_arg };
	size_t gws[] = { round_mul_up(nquarts, lws[0]) };

	printf("fixup: %u => %u | %zu = %zu\n", nels, nquarts, lws[0], gws[0]);
	cl_int err;
	cl_event ret;

	int arg = 0;
	err = clSetKernelArg(fixup_kernel, arg, sizeof(d_out), &d_out);
	ocl_check(err, "set fixup_array arg %d", arg++);
	err = clSetKernelArg(fixup_kernel, arg, sizeof(d_tails), &d_tails);
	ocl_check(err, "set fixup_array arg %d", arg++);
	err = clSetKernelArg(fixup_kernel, arg, sizeof(nquarts), &nquarts);
	ocl_check(err, "set fixup_array arg %d", arg++);

	err = clEnqueueNDRangeKernel(que, fixup_kernel, 1,
		NULL, gws, lws,
		1, &init_evt,  &ret);
	ocl_check(err, "enqueue fixup");

	return ret;
}

void verify(const cl_int *scan, int nels)
{
	cl_int expected = 0;
	for (int i = 0; i < nels; ++i) {
		expected += i;
		cl_int val = scan[i];
		if (expected != val) {
			fprintf(stderr, "mismatch @ %d: %d != %d\n", i, val, expected);
		}
	}
}

int main(int argc, char *argv[])
{
	if (argc < 3) error("please specify number of elements and lws");

	int nels = atoi(argv[1]);

	if (nels <= 0) error("please specify a positive integer");
	if (nels & 3) error("please specify a multiple of 4");

	int lws = atoi(argv[2]);

	size_t memsize = nels*sizeof(cl_int);

	size_t ngroups = scan_workgroups(nels, lws);
	size_t tails_memsize = ngroups*sizeof(cl_int);

	size_t tail_scan_groups = scan_workgroups(ngroups, lws);
	if (tail_scan_groups > 1) {
		fprintf(stderr, "%u elements require %zu groups, that require %zu > 1 groups\n",
			nels, ngroups, tail_scan_groups);
		error("too many elements");
	}

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("scan.ocl", ctx, d);

	cl_int err;
	cl_kernel init_kernel = clCreateKernel(prog, "init_kernel", &err);
	ocl_check(err, "create init_kernel");
	cl_kernel scan_kernel = clCreateKernel(prog, "scan", &err);
	ocl_check(err, "create scan_kernel");
	cl_kernel fixup_kernel = clCreateKernel(prog, "scan_fixup", &err);
	ocl_check(err, "create fixup_kernel");

	cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_WRITE, memsize, NULL, &err);
	ocl_check(err, "create d_in failed");
	cl_mem d_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, memsize, NULL, &err);
	ocl_check(err, "create d_out failed");
	cl_mem d_tails = clCreateBuffer(ctx, CL_MEM_READ_WRITE, tails_memsize, NULL, &err);
	ocl_check(err, "create d_tails failed");

	size_t preferred_rounding_init;

	err = clGetKernelWorkGroupInfo(init_kernel, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(preferred_rounding_init), &preferred_rounding_init, NULL);
	ocl_check(err, "get preferred work-group size multiple");

	cl_event init_evt = init_array(que, init_kernel, d_in, nels, preferred_rounding_init, lws);

	cl_event scan_partial_evt = scan(que, scan_kernel, init_evt, d_out, d_tails, d_in, nels, lws);

	cl_event scan_tails_evt, fixup_evt = scan_partial_evt;

	if (ngroups > 1) {
		scan_tails_evt = scan(que, scan_kernel, scan_partial_evt, d_tails, NULL, d_tails, ngroups, lws);
		fixup_evt = fixup(que, fixup_kernel, scan_tails_evt, d_out, d_tails, nels, lws);
	}

	cl_event map_evt;
	cl_int *h_out = clEnqueueMapBuffer(que, d_out, CL_TRUE, CL_MAP_READ, 0, memsize,
		1, &fixup_evt, &map_evt, &err);
	ocl_check(err, "read value");

	verify(h_out, nels);

	clEnqueueUnmapMemObject(que, d_out, h_out, 0, NULL, NULL);
	err = clFinish(que);
	ocl_check(err, "finish");

	double init_runtime = runtime_ms(init_evt);
	double scan_partial_runtime = runtime_ms(scan_partial_evt);
	double scan_tails_runtime = ngroups > 1 ? runtime_ms(scan_tails_evt) : 0.0f;
	double fixup_runtime = ngroups > 1 ? runtime_ms(fixup_evt) : 0.0f;
	double scan_runtime = total_runtime_ms(scan_partial_evt, fixup_evt);
	double map_runtime = runtime_ms(map_evt);

	printf("init: %gms, %gGE/s, %gGB/s\n", init_runtime, nels/init_runtime/1.0e6, memsize/init_runtime/1.e6);
	printf("scan[partial]: %gms, %gGE/s, %gGB/s\n", scan_partial_runtime,
		nels/scan_partial_runtime/1.0e6,
		(2*memsize+(ngroups > 1 ? tails_memsize : 0))/scan_partial_runtime/1.0e6);
	if (ngroups > 1) {
		printf("scan[tails]: %gms, %gGE/s, %gGB/s\n", scan_tails_runtime,
			ngroups/scan_tails_runtime/1.0e6,
			tails_memsize/scan_tails_runtime/1.0e6);
		printf("fixup: %gms, %gGE/s, %gGB/s\n", fixup_runtime,
			nels/fixup_runtime/1.0e6,
			(2*(nels - lws) + ngroups)*sizeof(cl_int)/fixup_runtime/1.0e6);
	}
	printf("scan[total]: %gms, %gGE/s\n", scan_runtime, nels/scan_runtime/1.0e6);
	printf("map: %gms, %gGB/s\n", map_runtime, memsize/map_runtime/1.0e6);

	clReleaseKernel(init_kernel);
	clReleaseKernel(scan_kernel);
	clReleaseKernel(fixup_kernel);
	clReleaseMemObject(d_out);
	clReleaseMemObject(d_tails);
	clReleaseMemObject(d_in);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);
}
