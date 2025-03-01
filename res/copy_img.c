#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"
#include "pamalign.h"
#include <stdio.h>
#include <stdlib.h>

void error(const char *err)
{
	fprintf(stderr, "%s\n", err);
	exit(1);
}


cl_event copy_img(cl_command_queue que, cl_kernel copy_kernel,
	cl_mem d_out, cl_mem d_in, const imgInfo *img)
{
	size_t gws[] = { round_mul_up(img->width, 32), img->height };

	cl_int err;
	cl_event ret;

	cl_int i = 0;
	err = clSetKernelArg(copy_kernel, i, sizeof(d_out), &d_out);
	ocl_check(err, "set copy_array arg %d", i++);
	err = clSetKernelArg(copy_kernel, i, sizeof(d_in), &d_in);
	ocl_check(err, "set copy_array arg %d", i++);

	err = clEnqueueNDRangeKernel(que, copy_kernel, 2,
		NULL, gws, NULL,
		0, NULL,  &ret);
	ocl_check(err, "enqueue copy");

	return ret;
}

int main(void)
{
	imgInfo img;

	if (load_pam("tux.pam", &img))
		exit(1);

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("copy_img.ocl", ctx, d);

	cl_int err;
	cl_kernel copy_kernel = clCreateKernel(prog, "copy_img", &err);
	ocl_check(err, "create copy_kernel");

	if (img.channels != 4) error("only 4-channel images supported");
	if (img.depth != 8) error("only 8-bit-per-channel images supported");

	cl_image_format array_format = { .image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8 };

	cl_image_desc img_desc;
	memset(&img_desc, 0, sizeof(img_desc));

	img_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	img_desc.image_width = img.width;
	img_desc.image_height = img.height;

	cl_mem d_in = clCreateImage(ctx, CL_MEM_READ_WRITE, &array_format, &img_desc, NULL, &err);
	ocl_check(err, "create d_in failed");
	cl_mem d_out = clCreateImage(ctx, CL_MEM_READ_WRITE, &array_format, &img_desc, NULL, &err);
	ocl_check(err, "create d_out failed");

	size_t origin[] = {0, 0, 0};
	size_t region[] = {img.width, img.height, 1 };

	cl_event h2d_evt, d2h_evt;
	err = clEnqueueWriteImage(que, d_in, CL_TRUE, origin, region, 0, 0, img.data,
		0, NULL, &h2d_evt);
	ocl_check(err, "write image");

	cl_event copy_evt = copy_img(que, copy_kernel, d_out, d_in, &img);

	memset(img.data, 0, img.data_size);

	err = clEnqueueReadImage(que, d_out, CL_TRUE, origin, region, 0, 0, img.data,
		1, &copy_evt, &d2h_evt);
	ocl_check(err, "write image");

	clWaitForEvents(1, &d2h_evt);

	save_pam("copia.pam", &img);

	clReleaseKernel(copy_kernel);
	clReleaseMemObject(d_out);
	clReleaseMemObject(d_in);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);
}
