#ifndef CWZ_CL_DATA_TYPE_H
#define CWZ_CL_DATA_TYPE_H

#include "common_func.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


typedef struct {
	int max_x_d;
	int max_y_d;
	int node_c;
	int img_width;
	int img_height;
	uchar *offset;//ªø«×¬°9

	void printf_match_info(const char *str);
} match_info;

void printDeviceInfo(cl_device_id device);

const char *getErrorString(cl_int error);


#endif