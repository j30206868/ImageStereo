#ifndef CWZ_CL_DATA_TYPE_H
#define CWZ_CL_DATA_TYPE_H

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
	float th;
	float least_w;
	char channel;
	void printf_match_info(const char *str);
} match_info;

match_info *createMatchInfo(int w, int h, int channel);
const char *getErrorString(cl_int error);


#endif