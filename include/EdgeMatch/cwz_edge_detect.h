#ifndef CWZ_EDGE_DETECTOR_H
#define CWZ_EDGE_DETECTOR_H

#include "common_func.h"
#include "cwz_img_proc.h"
#include "GuidedFilter/cwz_integral_img.h"
#include "PxlMatch/cwz_cl_cpp_functions.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

//synchronized header file with edge_detect.cl
#define CWZ_CL_HOST_SIDE
#include "cwz_edge_cl_config.h"
//

#define EDGE_DETECT_CL_FILENAME "edge_detect.cl"
//for 1D kernel setting
#define EDGE_DETECT_1D_EXIST true
#define EDGE_DETECT_1D_KERNEL_ID 1
#define EDGE_DETECT_1D_KERNEL_NAME "edgeDetect_1d"
#define EDGE_DETECT_1D_EXPAND_KERNEL_NAME "edgeDetect_1d_exp"
//

//used to control which kernel is going to be used
#define EDGE_DETECT_DEFAULT_KERNEL_ID EDGE_DETECT_1D_KERNEL_ID
//

void showEdgeKernelInfo_1D(EdgeKernelInfo_1D &info);

class cwz_edge_detector{
private:
	uchar *left_gray;
	uchar *right_gray;
	uchar *left_edge;
	uchar *right_edge;
	
	cl_program program;
	cl_context context; 
	cl_device_id device;
	cl_kernel edge_detect_kernel;
	//
	size_t dim_max_items[3];
	size_t max_group_size;
	cl_ulong max_local_mem;
	//
	int expand_kw;
	int expand_kh;
	//
	int w, h;
	//
#ifdef EDGE_DETECT_1D_EXIST
	EdgeKernelInfo_1D info;
#endif

	int exe_kernel_id;
public:
	bool useExpandImg;

	cl_int err;
	void init(cl_context &_context, cl_device_id &_device, int _w, int _h, bool _useExpandImg, int _exp_w, int _exp_h);//for expand image
	void init(cl_context &context, cl_device_id &device, int _w, int _h, bool _useExpandImg = false);//for default no expand image used
	void releaseRes();
	int edgeDetect(uchar *gray_img, EDGE_DETECT_RESULT_TYPE *result_img);
	void setDeviceInfo();
};

class cwz_lth_proc{
private:
	int max_kw, max_kh;
	uchar *exp_img;
	uchar *exp_result;
	int *exp_int_img;
	int w, h;
	int exp_w, exp_h;
	//
	int ver_kw;
	int ver_kh;
	int hor_kw;
	int hor_kh;
	int sqr_kw;
	int sqr_kh;
	int th;
public:
	uchar *hor_result;
	uchar *ver_result;
	uchar *sqr_result;
	void init(int _w, int _h);
	void doLocalTh(uchar *img);
	uchar *do_sqr(uchar *img);
	void showResult();
	void releaseRes();
};

void cwz_local_threshold(uchar *img, int *exp_int_img, uchar *result, int w, int h, int kw, int kh, int th);

void cwz_local_variance(uchar *img, uchar *result, int w, int h, int kw, int kh, int th);
void cwz_local_th_by_var(uchar *img, uchar *result, int w, int h, int kw, int kh, int th);

#endif //CWZ_EDGE_DETECTOR_H