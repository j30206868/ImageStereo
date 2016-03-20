#ifndef CWZ_EDGE_DETECTOR_H
#define CWZ_EDGE_DETECTOR_H

#include "common_func.h"
#include "cwz_cl_cpp_functions.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define EDGE_DETECT_RESULT_TYPE uchar
#define EDGE_DETECT_CL_FILENAME "edge_detect.cl"
//for 1D kernel setting
#define EDGE_DETECT_1D_EXIST true
#define EDGE_DETECT_1D_KERNEL_ID 1
#define EDGE_DETECT_1D_KERNEL_NAME "edgeDetect_1d"
//

//used to control which kernel is going to be used
#define EDGE_DETECT_DEFAULT_KERNEL_ID EDGE_DETECT_1D_KERNEL_ID
//

typedef struct{
	cl_int width;
	cl_int half_width;
	cl_int height;
	cl_int xoffset;
	cl_float k1[2];
	cl_float k2[4];
	cl_float k3[8];
} EdgeKernelInfo_1D;
void showEdgeKernelInfo_1D(EdgeKernelInfo_1D &info){
	printf("info.width      :%5d\n", info.width);
	printf("info.half_width :%5d\n", info.half_width);
	printf("info.height     :%5d\n", info.height);
	printf("info.xoffset    :%5d\n", info.xoffset);
}

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
	int w, h;
	//
#ifdef EDGE_DETECT_1D_EXIST
	EdgeKernelInfo_1D info;
#endif

	int exe_kernel_id;
public:
	cl_int err;
	
	void init(cl_context &context, cl_device_id &device, int _w, int _h);
	void releaseRes();
	int edgeDetect(uchar *gray_img, EDGE_DETECT_RESULT_TYPE *result_img);
	void setDeviceInfo();
};

int cwz_edge_detector::edgeDetect(uchar *gray_img, EDGE_DETECT_RESULT_TYPE *result_img){
	//¼È®É©w¸q
	int _1D_K3_Left_len = 4;

	cl_command_queue queue = clCreateCommandQueue(context, device, 0, 0);

	int node_c = w * h;
	cl_mem reuslt   = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(EDGE_DETECT_RESULT_TYPE) * node_c, NULL, NULL);
	cl_mem img_mem  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uchar) * node_c, gray_img, NULL);
#ifdef EDGE_DETECT_1D_EXIST
	cl_mem ker_info = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(EdgeKernelInfo_1D), (&info), NULL);
	//cl_mem ker_info = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(EdgeKernelInfo_1D), &info, NULL);
	//clEnqueueWriteBuffer(queue, ker_info, CL_TRUE, 0, sizeof(EdgeKernelInfo_1D), &info, 0, NULL, NULL);
#endif

	clSetKernelArg(edge_detect_kernel, 0, sizeof(cl_mem), &reuslt);
	clSetKernelArg(edge_detect_kernel, 1, sizeof(cl_mem), &img_mem);
	clSetKernelArg(edge_detect_kernel, 2, sizeof(cl_mem), &ker_info);
	clSetKernelArg(edge_detect_kernel, 3, (info.half_width + _1D_K3_Left_len) * sizeof(uchar), 0 );

	size_t global_work_size = node_c;
	size_t local_work_size = info.half_width;
	err = clEnqueueNDRangeKernel(queue, edge_detect_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

	if(err == CL_SUCCESS) {
		err = clEnqueueReadBuffer(queue, reuslt, CL_TRUE, 0, sizeof(EDGE_DETECT_RESULT_TYPE) * node_c, result_img, 0, 0, 0);
		std::cout << "cwz_edge_detector::edgeDetect: got result\n" << std::endl;
	}else{
		std::cerr << "cwz_edge_detector::edgeDetect: Can't run kernel or read back data\n";	
	}

	clReleaseMemObject(reuslt);
	clReleaseMemObject(img_mem);
	clReleaseMemObject(ker_info);
	clReleaseCommandQueue(queue);
	return true;
}
void cwz_edge_detector::init(cl_context &_context, cl_device_id &_device, int _w, int _h){
	exe_kernel_id = EDGE_DETECT_DEFAULT_KERNEL_ID;

	program = load_program(_context, EDGE_DETECT_CL_FILENAME);
	this->context = _context;
	this->device = _device;
	this->w = _w;
	this->h = _h;
	left_edge  = new uchar[w*h];
	right_edge = new uchar[w*h];
	this->setDeviceInfo();

	if(exe_kernel_id == EDGE_DETECT_1D_KERNEL_ID){
#ifdef EDGE_DETECT_1D_EXIST
		this->info.width = _w;
		this->info.half_width = _w/2;
		this->info.height = _h;
		this->info.xoffset = 0;
		this->info.k1[0] = -1;
		this->info.k1[1] = 1;

		this->info.k2[0] = -1.41421;
		this->info.k2[1] = -1.41421;
		this->info.k2[2] = 1.41421;
		this->info.k2[3] = 1.41421;
	
		this->info.k3[0] = -2;
		this->info.k3[1] = -2;
		this->info.k3[2] = -2;
		this->info.k3[3] = -2;
		this->info.k3[4] = 2;
		this->info.k3[5] = 2;
		this->info.k3[6] = 2;
		this->info.k3[7] = 2;
#endif

		if(this->w % 2 != 0) printf("cwz_edge_detector::Error image width should be multiple of 2.\n");
		if(this->dim_max_items[0] * 2 < this->w) printf("cwz_edge_detector::Error image width is too large.\n");

		edge_detect_kernel = clCreateKernel(program, EDGE_DETECT_1D_KERNEL_NAME, 0);
		if(edge_detect_kernel == 0) { std::cerr << "Can't load " << EDGE_DETECT_1D_KERNEL_NAME << " kernel\n"; system("PAUSE"); }
	}
}
void cwz_edge_detector::setDeviceInfo(){
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dim_max_items), &dim_max_items, NULL);
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_group_size, NULL);
	clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &max_local_mem, NULL);
}
void cwz_edge_detector::releaseRes(){
	clReleaseProgram(program); 
	if(exe_kernel_id == EDGE_DETECT_1D_KERNEL_ID)
		clReleaseKernel(edge_detect_kernel);
	delete[] left_edge;
	delete[] right_edge;
}


#endif //CWZ_EDGE_DETECTOR_H