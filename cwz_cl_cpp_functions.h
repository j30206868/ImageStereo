#ifndef CWZ_CL_CPP_FUNCTIONS_H
#define CWZ_CL_CPP_FUNCTIONS_H

#include "cwz_cl_data_type.h"
#include "common_func.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

cl_program load_program(cl_context context, const char* filename);

cl_device_id setup_opencl(cl_context &context, cl_int &err);
int apply_cl_cost_match(cl_context &context, cl_device_id &device, cl_program &program, cl_int &err, 
						cl_match_elem *left_cwz_img, cl_match_elem *right_cwz_img, float *matching_result, int match_result_len, match_info &info, bool inverse);

template<class T>
T *apply_cl_color_img_mdf(cl_context &context, cl_device_id &device, cl_program &program, cl_int &err,
						   T *color_1d_arr, match_info &info, bool apply_median_filter)
{
	if(!apply_median_filter){
		return color_1d_arr;
	}

	int w = info.img_width;
	int h = info.img_height;
	int node_c = info.node_c;
							   
	cl_kernel mdf_kernel;

	if(eqTypes<int, T>()){
		mdf_kernel = clCreateKernel(program, "MedianFilterBitonic", 0);
		if(mdf_kernel == 0) { std::cerr << "Can't load MedianFilterBitonic kernel\n"; return 0; }
	}else if(eqTypes<uchar, T>()){
		mdf_kernel = clCreateKernel(program, "MedianFilterGrayScale", 0);
		if(mdf_kernel == 0) { std::cerr << "Can't load MedianFilterGrayScale kernel\n"; return 0; }
	}else{
		return 0;
	}
	cl_mem cl_arr = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * node_c, color_1d_arr, NULL);
	cl_mem cl_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(T) * node_c, NULL, NULL);
	cl_mem cl_match_info = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(match_info), &info, NULL);

	if(cl_arr == 0 || cl_result == 0 || cl_match_info == 0) {
		std::cerr << "Can't create OpenCL buffer for median filter\n";
		clReleaseKernel(mdf_kernel);
		clReleaseMemObject(cl_arr);
		clReleaseMemObject(cl_result);
		clReleaseMemObject(cl_match_info);
		return 0;
	}

	clSetKernelArg(mdf_kernel, 0, sizeof(cl_mem), &cl_arr);
	clSetKernelArg(mdf_kernel, 1, sizeof(cl_mem), &cl_result);
	clSetKernelArg(mdf_kernel, 2, sizeof(cl_mem), &cl_match_info);

	cl_command_queue queue = clCreateCommandQueue(context, device, 0, 0);
	if(queue == 0) {
		std::cerr << "Can't create command queue for median filter\n";
		clReleaseKernel(mdf_kernel);
		clReleaseMemObject(cl_arr);
		clReleaseMemObject(cl_result);
		clReleaseMemObject(cl_match_info);
		return 0;
	}

	T *result_arr = new T[node_c];

	size_t offset_size[2] = {0,0};
	size_t work_size[2] = {w, h};
	
    /* Do your stuff here */
	//cwz_timer::start();
	err = clEnqueueNDRangeKernel(queue, mdf_kernel, 2, offset_size, work_size, 0, 0, 0, 0);

	if(err == CL_SUCCESS) {
		err = clEnqueueReadBuffer(queue, cl_result, CL_TRUE, 0, sizeof(T) * node_c, &result_arr[0], 0, 0, 0);
	}
	//cwz_timer::time_display("Median Filtering");

	if(err != CL_SUCCESS)  {
		std::cout << getErrorString(err) << std::endl;
		std::cerr << "Can't run kernel or read back data for median filter\n";
		delete[] result_arr;
		clReleaseKernel(mdf_kernel);
		clReleaseMemObject(cl_arr);
		clReleaseMemObject(cl_result);
		clReleaseCommandQueue(queue);
		clReleaseMemObject(cl_match_info);
		return 0;
	}

	clReleaseKernel(mdf_kernel);
	clReleaseMemObject(cl_arr);
	clReleaseMemObject(cl_result);
	clReleaseCommandQueue(queue);
	clReleaseMemObject(cl_match_info);

	return result_arr;
}

template<class T>
int apply_cl_color_img_mdf(cl_context &context, cl_device_id &device, cl_program &program, cl_int &err,
						   T *color_1d_arr, T *result_arr, match_info &info, bool apply_median_filter)
{
	if(!apply_median_filter){
		for(int i=0 ; i<info.node_c ; i++)
			result_arr[i] = color_1d_arr[i];
	}

	int w = info.img_width;
	int h = info.img_height;
	int node_c = info.node_c;
							   
	cl_kernel mdf_kernel;

	if(eqTypes<int, T>()){
		mdf_kernel = clCreateKernel(program, "MedianFilterBitonic", 0);
		if(mdf_kernel == 0) { std::cerr << "Can't load MedianFilterBitonic kernel\n"; return 0; }
	}else if(eqTypes<uchar, T>()){
		mdf_kernel = clCreateKernel(program, "MedianFilterGrayScale", 0);
		if(mdf_kernel == 0) { std::cerr << "Can't load MedianFilterGrayScale kernel\n"; return 0; }
	}else{
		return 0;
	}
	cl_mem cl_arr = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * node_c, color_1d_arr, NULL);
	cl_mem cl_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(T) * node_c, NULL, NULL);
	cl_mem cl_match_info = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(match_info), &info, NULL);

	if(cl_arr == 0 || cl_result == 0 || cl_match_info == 0) {
		std::cerr << "Can't create OpenCL buffer for median filter\n";
		clReleaseKernel(mdf_kernel);
		clReleaseMemObject(cl_arr);
		clReleaseMemObject(cl_result);
		clReleaseMemObject(cl_match_info);
		return 0;
	}

	clSetKernelArg(mdf_kernel, 0, sizeof(cl_mem), &cl_arr);
	clSetKernelArg(mdf_kernel, 1, sizeof(cl_mem), &cl_result);
	clSetKernelArg(mdf_kernel, 2, sizeof(cl_mem), &cl_match_info);

	cl_command_queue queue = clCreateCommandQueue(context, device, 0, 0);
	if(queue == 0) {
		std::cerr << "Can't create command queue for median filter\n";
		clReleaseKernel(mdf_kernel);
		clReleaseMemObject(cl_arr);
		clReleaseMemObject(cl_result);
		clReleaseMemObject(cl_match_info);
		return 0;
	}

	size_t offset_size[2] = {0,0};
	size_t work_size[2] = {w, h};
	
    /* Do your stuff here */
	//cwz_timer::start();
	err = clEnqueueNDRangeKernel(queue, mdf_kernel, 2, offset_size, work_size, 0, 0, 0, 0);

	if(err == CL_SUCCESS) {
		err = clEnqueueReadBuffer(queue, cl_result, CL_TRUE, 0, sizeof(T) * node_c, &result_arr[0], 0, 0, 0);
	}
	//cwz_timer::time_display("Median Filtering");

	if(err != CL_SUCCESS)  {
		std::cout << getErrorString(err) << std::endl;
		std::cerr << "Can't run kernel or read back data for median filter\n";
		delete[] result_arr;
		clReleaseKernel(mdf_kernel);
		clReleaseMemObject(cl_arr);
		clReleaseMemObject(cl_result);
		clReleaseCommandQueue(queue);
		clReleaseMemObject(cl_match_info);
		return 0;
	}

	clReleaseKernel(mdf_kernel);
	clReleaseMemObject(cl_arr);
	clReleaseMemObject(cl_result);
	clReleaseCommandQueue(queue);
	clReleaseMemObject(cl_match_info);

	return 1;
}

#endif