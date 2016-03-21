#include "cwz_cl_cpp_functions.h"

cl_program load_program(cl_context context, const char* filename)
{
	std::ifstream in(filename, std::ios_base::binary);
	if(!in.good()) {
		return 0;
	}

	// get file length
	in.seekg(0, std::ios_base::end);
	size_t length = in.tellg();
	in.seekg(0, std::ios_base::beg);

	// read program source
	std::vector<char> data(length + 1);
	in.read(&data[0], length);
	data[length] = 0;

	// create and build program 
	const char* source = &data[0];
	cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0);
	if(program == 0) {
		return 0;
	}

	if(clBuildProgram(program, 0, 0, 0, 0, 0) != CL_SUCCESS) {
		return 0;
	}

	return program;
}

cl_device_id setup_opencl(cl_context &context, cl_int &err){
	cl_uint num;
	err = clGetPlatformIDs(0, 0, &num);
	if(err != CL_SUCCESS) {
		std::cerr << "Unable to get platforms\n";
		return 0;
	}

	std::vector<cl_platform_id> platforms(num);
	err = clGetPlatformIDs(num, &platforms[0], &num);
	if(err != CL_SUCCESS) {
		std::cerr << "Unable to get platform ID\n";
		return 0;
	}
	/*printf("There are %d OpenCL platforms.\n", num);
	for(int i=0 ; i<num ; i++){
		char buffer[10240];
		printf("  -- %d --\n", i+1);
		(clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 10240, buffer, NULL));
		printf("  PROFILE = %s\n", buffer);
		(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 10240, buffer, NULL));
		printf("  VERSION = %s\n", buffer);
		(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 10240, buffer, NULL));
		printf("  NAME = %s\n", buffer);
		(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 10240, buffer, NULL));
		printf("  VENDOR = %s\n", buffer);
		(clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 10240, buffer, NULL));
		printf("  EXTENSIONS = %s\n", buffer);
		printf("  ---------\n");
	}*/

	cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[0]), 0 };
	context = clCreateContextFromType(prop, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, NULL);
	if(context == 0) {
		std::cerr << "Can't create OpenCL context\n";
		return 0;
	}

	
	cl_device_id devices[100];
	cl_uint devices_n = 0;
	// CL_CHECK(clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 100, devices, &devices_n));
	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 100, devices, &devices_n);
	printf("There are %d devices\n", devices_n);
	uchar buffer[1024];
	for(int i=0 ; i<devices_n ; i++){
		printf("device id: %d \n",devices[i] );
		printDeviceInfo(devices[i]);
	}

	return devices[0];
}
int apply_cl_cost_match(cl_context &context, cl_device_id &device, cl_program &program, cl_int &err, 
						cl_match_elem *left_cwz_img, cl_match_elem *right_cwz_img, float *matching_result, int match_result_len, match_info &info, bool inverse)
{
	int w = info.img_width;
	int h = info.img_height;
	//cwz_timer::start();
	cl_kernel matcher;
	if( inverse == false ){
		matcher = clCreateKernel(program, "matching_cost", 0);
		if(matcher == 0) { std::cerr << "Can't load matching_cost kernel\n"; return 0; }
	}else{
		matcher = clCreateKernel(program, "matching_cost_inverse", 0);
		if(matcher == 0) { std::cerr << "Can't load matching_cost_inverse kernel\n"; return 0; }
	}

	time_t step_up_kernel_s = clock();

	cl_mem cl_l_rgb = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * left_cwz_img->node_c, &left_cwz_img->rgb[0], NULL);
	cl_mem cl_l_gradient = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * left_cwz_img->node_c, &left_cwz_img->gradient[0], NULL);

	cl_mem cl_r_rgb = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * right_cwz_img->node_c, &right_cwz_img->rgb[0], NULL);
	cl_mem cl_r_gradient = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * right_cwz_img->node_c, &right_cwz_img->gradient[0], NULL);

	cl_mem cl_match_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * match_result_len, NULL, NULL);

	cl_mem cl_match_info = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(match_info), &info, NULL);

	if(cl_l_rgb == 0 || cl_l_gradient == 0 ||
	   cl_r_rgb == 0 || cl_r_gradient == 0 ||
	   cl_match_result == 0 || cl_match_info == 0) {
		std::cerr << "Can't create OpenCL buffer\n";
		clReleaseKernel(matcher);
		clReleaseMemObject(cl_l_rgb);
		clReleaseMemObject(cl_l_gradient);
		clReleaseMemObject(cl_r_rgb);
		clReleaseMemObject(cl_r_gradient);
		clReleaseMemObject(cl_match_result);
		clReleaseMemObject(cl_match_info);
		return 0;
	}

	clSetKernelArg(matcher, 0, sizeof(cl_mem), &cl_l_rgb);
	clSetKernelArg(matcher, 1, sizeof(cl_mem), &cl_l_gradient);
	clSetKernelArg(matcher, 2, sizeof(cl_mem), &cl_r_rgb);
	clSetKernelArg(matcher, 3, sizeof(cl_mem), &cl_r_gradient);
	clSetKernelArg(matcher, 4, sizeof(cl_mem), &cl_match_result);
	clSetKernelArg(matcher, 5, sizeof(cl_mem), &cl_match_info);
	
	cl_command_queue queue = clCreateCommandQueue(context, device, 0, 0);
	if(queue == 0) {
		std::cerr << "Can't create command queue\n";
		clReleaseKernel(matcher);
		clReleaseMemObject(cl_l_rgb);
		clReleaseMemObject(cl_l_gradient);
		clReleaseMemObject(cl_r_rgb);
		clReleaseMemObject(cl_r_gradient);
		clReleaseMemObject(cl_match_result);
		clReleaseMemObject(cl_match_info);
		return 0;
	}

	size_t work_size = (h * w);
	
    /* Do your stuff here */
	err = clEnqueueNDRangeKernel(queue, matcher, 1, NULL, &work_size, 0, 0, 0, 0);

	if(err == CL_SUCCESS) {
		err = clEnqueueReadBuffer(queue, cl_match_result, CL_TRUE, 0, sizeof(float) * match_result_len, &matching_result[0], 0, 0, 0);
	}

	if(err != CL_SUCCESS ){
		std::cerr << "Can't run kernel or read back data\n";	
	}

	clReleaseKernel(matcher);
	clReleaseMemObject(cl_l_rgb);
	clReleaseMemObject(cl_l_gradient);
	clReleaseMemObject(cl_r_rgb);
	clReleaseMemObject(cl_r_gradient);
	clReleaseMemObject(cl_match_result);
	clReleaseCommandQueue(queue);
	clReleaseMemObject(cl_match_info);
	
	/*if( inverse == false )
		cwz_timer::time_display("Left eye image cost match(+set kernel)");
	else
		cwz_timer::time_display("Right eye image cost match(+set kernel)");*/

	return 1;
}
