#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>

#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <opencv2\opencv.hpp>

#include "cwz_cl_data_type.h"
#include "cwz_cl_cpp_functions.h"
#include "cwz_mst.h"
#include "cwz_integral_img.h"

//const char* LeftIMGName  = "tsukuba/scene1.row3.col1.ppm"; 
//const char* RightIMGName = "tsukuba/scene1.row3.col3.ppm";
//const char* LeftIMGName  = "face/face1.png"; 
//const char* RightIMGName = "face/face2.png";
const char* LeftIMGName  = "dolls/dolls1.png"; 
const char* RightIMGName = "dolls/dolls2.png";
//const char* LeftIMGName  = "structure/struct_left.bmp"; 
//const char* RightIMGName = "structure/struct_right.bmp";

//void apply_match_cost_to_bf(int max_disparity, int h, int w, double *match_cost, cv::Mat guided_i)
void apply_match_cost_to_bf(int max_disparity, int h, int w, float *match_cost, float *guided_i){
	const int buffer_size = 160;
	cv::Mat inputs[buffer_size];
	cv::Mat outputs[buffer_size];

	if(buffer_size < max_disparity){
		printf("apply_match_cost_to_bf: buffer_size < max_disparity, error.\n");
		system("PAUSE");
	}
	//放入值
	for(int d=0 ; d < max_disparity ; d++){
		inputs[d] = cv::Mat(h, w, CV_32FC1);
		outputs[d] = cv::Mat(h, w, CV_32FC1);
	
		int idx = 0 + d;
		for(int y=0 ; y<h ; y++)
		for(int x=0 ; x<w ; x++)
		{
			inputs[d].at<float>(y, x) = match_cost[idx];

			idx += max_disparity;
		}
	}
	guided_img<float, float> gfilter;
	gfilter.init(guided_i, NULL, w, h);
	//gfilter.img_p = guided_i.data;

	//bilateral
	for(int d=0 ; d < max_disparity ; d++){
		//cv::bilateralFilter(inputs[d], outputs[d], 15, 60, 60 );
		gfilter.img_p = (float *)inputs[d].data;
		gfilter.filter_result = (float *)outputs[d].data;
		gfilter.filtering();
	}

	//放回值
	for(int d=0 ; d < max_disparity ; d++){
		int idx = 0 + d;
		for(int y=0 ; y<h ; y++)
		for(int x=0 ; x<w ; x++)
		{
			match_cost[idx] = outputs[d].at<float>(y, x);

			idx += max_disparity;
		}
	}
}


int main()
{
	cl_int err;
	cl_context context;
	cl_device_id device = setup_opencl(context, err);

	cl_program program = load_program(context, "test.cl");
	if(program == 0) { std::cerr << "Can't load or build program\n"; clReleaseContext(context); return 0; }
	//OpenCL Build

	cv::Mat imL = cv::imread(LeftIMGName , CV_LOAD_IMAGE_COLOR);
	cv::Mat imR = cv::imread(RightIMGName, CV_LOAD_IMAGE_COLOR);

	int *left_color  = c3_mat_to_1d_int_arr(imL, imL.rows, imL.cols);
	int *right_color = c3_mat_to_1d_int_arr(imR, imR.rows, imR.cols);

	int h        = imL.rows;
	int w        = imL.cols;
	int node_amt = w * h;
	
	match_info info;
	info.img_height = h; 
	info.img_width  = w; 
	info.max_y_d = h / max_d_to_img_len_pow; 
	info.max_x_d = w / max_d_to_img_len_pow; 
	info.node_c  = node_amt;
	info.printf_match_info("縮小影像資訊");
	//image information set

	uchar *left_gray  = int_1d_arr_to_gray_arr(left_color , node_amt);
	uchar *right_gray = int_1d_arr_to_gray_arr(right_color, node_amt);
	float *left_1d_gradient  = new float[node_amt];
	float *right_1d_gradient = new float[node_amt];

	//uchar **left_gray_2d_arr  = map_1d_arr_to_2d_arr<uchar>(left_gray , h, w);
	//uchar **right_gray_2d_arr = map_1d_arr_to_2d_arr<uchar>(right_gray, h, w);

	//int * left_color_mdf_1d_arr = apply_cl_color_img_mdf<int>(context, device, program, err,  left_color, info, img_pre_mdf);
	//int *right_color_mdf_1d_arr = apply_cl_color_img_mdf<int>(context, device, program, err, right_color, info, img_pre_mdf);

	/// 做像素的正規化
	float *nor_left_gray = new float[node_amt];
	float *nor_right_gray = new float[node_amt];
	float *nor_left_rgb = new float[node_amt * 3];
	float *nor_right_rgb = new float[node_amt * 3];
	
	int max_intensity = IntensityLimit - 1;
	normalize_arr<uchar>(left_gray , nor_left_gray , node_amt, max_intensity);
	normalize_arr<uchar>(right_gray, nor_right_gray, node_amt, max_intensity);
	normalize_arr<uchar>(imL.data, nor_left_rgb , node_amt * 3, max_intensity);
	normalize_arr<uchar>(imR.data, nor_right_rgb, node_amt * 3, max_intensity);

	cl_nor_match_elem *left_cwz_img  = new cl_nor_match_elem(node_amt, nor_left_rgb , left_1d_gradient );
	cl_nor_match_elem *right_cwz_img = new cl_nor_match_elem(node_amt, nor_right_rgb, right_1d_gradient);

	compute_nor_gradient(left_cwz_img->gradient , nor_left_gray , h, w);
	compute_nor_gradient(right_cwz_img->gradient, nor_right_gray, h, w);
	//cost match info gotten

	cwz_mst mst;
	mst.init(h, w, 1, info.max_x_d, info.max_y_d);
	mst.set_img(nor_left_gray);
	//mst set

	int match_result_len   = w * h * info.max_x_d;
	float *matching_result = mst.get_agt_result();
	bool inverse = false; // if it is right eye refer to left eye
	//if( !apply_cl_cost_match(context, device, program, err, 
	//						left_cwz_img, right_cwz_img, matching_result, match_result_len, info, inverse) )
	if( !apply_cl_nor_cost_match(context, device, program, err, 
							 left_cwz_img, right_cwz_img, matching_result, match_result_len, info, inverse) )
	{ printf("apply_cl_cost_match failed.\n"); }

	cwz_timer::start();
	apply_match_cost_to_bf(info.max_x_d, h, w, mst.get_agt_result(), nor_left_gray);
	cwz_timer::time_display("Running Time of Guided image filtering");
	//
	float *best_disparity = mst.pick_best_dispairty();
	cv::Mat disparityMap(info.img_height, info.img_width, CV_8U);
	int idx = 0;
	for(int y=0 ; y<info.img_height ; y++) for(int x=0 ; x<info.img_width ; x++)
	{
		//dMap.at<uchar>(y,x) = nodeList[y][x].dispairty * (double) IntensityLimit / (double)info.max_x_d;
		disparityMap.at<uchar>(y,x) = best_disparity[idx] * (double) IntensityLimit / (double)info.max_x_d;
		//dMap.at<uchar>(y,x) = best_disparity[idx];
		idx++;
	}
	cv::namedWindow("test", CV_WINDOW_KEEPRATIO);
	cv::imshow("test",disparityMap);
	cv::waitKey(0);
	//顯示深度圖

	//get match cost*/
	/*
	cv::Mat imL = cv::imread(LeftIMGName , CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat imLR;

	cv::bilateralFilter(imL, imLR, 15, 30, 30 );
	
	cv::namedWindow("imLR", CV_WINDOW_KEEPRATIO);
	cv::imshow("imLR",imLR);
	cv::waitKey(0);
	*/
	return 0;
}