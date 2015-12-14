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
#include "cwz_disparity_generation.h"

const char* LeftIMGName  = "tsukuba/scene1.row3.col1.ppm"; 
const char* RightIMGName = "tsukuba/scene1.row3.col3.ppm";
//const char* LeftIMGName  = "face/face1.png"; 
//const char* RightIMGName = "face/face2.png";
//const char* LeftIMGName  = "dolls/dolls1.png"; 
//const char* RightIMGName = "dolls/dolls2.png";
//const char* LeftIMGName  = "structure/struct_left.bmp"; 
//const char* RightIMGName = "structure/struct_right.bmp";

void img_rotate_8UC1(double degree, cv::Mat before){
	cv::Mat after = cv::Mat(before.rows, before.cols, CV_8UC1);

	//inverse mapping 逆推回去before抓對應點到after
	double radian = -1 * (degree * 3.1415926 / 180.0);

	for(int y=0 ; y<before.rows ; y++)
	for(int x=0 ; x<before.cols ; x++)
	{
		int before_x = std::floor( x * cos(radian) - y * sin(radian) );
		int before_y = std::floor( x * sin(radian) + y * cos(radian) );
		uchar pixel;
	
		if((before_x >= 0 && before_x <before.cols) &&
			(before_y >= 0 && before_y <before.rows) )
		{
			pixel = before.at<uchar>(before_y, before_x);
		}else{
			pixel = 0;
		}

		after.at<uchar>(y, x) = pixel;
	}
}

int main()
{

	cv::Mat hand = cv::imread("face_small_depth_color.ppm", CV_LOAD_IMAGE_GRAYSCALE);
	//img_rotate_8UC1(2, hand);
	const int down_sample_pow = 4;

	cwz_mst mstL_b;
	cwz_mst mstL;
	cwz_mst mstR;

	/*******************************************************
							 OpenCL
	*******************************************************/
	cl_int err;
	cl_context context;
	cl_device_id device = setup_opencl(context, err);

	cl_program program = load_program(context, "test.cl");
	if(program == 0) { std::cerr << "Can't load or build program\n"; clReleaseContext(context); return 0; }

	//cv::Mat ppmimg = cv::imread("hand.ppm");
	//cv::imwrite("hand_mst_no_ctmf.bmp", ppmimg);

	//build MST
	cv::Mat left_b  = cv::imread(LeftIMGName , CV_LOAD_IMAGE_COLOR);
	cv::Mat right_b = cv::imread(RightIMGName, CV_LOAD_IMAGE_COLOR);

	cv::Mat left; 
	cv::Mat right; 
	cv::resize(left_b, left, cv::Size(left_b.cols/down_sample_pow, left_b.rows/down_sample_pow));
	cv::resize(right_b, right, cv::Size(right_b.cols/down_sample_pow, right_b.rows/down_sample_pow));
	//cvmat_subsampling(left_b , left , 3, down_sample_pow);
	//cvmat_subsampling(right_b, right, 3, down_sample_pow);
	/************************************/

	cv::imwrite("face_left_small_color.ppm" , left);
	cv::imwrite("face_right_small_color.ppm", right);

	/*cv::FileStorage fs("imageLR.xml", cv::FileStorage::READ);
    if( fs.isOpened() == false){
        printf( "No More....Quitting...!" );
        return 0;
    }

    cv::Mat matL , matR; //= Mat(480, 640, CV_16UC1);
    fs["left"] >> matL; 
	fs["right"] >> matR;                
    fs.release();

	cv::Mat left_b = cv::Mat(480, 640, CV_8UC3);
	cv::Mat right_b = cv::Mat(480, 640, CV_8UC3);

	for(int y=0; y<left_b.rows ; y++){
		int x_ = 0;
		for(int x=0; x<left_b.cols ; x++)
		{
			uchar lvalue = matL.at<unsigned short>(y, x) / 4;
			left_b.at<uchar>(y, x_  ) = lvalue;
			left_b.at<uchar>(y, x_+1) = lvalue;
			left_b.at<uchar>(y, x_+2) = lvalue;

			uchar rvalue = matR.at<unsigned short>(y, x) / 4;
			right_b.at<uchar>(y, x_  ) = rvalue;
			right_b.at<uchar>(y, x_+1) = rvalue;
			right_b.at<uchar>(y, x_+2) = rvalue;

			x_+=3;
		}
	}

	cv::Mat left; 
	cv::Mat right; 
	cv::resize(left_b, left, cv::Size(left_b.cols/down_sample_pow, left_b.rows/down_sample_pow));
	cv::resize(right_b, right, cv::Size(right_b.cols/down_sample_pow, right_b.rows/down_sample_pow));

	/************************************/
//sub sampling and producing sub sampled depth map
	int sub_w = left.cols;
	int sub_h = left.rows;

	match_info sub_info;
	sub_info.img_height = sub_h; 
	sub_info.img_width  = sub_w; 
	sub_info.max_x_d = sub_w / max_d_to_img_len_pow; 
	sub_info.max_y_d = sub_h / max_d_to_img_len_pow; 
	sub_info.node_c  = sub_h * sub_w;
	sub_info.printf_match_info("縮小影像資訊");

cwz_timer::t_start();
	uchar *left_dmap;
	if( !(left_dmap = cwz_dmap_generate(context, device, program, err, left, right, mstL, sub_info, false)) )
	{printf( "cwz_dmap_generate left_dmap failed...!" );return 0;}

	uchar *right_dmap;
	if( !(right_dmap = cwz_dmap_generate(context, device, program, err, right, left, mstR, sub_info, true)) )
	{printf( "cwz_dmap_generate right_dmap failed...!" );return 0;}

	uchar *refined_dmap;
	cwz_timer::start();
	refined_dmap = refinement(left_dmap, right_dmap, mstL, mstR, sub_info);
	cwz_timer::time_display("calc_new_cost_after_left_right_check");

	cv::Mat refinedDMap(sub_h, sub_w, CV_8U);
	int idx = 0;
	for(int y=0 ; y<sub_h ; y++) for(int x=0 ; x<sub_w ; x++)
	{
		//dMap.at<uchar>(y,x) = nodeList[y][x].dispairty * (double) IntensityLimit / (double)info.max_x_d;
		refinedDMap.at<uchar>(y,x) = refined_dmap[idx] * (double) IntensityLimit / (double)sub_info.max_x_d;
		//dMap.at<uchar>(y,x) = best_disparity[idx];
		idx++;
	}

//do up sampling
	match_info info;
	info.img_height = left_b.rows; 
	info.img_width = left_b.cols; 
	info.max_y_d = info.img_height / max_d_to_img_len_pow; 
	info.max_x_d = info.img_width  / max_d_to_img_len_pow; 
	info.node_c = info.img_height * info.img_width;
	info.printf_match_info("原影像資訊");
	uchar *upsampled_dmap;
	if(	!(upsampled_dmap = cwz_up_sampling(context, device, program, err, left_b, mstL_b, info, sub_info, refined_dmap, 
										   down_sample_pow, max_d_to_img_len_pow, true, true)) )
	{ printf("cwz_up_sampling failed"); return 0; }
	//
cwz_timer::t_time_display("total");

	cv::Mat upDMap(info.img_height, info.img_width, CV_8U);
	idx = 0;
	for(int y=0 ; y<info.img_height ; y++) for(int x=0 ; x<info.img_width ; x++)
	{
		//dMap.at<uchar>(y,x) = nodeList[y][x].dispairty * (double) IntensityLimit / (double)info.max_x_d;
		upDMap.at<uchar>(y,x) = upsampled_dmap[idx] * (double) IntensityLimit / (double)info.max_x_d;
		//dMap.at<uchar>(y,x) = best_disparity[idx];
		idx++;
	}
	cv::namedWindow("upDMap", CV_WINDOW_KEEPRATIO);
	cv::imshow("upDMap",upDMap);
	cv::waitKey(0);

	cv::namedWindow("refinedDMap", CV_WINDOW_KEEPRATIO);
	cv::imshow("refinedDMap",refinedDMap);
	cv::waitKey(0);

	cv::Mat leftDMap(sub_h, sub_w, CV_8U);
	idx = 0;
	for(int y=0 ; y<sub_h ; y++) for(int x=0 ; x<sub_w ; x++)
	{
		//dMap.at<uchar>(y,x) = nodeList[y][x].dispairty * (double) IntensityLimit / (double)info.max_x_d;
		leftDMap.at<uchar>(y,x) = left_dmap[idx] * (double) IntensityLimit / (double)sub_info.max_x_d;
		//dMap.at<uchar>(y,x) = best_disparity[idx];
		idx++;
	}
	cv::Mat rightDMap(sub_h, sub_w, CV_8U);
	idx = 0;
	for(int y=0 ; y<sub_h ; y++) for(int x=0 ; x<sub_w ; x++)
	{
		//dMap.at<uchar>(y,x) = nodeList[y][x].dispairty * (double) IntensityLimit / (double)info.max_x_d;
		rightDMap.at<uchar>(y,x) = right_dmap[idx] * (double) IntensityLimit / (double)sub_info.max_x_d;
		//dMap.at<uchar>(y,x) = best_disparity[idx];
		idx++;
	}
	//

	//cv::imwrite("leftDMap.bmp", leftDMap);
	//cv::imwrite("rightDMap.bmp", rightDMap);

	/*cv::namedWindow("leftDMap", CV_WINDOW_KEEPRATIO);
	cv::imshow("leftDMap",leftDMap);
	cv::waitKey(0);
	cv::namedWindow("rightDMap", CV_WINDOW_KEEPRATIO);
	cv::imshow("rightDMap",rightDMap);
	cv::waitKey(0);*/

	system("PAUSE");

	clReleaseProgram(program);
	clReleaseContext(context);

	return 0;
}