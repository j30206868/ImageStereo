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

//const char* LeftIMGName  = "face/face1.png"; 
//const char* RightIMGName = "face/face2.png";
//const char* LeftIMGName  = "dolls/dolls1.png"; 
//const char* RightIMGName = "dolls/dolls2.png";
const char* LeftIMGName  = "structure/struct_left.bmp"; 
const char* RightIMGName = "structure/struct_right.bmp";

void compute_gradient(float*gradient, uchar **gray_image, int h, int w)
{
	float gray,gray_minus,gray_plus;
	int node_idx = 0;
	for(int y=0;y<h;y++)
	{
		gray_minus=gray_image[y][0];
		gray=gray_plus=gray_image[y][1];
		gradient[node_idx]=gray_plus-gray_minus+127.5;

		node_idx++;

		for(int x=1;x<w-1;x++)
		{
			gray_plus=gray_image[y][x+1];
			gradient[node_idx]=0.5*(gray_plus-gray_minus)+127.5;

			gray_minus=gray;
			gray=gray_plus;
			node_idx++;
		}
		
		gradient[node_idx]=gray_plus-gray_minus+127.5;
		node_idx++;
	}
}
uchar *cwz_dmap_generate(cl_context &context, cl_device_id &device, cl_program &program, cl_int &err,
						   cv::Mat left,  cv::Mat right, cwz_mst &mst, bool inverse = false)
{
	time_t img_init_s = clock();

	int w = left.cols;
	int h = left.rows;
	int node_c = w * h;

	mst.init(h, w, 1);
	
	int *left_color_1d_arr  = c3_mat_to_1d_int_arr(left , h, w);
	int *right_color_1d_arr = c3_mat_to_1d_int_arr(right, h, w);
	float *left_1d_gradient  = new float[node_c];
	float *right_1d_gradient = new float[node_c];
	/************************************************************************
		比較用原圖也不應該做median filtering, 否則也會導致
		深度圖的精確度大大降低
		apply_cl_color_img_mdf<int>(..., bool is_apply_median_filtering_or_not)
	************************************************************************/
	int * left_color_mdf_1d_arr = apply_cl_color_img_mdf<int>(context, device, program, err,  left_color_1d_arr, node_c, h, w, img_pre_mdf);
	int *right_color_mdf_1d_arr = apply_cl_color_img_mdf<int>(context, device, program, err, right_color_1d_arr, node_c, h, w, img_pre_mdf);

	cl_match_elem *left_cwz_img  = new cl_match_elem(node_c, left_color_mdf_1d_arr , left_1d_gradient );
	cl_match_elem *right_cwz_img = new cl_match_elem(node_c, right_color_mdf_1d_arr, right_1d_gradient);
	//printf("陣列init花費時間: %fs\n", double(clock() - img_init_s) / CLOCKS_PER_SEC);
	
	uchar *left_gray_1d_arr  = int_1d_arr_to_gray_arr(left_color_1d_arr , node_c);
	uchar *right_gray_1d_arr = int_1d_arr_to_gray_arr(right_color_1d_arr, node_c);

	uchar **left_gray_2d_arr  = map_1d_arr_to_2d_arr<uchar>(left_gray_1d_arr, h, w);
	uchar **right_gray_2d_arr = map_1d_arr_to_2d_arr<uchar>(right_gray_1d_arr, h, w);

	/************************************************************************
				用來產生gradient的灰階圖不要做median filtering
				否則模糊後邊界會失真
	************************************************************************/
	compute_gradient(left_cwz_img->gradient , left_gray_2d_arr , h, w);
	compute_gradient(right_cwz_img->gradient, right_gray_2d_arr, h, w);

	/************************************************************************
		用來做 mst 的灰階影像可以做Median filtering
		apply_cl_color_img_mdf<uchar>(..., bool is_apply_median_filtering_or_not)
	************************************************************************/
	uchar *left_gray_1d_arr_for_mst;
	if( !(left_gray_1d_arr_for_mst = apply_cl_color_img_mdf<uchar>(context, device, program, err, left_gray_1d_arr, h*w, h, w, mst_pre_mdf)) )
	{ printf("left_gray_1d_arr_for_mst median filtering failed.\n"); return 0; }
	mst.set_img(left_gray_1d_arr_for_mst);
	//mst.profile_mst();
	cwz_timer::start();
	mst.mst();
	if( inverse == false )
		cwz_timer::time_display("Left eye image Minimum spanning tree");
	else
		cwz_timer::time_display("Right eye image Minimum spanning tree");

	int match_result_len = h * w * disparityLevel;
	float *matching_result = mst.get_agt_result();

	/*******************************************************
							Matching cost
	*******************************************************/
	if( !apply_cl_cost_match(context, device, program, err, 
							left_cwz_img, right_cwz_img, matching_result, h, w, match_result_len, inverse) )
	{ printf("apply_cl_cost_match failed.\n"); }

	cwz_timer::start();
	mst.cost_agt();
	if( inverse == false )
		cwz_timer::time_display("Left eye image cost_agt");
	else
		cwz_timer::time_display("Right eye image cost_agt");

	cwz_timer::start();
	uchar *best_disparity = mst.pick_best_dispairty();
	if( inverse == false )
		cwz_timer::time_display("Left eye image pick best disparity");
	else
		cwz_timer::time_display("Right eye image pick best disparity");

	/************************************************************************
		取得深度圖後可以做median filtering
		apply_cl_color_img_mdf<uchar>(..., bool is_apply_median_filtering_or_not)
	************************************************************************/
	uchar *final_dmap;
	if( !(final_dmap = apply_cl_color_img_mdf<uchar>(context, device, program, err, best_disparity, node_c, h, w, depth_post_mdf)) )
	{ printf("dmap median filtering failed.\n"); return 0; }
	return final_dmap;
}

bool *detect_occlusion(uchar *left_depth, uchar *right_depth, int h, int w, int node_amt, int th = 0){
	bool *mask = new bool[node_amt];
	memset(mask, true, sizeof(bool) * node_amt);

	bool **left_mask = map_1d_arr_to_2d_arr<bool>(mask , h, w);
	uchar **left_2d  = map_1d_arr_to_2d_arr<uchar>(left_depth , h, w);
	uchar **right_2d = map_1d_arr_to_2d_arr<uchar>(right_depth, h, w);
	
	for(int y=0 ; y<h ; y++){
		for(int x=0 ; x<w ; x++){
			int d = left_2d[y][x];
			int rx = x-d;
			if( rx > 0 ){
				//printf("r:%d l:%d\n",left_2d[y][x],right_2d[y][rx]);
				//getchar();
				if( std::abs(left_2d[y][x] - right_2d[y][rx]) > th ){
					left_mask[y][x] = false;
				}
			}else{
				left_mask[y][x] = false;
			}
		}
	}
	delete[] left_mask;
	delete[] left_2d;
	delete[] right_2d;

	return mask;
}
void calc_new_cost_after_left_right_check(float *left_agt, uchar *left_dmap, bool *left_mask, int h, int w, int node_amt){
	int total_len = node_amt * disparityLevel;

	memset(left_agt, 0, sizeof(float) * total_len);
	for(int i=0 ; i<total_len ; i+=disparityLevel) if(left_mask[i/disparityLevel]){
		for(int d=0; d<disparityLevel ; d++){
			left_agt[i+d] = std::abs(d - left_dmap[i/disparityLevel]);
		}
	}
}
uchar* refinement(uchar *left_dmap, uchar *right_dmap, cwz_mst &mstL, cwz_mst &mstR, int h, int w, bool applyTreeRefine = doTreeRefinement){

	if(applyTreeRefine){
		bool *left_mask = detect_occlusion(left_dmap, right_dmap, h, w, w*h, 0);
		calc_new_cost_after_left_right_check(mstL.get_agt_result(), left_dmap, left_mask, h, w, w*h);
		mstL.cost_agt();
		return mstL.pick_best_dispairty();
	}
	bool *left_mask = detect_occlusion(left_dmap, right_dmap, h, w, w*h, 0);
	uchar *refined_dmap = new uchar[w*h];
	for(int i=0 ; i<w*h ; i++){
		if(left_mask[i] == false)
			refined_dmap[i] = 0;
		else
			refined_dmap[i] = left_dmap[i];
	}
	return refined_dmap;
}

int main()
{
	//cv::Mat hand = cv::imread("hand.ppm", CV_LOAD_IMAGE_COLOR);

	cwz_mst mstL;
	cwz_mst mstR;
	//mst.test_correctness();

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
	cv::Mat left_b = cv::imread(LeftIMGName, CV_LOAD_IMAGE_COLOR);
	cv::Mat right_b = cv::imread(RightIMGName, CV_LOAD_IMAGE_COLOR);

	cv::Mat left; 
	cv::Mat right; 
	cv::resize(left_b, left, cv::Size(left_b.cols/1, left_b.rows/1));
	cv::resize(right_b, right, cv::Size(right_b.cols/1, right_b.rows/1));

	/*cv::FileStorage fs("imageLR.xml", cv::FileStorage::READ);
    if( fs.isOpened() == false){
        printf( "No More....Quitting...!" );
        return 0;
    }

    cv::Mat matL , matR; //= Mat(480, 640, CV_16UC1);
    fs["left"] >> matL; 
	fs["right"] >> matR;                
    fs.release();

	cv::Mat left = cv::Mat(480, 640, CV_8UC3);
	cv::Mat right = cv::Mat(480, 640, CV_8UC3);

	for(int y=0; y<left.rows ; y++){
		int x_ = 0;
		for(int x=0; x<left.cols ; x++)
		{
			uchar lvalue = matL.at<unsigned short>(y, x) / 4;
			left.at<uchar>(y, x_  ) = lvalue;
			left.at<uchar>(y, x_+1) = lvalue;
			left.at<uchar>(y, x_+2) = lvalue;

			uchar rvalue = matR.at<unsigned short>(y, x) / 4;
			right.at<uchar>(y, x_  ) = rvalue;
			right.at<uchar>(y, x_+1) = rvalue;
			right.at<uchar>(y, x_+2) = rvalue;

			x_+=3;
		}
	}

	/************************************/
	int w = left.cols;
	int h = left.rows;
cwz_timer::t_start();
	uchar *left_dmap;
	if( !(left_dmap = cwz_dmap_generate(context, device, program, err, left, right, mstL, false)) )
	{printf( "cwz_dmap_generate left_dmap failed...!" );return 0;}

	uchar *right_dmap;
	if( !(right_dmap = cwz_dmap_generate(context, device, program, err, right, left, mstR, true)) )
	{printf( "cwz_dmap_generate right_dmap failed...!" );return 0;}

	uchar *refined_dmap;
	cwz_timer::start();
	refined_dmap = refinement(left_dmap, right_dmap, mstL, mstR, h, w);
	cwz_timer::time_display("calc_new_cost_after_left_right_check");

	cv::Mat refinedDMap(h, w, CV_8U);
	int idx = 0;
	for(int y=0 ; y<h ; y++) for(int x=0 ; x<w ; x++)
	{
		//dMap.at<uchar>(y,x) = nodeList[y][x].dispairty * (double) IntensityLimit / (double)disparityLevel;
		refinedDMap.at<uchar>(y,x) = refined_dmap[idx] * (double) IntensityLimit / (double)disparityLevel;
		//dMap.at<uchar>(y,x) = best_disparity[idx];
		idx++;
	}

	cv::Mat leftDMap(h, w, CV_8U);
	idx = 0;
	for(int y=0 ; y<h ; y++) for(int x=0 ; x<w ; x++)
	{
		//dMap.at<uchar>(y,x) = nodeList[y][x].dispairty * (double) IntensityLimit / (double)disparityLevel;
		leftDMap.at<uchar>(y,x) = left_dmap[idx] * (double) IntensityLimit / (double)disparityLevel;
		//dMap.at<uchar>(y,x) = best_disparity[idx];
		idx++;
	}
	cv::Mat rightDMap(h, w, CV_8U);
	idx = 0;
	for(int y=0 ; y<h ; y++) for(int x=0 ; x<w ; x++)
	{
		//dMap.at<uchar>(y,x) = nodeList[y][x].dispairty * (double) IntensityLimit / (double)disparityLevel;
		rightDMap.at<uchar>(y,x) = right_dmap[idx] * (double) IntensityLimit / (double)disparityLevel;
		//dMap.at<uchar>(y,x) = best_disparity[idx];
		idx++;
	}
	//
cwz_timer::t_time_display("total");

	cv::imwrite("refinedDMap.bmp", refinedDMap);
	//cv::imwrite("leftDMap.bmp", leftDMap);
	//cv::imwrite("rightDMap.bmp", rightDMap);

	cv::namedWindow("refinedDMap", CV_WINDOW_KEEPRATIO);
	cv::imshow("refinedDMap",refinedDMap);
	cv::waitKey(0);
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