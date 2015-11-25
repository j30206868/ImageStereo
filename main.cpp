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

const char* LeftIMGName  = "tsukuba/scene1.row3.col1.ppm"; 
const char* RightIMGName = "tsukuba/scene1.row3.col3.ppm";
//const char* LeftIMGName  = "face/face1.png"; 
//const char* RightIMGName = "face/face2.png";
//const char* LeftIMGName  = "dolls/dolls1.png"; 
//const char* RightIMGName = "dolls/dolls2.png";
//const char* LeftIMGName  = "structure/struct_left.bmp"; 
//const char* RightIMGName = "structure/struct_right.bmp";

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
						 cv::Mat left,  cv::Mat right, cwz_mst &mst, match_info &info, bool inverse = false)
{
	time_t img_init_s = clock();

	int w = info.img_width;
	int h = info.img_height;
	int node_c = info.node_c;

	mst.init(h, w, 1, info.max_x_d, info.max_y_d);
	
	int *left_color_1d_arr  = c3_mat_to_1d_int_arr(left , h, w);
	int *right_color_1d_arr = c3_mat_to_1d_int_arr(right, h, w);
	float *left_1d_gradient  = new float[node_c];
	float *right_1d_gradient = new float[node_c];
	/************************************************************************
		比較用原圖也不應該做median filtering, 否則也會導致
		深度圖的精確度大大降低
		apply_cl_color_img_mdf<int>(..., bool is_apply_median_filtering_or_not)
	************************************************************************/
	int * left_color_mdf_1d_arr = apply_cl_color_img_mdf<int>(context, device, program, err,  left_color_1d_arr, info, img_pre_mdf);
	int *right_color_mdf_1d_arr = apply_cl_color_img_mdf<int>(context, device, program, err, right_color_1d_arr, info, img_pre_mdf);

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
	if( !(left_gray_1d_arr_for_mst = apply_cl_color_img_mdf<uchar>(context, device, program, err, left_gray_1d_arr, info, mst_pre_mdf)) )
	{ printf("left_gray_1d_arr_for_mst median filtering failed.\n"); return 0; }
	mst.set_img(left_gray_1d_arr_for_mst);
	//mst.profile_mst();
	cwz_timer::start();
	mst.mst();
	if( inverse == false )
		cwz_timer::time_display("Left eye image Minimum spanning tree");
	else
		cwz_timer::time_display("Right eye image Minimum spanning tree");

	int match_result_len = h * w * info.max_x_d;
	float *matching_result = mst.get_agt_result();

	/*******************************************************
							Matching cost
	*******************************************************/
	if( !apply_cl_cost_match(context, device, program, err, 
							left_cwz_img, right_cwz_img, matching_result, match_result_len, info, inverse) )
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
	if( !(final_dmap = apply_cl_color_img_mdf<uchar>(context, device, program, err, best_disparity, info, depth_post_mdf)) )
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
void calc_new_cost_after_left_right_check(float *left_agt, uchar *left_dmap, bool *left_mask, match_info &info){
	int total_len = info.node_c * info.max_x_d;

	memset(left_agt, 0, sizeof(float) * total_len);
	for(int i=0 ; i<total_len ; i+=info.max_x_d) if(left_mask[i/info.max_x_d]){
		for(int d=0; d<info.max_x_d ; d++){
			left_agt[i+d] = std::abs(d - left_dmap[i/info.max_x_d]);
		}
	}
}
uchar* refinement(uchar *left_dmap, uchar *right_dmap, cwz_mst &mstL, cwz_mst &mstR, match_info &info, bool applyTreeRefine = doTreeRefinement){
	int w = info.img_width;
	int h = info.img_height;
	if(applyTreeRefine){
		bool *left_mask = detect_occlusion(left_dmap, right_dmap, h, w, info.node_c, 0);
		calc_new_cost_after_left_right_check(mstL.get_agt_result(), left_dmap, left_mask, info);
		mstL.cost_agt();
		return mstL.pick_best_dispairty();
	}
	bool *left_mask = detect_occlusion(left_dmap, right_dmap, h, w, info.node_c, 0);
	uchar *refined_dmap = new uchar[w*h];
	for(int i=0 ; i<w*h ; i++){
		if(left_mask[i] == false)
			refined_dmap[i] = 0;
		else
			refined_dmap[i] = left_dmap[i];
	}
	return refined_dmap;
}

uchar *cwz_up_sampling(cl_context &context, cl_device_id &device, cl_program &program, cl_int &err,
						 cv::Mat left_b, cwz_mst &mstL_b, match_info &info, match_info &sub_info, uchar *disparity_map, 
						 int down_sample_pow, int disparity_to_img_len_pow, bool do_mst_mdf, bool do_dmap_mdf)
{
	mstL_b.init(info.img_height, info.img_width, 1, info.max_x_d, info.max_y_d);
	mstL_b.updateSigma( cwz_mst::sigma/3 );
	//建原本size大小的tree
	int *left_color_1d_arr  = c3_mat_to_1d_int_arr(left_b , info.img_height, info.img_width);
	uchar *left_gray_1d_arr  = int_1d_arr_to_gray_arr(left_color_1d_arr , info.node_c);
	uchar *left_gray_1d_arr_for_mst;
	if( !(left_gray_1d_arr_for_mst = apply_cl_color_img_mdf<uchar>(context, device, program, err, left_gray_1d_arr, info, do_mst_mdf)) )
	{ printf("left_gray_1d_arr_for_mst median filtering failed.\n"); return 0; }
	mstL_b.set_img(left_gray_1d_arr_for_mst);
	cwz_timer::start();
	mstL_b.mst();
	cwz_timer::time_display("original size left mst");
	//用subsampled depth map算cost
	int cost_len = info.img_width * info.img_height * info.max_x_d;
	float *agt_cost = mstL_b.get_agt_result();
	memset(agt_cost, 0, sizeof(float) * cost_len);
	int cen_ofset = down_sample_pow/2;
	for(int s_y=0 ; s_y<sub_info.img_height; s_y++)
	{
		int y = s_y * down_sample_pow;
		for(int s_x=0 ; s_x<sub_info.img_width; s_x++)
		{
				int x = s_x * down_sample_pow;
				int s_i = s_y * sub_info.img_width + s_x;
				int idx    = ((y + cen_ofset) * info.img_width + (x + cen_ofset)) * info.max_x_d;
				int best_d = disparity_map[ s_i ] * (info.max_x_d / sub_info.max_x_d);

				if(best_d > 2)
					for(int d=0 ; d < info.max_x_d ; d++){
						agt_cost[idx+d] = std::abs(d - best_d);
					}
		}
	}
	//cost aggregate
	uchar *upsampled_dmap;
	mstL_b.cost_agt();
	upsampled_dmap = mstL_b.pick_best_dispairty();
	if( !(upsampled_dmap = apply_cl_color_img_mdf<uchar>(context, device, program, err, upsampled_dmap, info, do_dmap_mdf)) )
	{ printf("left_gray_1d_arr_for_mst median filtering failed.\n"); return 0; }
	return upsampled_dmap;
}

int main()
{
	//cv::Mat hand = cv::imread("hand.ppm", CV_LOAD_IMAGE_COLOR);
	const int down_sample_pow = 1;

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
	cv::Mat left_b = cv::imread(LeftIMGName, CV_LOAD_IMAGE_COLOR);
	cv::Mat right_b = cv::imread(RightIMGName, CV_LOAD_IMAGE_COLOR);

	cv::Mat left; 
	cv::Mat right; 
	cv::resize(left_b, left, cv::Size(left_b.cols/down_sample_pow, left_b.rows/down_sample_pow));
	cv::resize(right_b, right, cv::Size(right_b.cols/down_sample_pow, right_b.rows/down_sample_pow));
	//cvmat_subsampling(left_b , left , 3, down_sample_pow);
	//cvmat_subsampling(right_b, right, 3, down_sample_pow);
	/************************************/

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
	sub_info.img_width = sub_w; 
	sub_info.max_x_d = sub_w / max_d_to_img_len_pow; 
	sub_info.max_y_d = sub_h / max_d_to_img_len_pow; 
	sub_info.node_c = sub_h * sub_w;
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

	/*cv::Mat leftDMap(sub_h, sub_w, CV_8U);
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