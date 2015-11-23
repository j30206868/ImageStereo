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

short *cwz_dmap_generate(cl_context &context, cl_device_id &device, cl_program &program, cl_int &err,
						   cv::Mat &left, cv::Mat &right, cwz_mst &mst, bool inverse = false)
{
	time_t img_init_s = clock();

	int w = left.cols;
	int h = left.rows;
	int node_c = w * h;

	mst.init(h, w, 1);
	
	short *left_color_1d_arr  = lips_c1_mat_to_1d_int_arr(left , h, w);
	short *right_color_1d_arr = lips_c1_mat_to_1d_int_arr(right, h, w);
	float *left_1d_gradient  = new float[node_c];
	float *right_1d_gradient = new float[node_c];
	/************************************************************************
		����έ�Ϥ]�����Ӱ�median filtering, �_�h�]�|�ɭP
		�`�׹Ϫ���T�פj�j���C
		apply_cl_color_img_mdf<int>(..., bool is_apply_median_filtering_or_not)
	************************************************************************/
	short * left_color_mdf_1d_arr = apply_cl_color_img_mdf<short>(context, device, program, err,  left_color_1d_arr, node_c, h, w, false);
	short *right_color_mdf_1d_arr = apply_cl_color_img_mdf<short>(context, device, program, err, right_color_1d_arr, node_c, h, w, false);

	cl_match_elem *left_cwz_img  = new cl_match_elem(node_c, left_color_mdf_1d_arr , left_1d_gradient );
	cl_match_elem *right_cwz_img = new cl_match_elem(node_c, right_color_mdf_1d_arr, right_1d_gradient);
	printf("�}�Cinit��O�ɶ�: %fs\n", double(clock() - img_init_s) / CLOCKS_PER_SEC);
	
	short *left_gray_1d_arr  = left_color_1d_arr;
	short *right_gray_1d_arr = right_color_1d_arr;

	short **left_gray_2d_arr  = map_1d_arr_to_2d_arr<short>(left_gray_1d_arr, h, w);
	short **right_gray_2d_arr = map_1d_arr_to_2d_arr<short>(right_gray_1d_arr, h, w);

	/************************************************************************
				�ΨӲ���gradient���Ƕ��Ϥ��n��median filtering
				�_�h�ҽk����ɷ|���u
	************************************************************************/
	compute_gradient(left_cwz_img->gradient , left_gray_2d_arr , h, w);
	compute_gradient(right_cwz_img->gradient, right_gray_2d_arr, h, w);

	/************************************************************************
		�ΨӰ� mst ���Ƕ��v���i�H��Median filtering
		apply_cl_color_img_mdf<uchar>(..., bool is_apply_median_filtering_or_not)
	************************************************************************/
	short *left_gray_1d_arr_for_mst;
	if( !(left_gray_1d_arr_for_mst = apply_cl_color_img_mdf<short>(context, device, program, err, left_gray_1d_arr, h*w, h, w, false)) )
	{ printf("left_gray_1d_arr_for_mst median filtering failed.\n"); return 0; }
	mst.set_img(left_gray_1d_arr_for_mst);
	mst.profile_mst();

	int match_result_len = h * w * disparityLevel;
	//float *matching_result = new float[match_result_len];
	float *matching_result = mst.get_agt_result();

	/*******************************************************
							Matching cost
	*******************************************************/
	if( !apply_lips_10bits_cl_cost_match(context, device, program, err, 
											left_cwz_img, right_cwz_img, matching_result, h, w, match_result_len, inverse) )
	{ printf("apply_lips_10bits_cl_cost_match failed.\n"); }

	double agt_total_t = 0;
	mst.cost_agt(matching_result, &agt_total_t);

	time_t pick_best_disparity = clock();
	short *best_disparity = mst.pick_best_dispairty();
	if(agt_total_t != 0){
		double tmp;
		printf("get best d time consumption:%2.8fs\n", tmp = double(clock()-pick_best_disparity) / CLOCKS_PER_SEC);
		printf("------------------------------------\n");
		printf("     total time consumption:%2.8fs\n", agt_total_t+tmp);
	}
	/************************************************************************
		���o�`�׹ϫ�i�H��median filtering
		apply_cl_color_img_mdf<uchar>(..., bool is_apply_median_filtering_or_not)
	************************************************************************/
	short *final_dmap;
	if( !(final_dmap = apply_cl_color_img_mdf<short>(context, device, program, err, best_disparity, node_c, h, w, true)) )
	{ printf("dmap median filtering failed.\n"); return 0; }

	return final_dmap;
}


int main()
{
	cwz_mst mstL;
	cwz_mst mstR;
	//mst.test_correctness();

	cv::Mat hand = cv::imread("hand.ppm", CV_LOAD_IMAGE_GRAYSCALE);

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
	//cv::Mat left = cv::imread(LeftIMGName, CV_LOAD_IMAGE_COLOR);
	//cv::Mat right = cv::imread(RightIMGName, CV_LOAD_IMAGE_COLOR);
	
	cv::Mat matL = cv::imread(LeftIMGName, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat matR = cv::imread(RightIMGName, CV_LOAD_IMAGE_GRAYSCALE);

	cv::imwrite("left_hand.ppm", matL);
	cv::imwrite("right_hand.ppm", matR);

	cv::Mat left = cv::Mat(matL.rows, matL.cols, CV_16UC1);
	cv::Mat right = cv::Mat(matR.rows, matR.cols, CV_16UC1);
	for(int y=0; y<left.rows ; y++){
		for(int x=0; x<left.cols ; x++)
		{
			left.at<short>(y, x) = matL.at<uchar>(y, x);// * 4;// / 4;

			right.at<short>(y, x)  = matR.at<uchar>(y, x);// * 4;// / 4;
		}
	}

	/*cv::FileStorage fs("imageLR.xml", cv::FileStorage::READ);
    if( fs.isOpened() == false){
        printf( "No More....Quitting...!" );
        return 0;
    }

    cv::Mat matL , matR; //= Mat(480, 640, CV_16UC1);
    fs["left"] >> matL; 
	fs["right"] >> matR;                
    fs.release();

	cv::Mat left = cv::Mat(480, 640, CV_16UC1);
	cv::Mat right = cv::Mat(480, 640, CV_16UC1);

	for(int y=0; y<left.rows ; y++){
		for(int x=0; x<left.cols ; x++)
		{
			left.at<short>(y, x) = matL.at<unsigned short>(y, x);// / 4;

			right.at<short>(y, x)  = matR.at<unsigned short>(y, x);// / 4;
		}
	}

	/************************************/
	
	int w = left.cols;
	int h = left.rows;

	printf("Image width:%d height:%d\n", w, h);

	short *left_final_dmap;
	short *right_final_dmap;

	if( !(left_final_dmap = cwz_dmap_generate(context, device, program, err, left, right, mstL, false)) )
	{ printf( "left cwz_dmap_generate failed!\n" ); }

	if( !(right_final_dmap = cwz_dmap_generate(context, device, program, err, right, left, mstR, true)) )
	{ printf( "right cwz_dmap_generate failed!\n" ); }

	cv::Mat leftDMap(h, w, CV_8UC1);
	int idx = 0;
	for(int y=0 ; y<h ; y++) for(int x=0 ; x<w ; x++)
	{
		leftDMap.at<uchar>(y,x) = left_final_dmap[idx];// * (double) IntensityLimit / (double)disparityLevel;
		//leftDMap.at<ushort>(y,x) = left_final_dmap[idx] * 64 * (double) IntensityLimit / (double)disparityLevel;
		//dMap.at<ushort>(y,x) = final_dmap[idx];
		idx++;
	}
	cv::Mat rightDMap(h, w, CV_8UC1);
	idx = 0;
	for(int y=0 ; y<h ; y++) for(int x=0 ; x<w ; x++)
	{
		rightDMap.at<uchar>(y,x) = right_final_dmap[idx];// * (double) IntensityLimit / (double)disparityLevel;
		//rightDMap.at<ushort>(y,x) = right_final_dmap[idx] * 64 * (double) IntensityLimit / (double)disparityLevel;
		//dMap.at<ushort>(y,x) = final_dmap[idx];
		idx++;
	}
	//

	cv::imwrite("leftDMap.bmp", leftDMap);
	cv::imwrite("rightDMap.bmp", rightDMap);

	cv::namedWindow("left", CV_WINDOW_KEEPRATIO);
	cv::imshow("left",leftDMap);
	cv::waitKey(0);
	cv::namedWindow("right", CV_WINDOW_KEEPRATIO);
	cv::imshow("right",rightDMap);
	cv::waitKey(0);

	system("PAUSE");

	clReleaseProgram(program);
	clReleaseContext(context);

	return 0;
}