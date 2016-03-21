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
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include "cwz_cl_data_type.h"
#include "cwz_cl_cpp_functions.h"
#include "cwz_tree_filter_loop_ctrl.h"
#include "cwz_integral_img.h"
#include "cwz_edge_detect.h"
#include "cwz_img_proc.h"

// for change window name
#define _AFXDLL
#include <afxwin.h>

const char* LeftIMGName  = "ImgSeries/left01.bmp"; 
const char* RightIMGName = "ImgSeries/right01.bmp";

const int CWZ_STATUS_KEEPGOING = 0;
const int CWZ_STATUS_FRAME_BY_FRAME = 1;
const int CWZ_STATUS_MODIFY_PARAM = 2;
const int CWZ_STATUS_EXIT = 999;

const int CWZ_METHOD_TREE = 1;
const int CWZ_MEDTHO_CV_SGNM = 2;

void read_image(cv::Mat &stereo_frame, const char *path_and_prefix, int frame_num){
	std::stringstream sstm;
	if(frame_num < 10){
		sstm << path_and_prefix << "0" << frame_num << ".bmp";
	}else{
		sstm << path_and_prefix << frame_num << ".bmp";
	}
	stereo_frame = cv::imread(sstm.str(), CV_LOAD_IMAGE_COLOR);
}

int processInputKey(int inputkey, int &status, int &frame_count, int &method);//will return shouldbreak or not
void apply_opencv_stereoSGNM(cv::Mat &left, cv::Mat &right, cv::Mat &refinedDMap, match_info info);
void show1dGradient(const char *str, float *gradient_float, uchar *gradient_ch, int h, int w);
void showEdge(cv::Mat &left, cv::Mat &right, cv::Mat &left_g, cv::Mat &right_g, cv::Mat &left_edge, cv::Mat &right_edge, int lowThreshold, int ratio, int kernel_size);

int main()
{
	const int down_sample_pow = 1;

	/*******************************************************
							 OpenCL
	*******************************************************/
	cl_int err;
	cl_context context;
	cl_device_id device = setup_opencl(context, err);

	//cl_program edge_detect_cl = load_program(context, "edge_detect.cl");
	//if(edge_detect_cl == 0) { std::cerr << "Can't load or build edge_detect_cl\n"; clReleaseContext(context); return 0; }

	cv::Mat left_b  = cv::imread(LeftIMGName , CV_LOAD_IMAGE_COLOR);
	cv::Mat right_b = cv::imread(RightIMGName, CV_LOAD_IMAGE_COLOR);

	cv::Mat left; 
	cv::Mat right; 
	cv::resize(left_b, left, cv::Size(left_b.cols/down_sample_pow, left_b.rows/down_sample_pow));
	cv::resize(right_b, right, cv::Size(right_b.cols/down_sample_pow, right_b.rows/down_sample_pow));

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

	match_info info;
	info.img_height = left_b.rows; 
	info.img_width = left_b.cols; 
	info.max_y_d = info.img_height / max_d_to_img_len_pow; 
	info.max_x_d = info.img_width  / max_d_to_img_len_pow; 
	info.node_c = info.img_height * info.img_width;

	//for edge extraction
	int edgeThresh = 3;
	int lowThreshold = 5;
	int const max_lowThreshold = 100;
	int ratio = 2;
	int kernel_size = 3;
	char* window_name = "Edge Map";
	cv::Mat left_g(sub_info.img_height, sub_info.img_width, CV_8UC1);
	cv::Mat right_g(sub_info.img_height, sub_info.img_width, CV_8UC1);
	//
	//for gradient
	uchar *left_grad_ch = new uchar[sub_info.node_c];
	uchar *right_grad_ch = new uchar[sub_info.node_c];
	//

	//guided filtering
	guided_img<float, float> *gfilter;
	float *normalized_left_gray_img;
	float *normalized_right_gray_img;
	bool doGuildFiltering = DoGuidedFiltering;
	gfilter = new guided_img<float, float>();
	gfilter->init(NULL, NULL, sub_info.img_width, sub_info.img_height);
	normalized_left_gray_img  = new float[sub_info.node_c];
	normalized_right_gray_img = new float[sub_info.node_c];
	//

	//texture analysis
	cwz_texture_analyzer t_analyzer;
	t_analyzer.init(sub_info.img_width, sub_info.img_height);
	uchar *left_exp  = t_analyzer.createEmptyExpandImg();
	uchar *right_exp = t_analyzer.createEmptyExpandImg();
	//
	cwz_lth_proc left_th_proc;
	left_th_proc.init(sub_info.img_width, sub_info.img_height);
	cwz_lth_proc right_th_proc;
	right_th_proc.init(sub_info.img_width, sub_info.img_height);
	//
	
	bool useExpandImg = true;
	cwz_edge_detector edgeDetector;
	edgeDetector.init(context, device, t_analyzer.exp_w, t_analyzer.exp_h, useExpandImg, t_analyzer.expand_kw, t_analyzer.expand_kh);

	uchar *left_dmap;
	uchar *right_dmap;
	uchar *refined_dmap;

	int frame_count = 1;

	int status = CWZ_STATUS_FRAME_BY_FRAME;
	int method = default_method;
	char ch;
	do{
		//show_cv_img("左影像", left.data, left.rows, left.cols, 3, false);
		//show_cv_img("右影像", right.data, right.rows, right.cols, 3, false);

		cv::Mat edgeMap(sub_h, sub_w, CV_8UC1);
		
		cvtColor( left, left_g, CV_BGR2GRAY );
		cvtColor( right, right_g, CV_BGR2GRAY );

		if(doGuildFiltering){
			apply_gray_guided_img_filtering<float, float, float>
				(left_g.data, sub_info.img_height, sub_info.img_width, normalized_left_gray_img, *gfilter);
			apply_gray_guided_img_filtering<float, float, float>
				(right_g.data, sub_info.img_height, sub_info.img_width, normalized_right_gray_img, *gfilter);
		}
		show_cv_img("左影像(gray)", left_g.data, left.rows, left.cols, 1, false);
		show_cv_img("右影像(gray)", right_g.data, right.rows, right.cols, 1, false);

		cwz_timer::start();
		t_analyzer.expandImgBorder(left_g.data , left_exp);
		t_analyzer.expandImgBorder(right_g.data, right_exp);
		cwz_timer::time_display("expandImgBorder left and right");

		cv::Mat left_edge(t_analyzer.exp_h, t_analyzer.exp_w, CV_8UC1);
		cv::Mat right_edge(t_analyzer.exp_h, t_analyzer.exp_w, CV_8UC1);
		cwz_timer::start();
		edgeDetector.edgeDetect(left_exp, left_edge.data);
		edgeDetector.edgeDetect(right_exp, right_edge.data);
		cwz_timer::time_display("edgeDetect left and right");

		cwz_timer::start();
		left_th_proc.doLocalTh (left_g.data);
		right_th_proc.doLocalTh(right_g.data);
		cwz_timer::time_display("Local Threshold 3diff kernel for left and right");
		//left_th_proc.showResult();
		right_th_proc.showResult();

		show_cv_img("left_edge", left_edge.data, left_edge.rows, left_edge.cols, 1, false);
		show_cv_img("right_edge", right_edge.data, right_edge.rows, right_edge.cols, 1, false);

		//顯示深度影像 並在window標題加上frame_count編號
		std::stringstream sstm;
		/*sstm << "深度影像(" << frame_count << ")";
		cv::namedWindow("深度影像", CV_WINDOW_KEEPRATIO);
		HWND hWnd = (HWND)cvGetWindowHandle("深度影像");
		CWnd *wnd = CWnd::FromHandle(hWnd);
		CWnd *wndP = wnd->GetParent();
		wndP->SetWindowText((const char *) sstm.str().c_str()); 
		cv::imshow("深度影像",edgeMap);*/
		char inputkey = cv::waitKey(30);

		//儲存深度影像結果
		sstm.str("");
        sstm << "ImgSeries/dmap_" << frame_count << ".bmp";
        cv::imwrite(sstm.str().c_str(), edgeMap);

		
		//edge extraction
		//showEdge(left, right, left_g, right_g, left_edge, right_edge, lowThreshold, ratio, kernel_size);
		//

		// Loop Status Control
		int prcResult = processInputKey(inputkey, status, frame_count, method);
		if(prcResult == 0){}//do nothing
		else if(prcResult == 1) continue;
		else break;

		// Read Images for next loop
		frame_count++;
		
		if(down_sample_pow == 1){
			read_image(left, "ImgSeries/left", frame_count);
			read_image(right, "ImgSeries/right", frame_count);
		}else{
			read_image(left_b, "ImgSeries/left", frame_count);
			read_image(right_b, "ImgSeries/right", frame_count);
			cv::resize(left_b, left, cv::Size(left_b.cols/down_sample_pow, left_b.rows/down_sample_pow));
			cv::resize(right_b, right, cv::Size(right_b.cols/down_sample_pow, right_b.rows/down_sample_pow));
		}

	//}while((ch = getchar()) != 'e');
	}while(!left.empty()&&!right.empty());

	//clReleaseProgram(edge_detect_cl);
	clReleaseContext(context);
	edgeDetector.releaseRes();

	return 0;
}

void showEdge(cv::Mat &left, cv::Mat &right, cv::Mat &left_g, cv::Mat &right_g, cv::Mat &left_edge, cv::Mat &right_edge, int lowThreshold, int ratio, int kernel_size){
	//edge extraction
	//left_g.data = dmap_generator.left_gray_1d_arr;
	//right_g.data = dmap_generator.right_gray_1d_arr;
	cvtColor( left, left_g, CV_BGR2GRAY );
	cvtColor( right, right_g, CV_BGR2GRAY );
	blur( left_g, left_edge, cv::Size(5,5) );
	blur( right_g, right_edge, cv::Size(5,5) );
	cwz_timer::start();
	Canny( left_edge, left_edge, lowThreshold, lowThreshold*ratio, kernel_size );
	Canny( right_edge, right_edge, lowThreshold, lowThreshold*ratio, kernel_size );
	cwz_timer::time_display("Canny edge of left and right");
	show_cv_img("Left Edge", left_edge.data, left_edge.rows, left_edge.cols, 1, false);
	show_cv_img("Right Edge", right_edge.data, right_edge.rows, right_edge.cols, 1, false);
	//
}

int processInputKey(int inputkey, int &status, int &frame_count, int &method){
	enum{ result_nothing = 0, result_continue = 1, result_break = 2};
	do{
		if(inputkey == 'e'){
			status = CWZ_STATUS_EXIT;
		}else if(inputkey == 's' || inputkey == 'f'){
			status = CWZ_STATUS_FRAME_BY_FRAME;
		}else if(inputkey == 'p'){
			status = CWZ_STATUS_MODIFY_PARAM;
		}else if(inputkey == ','){//懶的+shift所以直接用跟<同格的,
			if(frame_count != 1)
				frame_count-=2;
			else//避免讀到index為00或負的檔案結果爆掉
				frame_count = 0;
			status = CWZ_STATUS_FRAME_BY_FRAME;
			break;
		}else if(inputkey == '.'){//懶的+shift所以直接用跟>同格的.
			status = CWZ_STATUS_FRAME_BY_FRAME;
			break;
		}else if(inputkey == 'k'){
			status = CWZ_STATUS_KEEPGOING;
		}else if(inputkey == 'm'){
			if(method == CWZ_MEDTHO_CV_SGNM)
				method = CWZ_METHOD_TREE;
			else
				method = CWZ_MEDTHO_CV_SGNM;
			frame_count--;
			break;
		}

		inputkey = -1;
		if(status == CWZ_STATUS_MODIFY_PARAM){
			cwz_cmd_processor cmd_proc(&frame_count);
			cmd_proc.showRule();
			bool isEnd = cmd_proc.readTreeLoopCommandStr();
			if(isEnd){ status = CWZ_STATUS_FRAME_BY_FRAME; }
		}
		else if(status == CWZ_STATUS_FRAME_BY_FRAME){	inputkey = cv::waitKey(0);  	}
	}while(inputkey != -1);
	if(status == CWZ_STATUS_EXIT){ return result_break; }
	else if(status == CWZ_STATUS_MODIFY_PARAM){
		return result_continue;
	}
	return result_nothing;
}