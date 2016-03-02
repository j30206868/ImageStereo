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
#include "cwz_mst.h"
#include "cwz_disparity_generation.h"
#include "cwz_tree_filter_loop_ctrl.h"



// for change window name
#define _AFXDLL
#include <afxwin.h>

//const char* LeftIMGName  = "tsukuba/scene1.row3.col1.ppm"; 
//const char* RightIMGName = "tsukuba/scene1.row3.col3.ppm";
//const char* LeftIMGName  = "face/face1.png"; 
//const char* RightIMGName = "face/face2.png";
//const char* LeftIMGName  = "dolls/dolls1.png"; 
//const char* RightIMGName = "dolls/dolls2.png";
//const char* LeftIMGName  = "structure/struct_left.bmp"; 
//const char* RightIMGName = "structure/struct_right.bmp";
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


int main()
{
	/*cv::Mat testleft = cv::imread("ImgSeries/left52.bmp", CV_LOAD_IMAGE_COLOR);
	cv::Mat testright = cv::imread("ImgSeries/right52.bmp", CV_LOAD_IMAGE_COLOR);
	cv::imwrite("ImgSeries/left52.ppm", testleft);
	cv::imwrite("ImgSeries/right52.ppm", testright);
	show_cv_img("dmap52.ppm", 1, true*/
	/*cv::Mat hand = cv::imread("eye.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat tmp = hand.clone();

	for(int i=0 ; i<=45 ; i++){
		img_rotate_8UC1(1, tmp, hand);

		cv::namedWindow("rightDMap", CV_WINDOW_KEEPRATIO);
		cv::imshow("rightDMap",hand);
		cv::waitKey(0);
	}*/
	const int down_sample_pow = 1;
	
	cwz_mst mstL_b;

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

	match_info info;
	info.img_height = left_b.rows; 
	info.img_width = left_b.cols; 
	info.max_y_d = info.img_height / max_d_to_img_len_pow; 
	info.max_x_d = info.img_width  / max_d_to_img_len_pow; 
	info.node_c = info.img_height * info.img_width;

	dmap_gen dmap_generator;
	dmap_refine dmap_ref;
	dmap_upsam dmap_ups;
	
	dmap_generator.init(context, device, program, err, left, right, sub_info);
	dmap_ref.init(dmap_generator.mst_L, sub_info, dmap_generator.left_dmap, dmap_generator.right_dmap);
	dmap_ups.init(context, device, program, err, down_sample_pow, left_b, info, sub_info, NULL);

	dmap_ups.setup_mst_img();
	
	uchar *left_dmap;
	uchar *right_dmap;
	uchar *refined_dmap;

	int frame_count = 1;

	cv::Mat lastLm = cv::Mat(info.img_height, info.img_width, CV_8UC3);
	cv::Mat lastRm = cv::Mat(info.img_height, info.img_width, CV_8UC3);
	cv::Mat diffLm = cv::Mat(info.img_height, info.img_width, CV_8UC3);
	cv::Mat diffRm = cv::Mat(info.img_height, info.img_width, CV_8UC3);
	int status = CWZ_STATUS_FRAME_BY_FRAME;
	int method = default_method;
	char ch;
	do{
		show_cv_img("左影像", left.data, diffLm.rows, diffLm.cols, 3, false);
		show_cv_img("右影像", right.data, diffRm.rows, diffRm.cols, 3, false);

		cv::Mat refinedDMap(sub_h, sub_w, CV_8UC1);
		if(method == CWZ_MEDTHO_CV_SGNM)
			apply_opencv_stereoSGNM(left, right, refinedDMap, info);
		else if(method == CWZ_METHOD_TREE)
		{
			cwz_timer::t_start();
		
			cwz_timer::start();
			dmap_generator.set_left_right(left, right);
			cwz_timer::time_display("- set_left_right -");

			cwz_timer::start();
			dmap_generator.filtering();
			dmap_generator.compute_cwz_img();
			cwz_timer::time_display("- filtering & compute_cwz_img -");

			cwz_timer::start();
			if( !(left_dmap = dmap_generator.generate_left_dmap()) )
			{printf( "cwz_dmap_generate left_dmap failed...!" );return 0;}
			cwz_timer::time_display("- generate left map -");

			if(CWZ_SHOW_LEFT_DMAP)
				show_cv_img("left_dmap", left_dmap, info.img_height, info.img_width, 1, false);

			cwz_timer::start();
			if( !(right_dmap = dmap_generator.generate_right_dmap()) )
			{printf( "cwz_dmap_generate right_dmap failed...!" );return 0;}
			cwz_timer::time_display("- generate right map -");

			if(CWZ_SHOW_RIGHT_DMAP)
				show_cv_img("right_dmap", right_dmap, info.img_height, info.img_width, 1, false);
		
			//dmap_ref.set_left_right_dmap_value(left_dmap, right_dmap);
			cwz_timer::start();
			refined_dmap = dmap_ref.refinement();
			cwz_timer::time_display("- calc_new_cost_after_left_right_check -");

			uchar *upsampled_dmap;
			if(down_sample_pow > 1){
				//do up sampling
				info.printf_match_info("原影像資訊");
		
				dmap_ups.set_sub_disparity_map(refined_dmap);
				if(	!(upsampled_dmap = dmap_ups.upsampling()) )
				{ printf("cwz_up_sampling failed"); return 0; }
			}
			cwz_timer::t_time_display("total");

			int idx = 0;
			for(int y=0 ; y<sub_h ; y++) for(int x=0 ; x<sub_w ; x++)
			{
				//dMap.at<uchar>(y,x) = nodeList[y][x].dispairty * (double) IntensityLimit / (double)info.max_x_d;
				refinedDMap.at<uchar>(y,x) = refined_dmap[idx] * (double) IntensityLimit / (double)sub_info.max_x_d;
				//dMap.at<uchar>(y,x) = best_disparity[idx];
				idx++;
			}

			if(down_sample_pow > 1){
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
			}

			dmap_generator.mst_L.reinit();
			dmap_generator.mst_R.reinit();
			dmap_ups.mst_b.reinit();
		}//end of method tree filtering
		
		//顯示深度影像 並在window標題加上frame_count編號
		std::stringstream sstm;
		sstm << "深度影像(" << frame_count << ")";
		cv::namedWindow("深度影像", CV_WINDOW_KEEPRATIO);
		HWND hWnd = (HWND)cvGetWindowHandle("深度影像");
		CWnd *wnd = CWnd::FromHandle(hWnd);
		CWnd *wndP = wnd->GetParent();
		wndP->SetWindowText((const char *) sstm.str().c_str()); 
		cv::imshow("深度影像",refinedDMap);
		char inputkey = cv::waitKey(30);

		//儲存深度影像結果
		sstm.str("");
        sstm << "ImgSeries/dmap_" << frame_count << ".bmp";
        cv::imwrite(sstm.str().c_str(), refinedDMap);

		//紀錄與上張影像不同的地方
		for(int i=0 ; i<info.img_height*info.img_width*3 ; i++){
			diffLm.data[i] = std::abs(lastLm.data[i] - left.data[i]);
			diffRm.data[i] = std::abs(lastRm.data[i] - right.data[i]);
		}
		uchar *diff_gray_left = new uchar[info.img_height*info.img_width * 3];
		uchar *diff_gray_right = new uchar[info.img_height*info.img_width * 3];
		for(int i=0 ; i<info.img_height*info.img_width ; i++){
			int _i = i*3;
			int leftdv  = (max_rgb( &(diffLm.data[_i]) ));
			int rightdv = (max_rgb( &(diffRm.data[_i]) ));

			for(int j=0 ; j<3 ; j++)diff_gray_left[_i+j]=0;
			for(int j=0 ; j<3 ; j++)diff_gray_right[_i+j]=0;

			if(leftdv > 50){
				diff_gray_left[_i+1]=leftdv;
			}else if(leftdv > 10){
				diff_gray_left[_i+2]=leftdv * 5;
			}else if(leftdv > 0){
				diff_gray_left[_i]=leftdv * 25;
			}

			if(rightdv > 50){
				diff_gray_right[_i+1]=rightdv;
			}else if(rightdv > 10){
				diff_gray_right[_i+2]=rightdv * 5;
			}else if(rightdv > 0){
				diff_gray_right[_i]=rightdv * 25;
			}
		}

		//show差值圖
		//show_cv_img("leftdiff", diffLm.data, diffLm.rows, diffLm.cols, 3, false);
		//show_cv_img("rightdiff", diffRm.data, diffRm.rows, diffRm.cols, 3, false);
		show_cv_img("左前後差值圖", diff_gray_left, diffLm.rows, diffLm.cols, 3, false);
		show_cv_img("右前後差值圖", diff_gray_right, diffRm.rows, diffRm.cols, 3, false);

		//儲存上一張影像
		for(int i=0 ; i<info.img_height*info.img_width*3 ; i++){
			lastLm.data[i] = left.data[i];
			lastRm.data[i] = right.data[i];
		}

		// Loop Status Control
		int prcResult = processInputKey(inputkey, status, frame_count, method);
		if(prcResult == 0){}//do nothing
		else if(prcResult == 1) continue;
		else break;

		// Read Images for next loop
		frame_count++;
		read_image(left, "ImgSeries/left", frame_count);
		read_image(right, "ImgSeries/right", frame_count);

	//}while((ch = getchar()) != 'e');
	}while(!left.empty()&&!right.empty());
	//

	//cv::imwrite("leftDMap.bmp", leftDMap);
	//cv::imwrite("rightDMap.bmp", rightDMap);

	/*cv::namedWindow("leftDMap", CV_WINDOW_KEEPRATIO);
	cv::imshow("leftDMap",leftDMap);
	cv::waitKey(0);
	cv::namedWindow("rightDMap", CV_WINDOW_KEEPRATIO);
	cv::imshow("rightDMap",rightDMap);
	cv::waitKey(0);*/

	//system("PAUSE");

	clReleaseProgram(program);
	clReleaseContext(context);

	return 0;
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

void apply_opencv_stereoSGNM(cv::Mat &left, cv::Mat &right, cv::Mat &refinedDMap, match_info info){
	int SADWindowSize= 5;
	int numberOfDisparities = info.max_x_d;
	int cn = left.channels();

	cv::Ptr<cv::StereoSGBM> sgbm = new cv::StereoSGBM(0,16,SADWindowSize);
	int color_mode = CV_LOAD_IMAGE_COLOR;
    cv::Size img_size = left.size();
	sgbm->preFilterCap = 63;
	sgbm->SADWindowSize = SADWindowSize;
	sgbm->P1 = 8*cn*SADWindowSize*SADWindowSize;
	sgbm->P2 = 32*cn*SADWindowSize*SADWindowSize;
	sgbm->minDisparity = 0;
	sgbm->numberOfDisparities = numberOfDisparities;
	sgbm->uniquenessRatio = 10;
	sgbm->speckleWindowSize = 100;
	sgbm->speckleRange = 32;
	sgbm->disp12MaxDiff = 1;

    cv::Mat disp;

    int64 t = cv::getTickCount();
    (*sgbm)(left, right, disp);    
    t = cv::getTickCount() - t;
    printf("Time elapsed: %fms\n", t*1000/cv::getTickFrequency());

    disp.convertTo(refinedDMap, CV_8U, 255/(numberOfDisparities*16.));
}

void getVirtualXY(int x, int y, int &vx, int &vy, int w, int h){
	vx = x;
	//vx = x - (w-1)/2;
	vy = y;
	//vy = (h-1) - y;
	//vy = y - (h-1);
	//vy = y - (h-1)/2;
}
void getRealXT(int vx, int vy, int &x, int &y, int w, int h){
	x = vx;
	//x = vx + (w-1)/2;
	
	y = vy;
	//y = (h-1) - vy;
	//y = vy + (h-1);
	//y = vy + (h-1)/2;
}
void img_rotate_8UC1(double degree, cv::Mat tmp, cv::Mat &before){
	int w = before.cols;
	int h = before.rows;

	//inverse mapping 逆推回去before抓對應點到after
	double radian = -1 * (degree * 3.1415926 / 180.0);
	int vx, vy;
	int vbx, vby, bx, by;

	for(int y=0 ; y<before.rows ; y++)
	for(int x=0 ; x<before.cols ; x++)
	{
		getVirtualXY(x, y, vx, vy, w, h);

		vbx = std::floor( vx * cos(radian) - vy * sin(radian) );
		vby = std::floor( vx * sin(radian) + vy * cos(radian) );
		uchar pixel;
	
		getRealXT(vbx, vby, bx, by, w, h);

		if((bx >= 0 && bx <before.cols) &&
		  (by >= 0 && by <before.rows))
		{
			pixel = before.at<uchar>(by, bx);
		}else{
			pixel = 0;
		}

		tmp.at<uchar>(y, x) = pixel;
	}
	
	tmp.copyTo(before);
}