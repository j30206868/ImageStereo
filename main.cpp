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

void read_image(cv::Mat &stereo_frame, const char *path_and_prefix, int frame_num){
	std::stringstream sstm;
	if(frame_num < 10){
		sstm << path_and_prefix << "0" << frame_num << ".bmp";
	}else{
		sstm << path_and_prefix << frame_num << ".bmp";
	}
	stereo_frame = cv::imread(sstm.str(), CV_LOAD_IMAGE_COLOR);
}

int main()
{
	
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

	const int CWZ_STATUS_KEEPGOING = 0;
	const int CWZ_STATUS_FRAME_BY_FRAME = 1;
	const int CWZ_STATUS_MODIFY_PARAM = 2;
	const int CWZ_STATUS_EXIT = 999;

	int status = CWZ_STATUS_FRAME_BY_FRAME;
	char ch;
	do{
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

		cwz_timer::start();
		if( !(right_dmap = dmap_generator.generate_right_dmap()) )
		{printf( "cwz_dmap_generate right_dmap failed...!" );return 0;}
		cwz_timer::time_display("- generate right map -");
		
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

		cv::Mat refinedDMap(sub_h, sub_w, CV_8U);
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

		std::stringstream sstm;
		sstm << "refinedDMap(" << frame_count << ")";
		cv::namedWindow("refinedDMap", CV_WINDOW_KEEPRATIO);

		HWND hWnd = (HWND)cvGetWindowHandle("refinedDMap");
		CWnd *wnd = CWnd::FromHandle(hWnd);
		CWnd *wndP = wnd->GetParent();
		wndP->SetWindowText((const char *) sstm.str().c_str()); 

		cv::imshow("refinedDMap",refinedDMap);
		char inputkey = cv::waitKey(30);

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
		}*/

		sstm.str("");
        sstm << "ImgSeries/dmap_" << frame_count << ".bmp";
        cv::imwrite(sstm.str().c_str(), refinedDMap);

		dmap_generator.mst_L.reinit();
		dmap_generator.mst_R.reinit();
		dmap_ups.mst_b.reinit();

		// Loop Status Control
		do{
			if(inputkey == 'e'){
				status = CWZ_STATUS_EXIT;
			}else if(inputkey == 's'){
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
			}

			inputkey = -1;
			if(status == CWZ_STATUS_MODIFY_PARAM){
				cwz_cmd_processor cmd_proc(&dmap_generator);
				cmd_proc.showRule();
				bool isEnd = cmd_proc.readTreeLoopCommandStr();
				if(isEnd){ status = CWZ_STATUS_FRAME_BY_FRAME; }
			}
			else if(status == CWZ_STATUS_FRAME_BY_FRAME){	inputkey = cv::waitKey(0);  	}
		}while(inputkey != -1);
		if(status == CWZ_STATUS_EXIT){ break; }
		else if(status == CWZ_STATUS_MODIFY_PARAM){
			continue;
		}

		// Read Images for next loop
		frame_count++;
		read_image(left, "ImgSeries/left", frame_count);
		read_image(right, "ImgSeries/right", frame_count);

	//}while((ch = getchar()) != 'e');
	}while(frame_count < 90);
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