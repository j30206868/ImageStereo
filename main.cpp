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
#include <opencv2\gpu\gpu.hpp>

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

using namespace cv;

void ProccTimePrint( unsigned long Atime , string msg)     
{     
 unsigned long Btime=0;     
 float sec, fps;     
 Btime = getTickCount();     
 sec = (Btime - Atime)/getTickFrequency();     
 fps = 1/sec;     
 printf("%s %.4lf(sec) / %.4lf(fps) \n", msg.c_str(),  sec, fps );     
}   

int main()
{
	unsigned long AAtime=0;  
	//image load  
	cv::Mat img = cv::imread(LeftIMGName, CV_LOAD_IMAGE_COLOR);  
	Mat outImg, outimg2;  
  
	//cpu version meanshift  
	AAtime = getTickCount();  
	pyrMeanShiftFiltering(img, outImg, 30, 30, 3);  
	ProccTimePrint(AAtime , "cpu");  
  
  
 //gpu version meanshift  
	gpu::GpuMat pimgGpu, imgGpu, outImgGpu;  
	AAtime = getTickCount();  
	pimgGpu.upload(img);  
	//gpu meanshift only support 8uc4 type.  
	gpu::cvtColor(pimgGpu, imgGpu, CV_BGR2BGRA);  
	gpu::meanShiftFiltering(imgGpu, outImgGpu, 30, 30);  
	outImgGpu.download(outimg2);  
	ProccTimePrint(AAtime , "gpu");  
  
	//show image  
	imshow("origin", img);  
	imshow("MeanShift Filter cpu", outImg);  
	imshow("MeanShift Filter gpu", outimg2);  
  
  
 waitKey();  
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