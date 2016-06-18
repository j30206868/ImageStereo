#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>

#include <time.h>

#include "cwz_config.h"
#include "cwz_tree_filter_loop_ctrl.h"

/*
//const char* LeftIMGName  = "Dataset/tsukuba/scene1.row3.col1.ppm"; 
//const char* RightIMGName = "Dataset/tsukuba/scene1.row3.col3.ppm";
//const char* LeftIMGName  = "Dataset/face/face1.png"; 
//const char* RightIMGName = "Dataset/face/face2.png";
//const char* LeftIMGName  = "Dataset/dolls/dolls1.png"; 
//const char* RightIMGName = "Dataset/dolls/dolls2.png";
//const char* LeftIMGName  = "Dataset/structure/struct_left.bmp"; 
//const char* RightIMGName = "Dataset/structure/struct_right.bmp";*/
const char* LeftIMGName  = "ImgSeries/left01.bmp"; 
const char* RightIMGName = "ImgSeries/right01.bmp";

const char* dmap_out_fname = "ImgSeries/dmap_";

const char *pxlmatch_kernel_path = "./include/PxlMatch/test.cl";
 
void read_image(cv::Mat &stereo_frame, const char *path_and_prefix, int frame_num);
void apply_opencv_stereoSGBM(cv::Mat &left, cv::Mat &right, cv::Mat &refinedDMap, match_info info);
void show1dGradient(const char *str, float *gradient_float, int h, int w);
void showEdge(uchar *left_g, uchar *right_g, match_info &info);
void enhanceDMap(uchar *dmap, match_info &info);
void show_img_diff_with_former(cv::Mat &lastLm, cv::Mat &lastRm, cv::Mat &left, cv::Mat &right, match_info &info);

int main()
{
	const int down_sample_pow = 2;
	/*******************************************************
						OpenCL context setup
	*******************************************************/
	cl_int err;
	cl_context context;
	cl_device_id device = setup_opencl(context, err);

	cl_program program = load_program(context, pxlmatch_kernel_path);
	if(program == 0) { std::cerr << "Can't load or build program\n"; clReleaseContext(context); return 0; }

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
//sub sampling and producing sub sampled depth map
	int sub_w = left.cols;
	int sub_h = left.rows;

	match_info sub_info;
	sub_info.img_height = sub_h; 
	sub_info.img_width  = sub_w; 
	sub_info.max_x_d = sub_w / max_d_to_img_len_pow; 
	sub_info.max_y_d = sub_h / max_d_to_img_len_pow; 
	sub_info.node_c  = sub_h * sub_w;
	sub_info.th = 1;
	sub_info.printf_match_info("縮小影像資訊");

	match_info info;
	info.img_height = left_b.rows; 
	info.img_width = left_b.cols; 
	info.max_y_d = info.img_height / max_d_to_img_len_pow; 
	info.max_x_d = info.img_width  / max_d_to_img_len_pow; 
	info.node_c = info.img_height * info.img_width;
	info.th = 1;

	dmap_gen dmap_generator;
	dmap_refine dmap_ref;
	dmap_upsam dmap_ups;
	
	dmap_generator.init(context, device, program, err, left, right, sub_info);
	dmap_ref.init(dmap_generator.mst_L, sub_info, dmap_generator.left_dmap, dmap_generator.right_dmap);
	//dmap_ups.init(context, device, program, err, down_sample_pow, left_b, info, sub_info, NULL);

	//dmap_ups.setup_mst_img();

	cwz_lth_proc left_th_proc;
	left_th_proc.init(sub_info.img_width, sub_info.img_height);
	
	uchar *left_dmap;
	uchar *right_dmap;
	uchar *refined_dmap;

	int frame_count = 1;

	//紀錄與上一張影像
	cv::Mat lastLm = cv::Mat(sub_info.img_height, sub_info.img_width, CV_8UC3);
	cv::Mat lastRm = cv::Mat(sub_info.img_height, sub_info.img_width, CV_8UC3);

	int status = cwz_loop_ctrl::CV_IMG_STATUS_FRAME_BY_FRAME;
	char ch;
	do{
		show_cv_img("左影像", left.data, left.rows, left.cols, 3, false);
		show_cv_img("右影像", right.data, right.rows, right.cols, 3, false);

		if(cwz_loop_ctrl::Mode == cwz_loop_ctrl::MEDTHO_CV_SGNM){
			cv::Mat refinedDMap(sub_h, sub_w, CV_8UC1);
			apply_opencv_stereoSGBM(left, right, refinedDMap, sub_info);
			//把黑色之外地方的深度全部歸零
            /*uchar *left_color_arr = left.data;
			uchar *refine_arr = refinedDMap.data;
			int max_v = 210;
			int min_v = 100;
			int max_diff = max_v - min_v;
			int step = 255 / (max_v - min_v);
            for(int i=0 ; i<sub_info.node_c*3 ; i+=3){
                int tmp_total = (left_color_arr[i] + left_color_arr[i+1] + left_color_arr[i+2]) / 3.0;
                if(tmp_total >= 170){
                    refine_arr[i/3] = 0;
				}else{
				/*	int tmp = refine_arr[i/3] - min_v;

					if(tmp >= max_diff){
						refine_arr[i/3] = 255;
					}else if(tmp > 0){
						refine_arr[i/3] = refine_arr[i/3] * step;
					}else{
						refine_arr[i/3] = 0;
					}
				*/
				//}
            //}
			write_cv_img(frame_count, dmap_out_fname, refinedDMap);
			show_cv_img(frame_count, "深度影像", refinedDMap, false);
			//cv::imshow("stereoSGBM",refinedDMap);
		}else if((cwz_loop_ctrl::Mode == cwz_loop_ctrl::METHOD_TREE || 
				  cwz_loop_ctrl::Mode == cwz_loop_ctrl::METHOD_TREE_NO_REFINE) || 
				  cwz_loop_ctrl::Mode == cwz_loop_ctrl::METHOD_FILL_SCANLINE)
		{   //update variable value from global config
			info.th			 = cwz_loop_ctrl::Match_Cost_Th;
			sub_info.th		 = cwz_loop_ctrl::Match_Cost_Th;
			info.least_w	 = cwz_loop_ctrl::Match_Cost_Least_W;
			sub_info.least_w = cwz_loop_ctrl::Match_Cost_Least_W;
			cwz_mst::updateSigma(cwz_mst::sigma);
			dmap_generator.doGuildFiltering = cwz_loop_ctrl::Do_Guided_Filer;
			//

			//拿掉left, right中不可能是血管的部分

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
		
			uchar *left_edge = left_th_proc.do_sqr(dmap_generator.left_gray_1d_arr);
			//cwz_mst::updateWtoOne(true);
			cwz_mst::updateSigma(cwz_mst::sigma * 4);
			cwz_timer::start();
			if(cwz_loop_ctrl::Mode == cwz_loop_ctrl::METHOD_FILL_SCANLINE){
				dmap_ref.set_left_edge_map( left_th_proc.do_sqr(dmap_generator.left_gray_1d_arr) );
				refined_dmap = dmap_ref.refinement(dmap_refine::MODE_SCANLINE_FILL);
			}else if(cwz_loop_ctrl::Mode == cwz_loop_ctrl::METHOD_TREE_NO_REFINE){
				refined_dmap = dmap_ref.refinement(dmap_refine::MODE_NO);
			}else{
				refined_dmap = dmap_ref.refinement(dmap_refine::MODE_TREE);
			}
			cwz_timer::time_display("- calc_new_cost_after_left_right_check -");
			//cwz_mst::updateWtoOne(cwz_mst::setWtoOne);
			cwz_mst::updateSigma(cwz_mst::sigma / 4);

			/*if(down_sample_pow > 1){
				//do up sampling
				uchar *upsampled_dmap;
				info.printf_match_info("原影像資訊");
				dmap_ups.set_sub_disparity_map(refined_dmap);
				if(	!(upsampled_dmap = dmap_ups.upsampling()) )
				{ printf("cwz_up_sampling failed"); return 0; }
				enhanceDMap(upsampled_dmap, info);
				show_cv_img("upDMap", upsampled_dmap, info.img_height, info.img_width, 1, false);
			}*/
			cwz_timer::t_time_display("total");

			//把黑色之外地方的深度全部歸零
			for(int i=0 ; i<sub_info.node_c ; i++){
				if(dmap_generator.left_gray_1d_arr[i] >= 170){
					refined_dmap[i] = 0;
				}
			}

			enhanceDMap(refined_dmap, sub_info);
			write_cv_img(frame_count, dmap_out_fname, refined_dmap, sub_info.img_height, sub_info.img_width, CV_8UC1);
			if(CWZ_SHOW_LEFT_DMAP)
				show_cv_img("left_dmap", left_dmap, sub_info.img_height, sub_info.img_width, 1, false);
			if(CWZ_SHOW_RIGHT_DMAP)
				show_cv_img("right_dmap", right_dmap, sub_info.img_height, sub_info.img_width, 1, false);
			//顯示深度影像 並在window標題加上frame_count編號
			show_cv_img(frame_count, "深度影像", refined_dmap, sub_info.img_height, sub_info.img_width, 1, false);

			dmap_generator.mst_L.reinit();
			dmap_generator.mst_R.reinit();
			//if(down_sample_pow > 1)	dmap_ups.mst_b.reinit();
		}//end of method tree filtering
		
		//edge extraction
		//showEdge(dmap_generator.left_gray_1d_arr, dmap_generator.right_gray_1d_arr, sub_info);
		//show1dGradient("Left 1D Gradient", dmap_generator.left_cwz_img->gradient, sub_info.img_height, sub_info.img_width);
		//show1dGradient("Right 1D Gradient", dmap_generator.right_cwz_img->gradient, sub_info.img_height, sub_info.img_width);
		//
		printf("======= cwz_loop_ctrl::Match_Cost_Th     : %.2f ======= \n", cwz_loop_ctrl::Match_Cost_Th);
		printf("======= cwz_loop_ctrl::Match_Cost_Least_W: %.2f ======= \n", cwz_loop_ctrl::Match_Cost_Least_W);
		printf("======= cwz_mst::upbound                 : %.2f ======= \n", cwz_mst::upbound);
		printf("======= cwz_loop_ctrl::Do_Guided_Filer   : %1d  ======= \n", cwz_loop_ctrl::Do_Guided_Filer);

		//顯示與上張影像的不同點
		//show_img_diff_with_former(lastLm, lastRm, left, right, info);
		//儲存上一張影像
		memcpy(lastLm.data, left.data  ,sub_info.node_c * 3);
		memcpy(lastRm.data, right.data ,sub_info.node_c * 3);
		
		// Loop Status Control
		char inputkey = cv::waitKey(30);
		int prcResult = processInputKey(inputkey, status, frame_count);
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
	//
	clReleaseProgram(program);
	clReleaseContext(context);

	return 0;
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
void show1dGradient(const char *str, float *gradient_float, int h, int w){
	double th = 127.5;
	double step = 1.5;
	printf("show1d gradient threshold boundry +-%1.1f\n", step);
	uchar upTh  = th + step;
	uchar btmTh = th - step;
	uchar *gradient_ch = new uchar[w*h];
	for(int i=0 ; i<w*h ; i++){
		uchar tmp = (uchar)cvRound(gradient_float[i]);
		if(tmp > btmTh && tmp < upTh ){
			gradient_ch[i] = 0;
		}else{
			gradient_ch[i] = tmp;
		}
		//gradient_ch[i] = tmp;
	}
	show_cv_img(str, gradient_ch, h, w, 1, false);
}
void showEdge(uchar *left_g, uchar *right_g, match_info &info){
	int edgeThresh = 3;
	int lowThreshold = 10;
	int const max_lowThreshold = 100;
	int ratio = 2;
	int kernel_size = 3;
	cv::Mat left_edge (info.img_height, info.img_width, CV_8UC1);
	cv::Mat right_edge(info.img_height, info.img_width, CV_8UC1);
	//edge extraction
	//left_g.data = dmap_generator.left_gray_1d_arr;
	//right_g.data = dmap_generator.right_gray_1d_arr;
	left_edge.data = left_g;
	right_edge.data = right_g;
	cwz_timer::start();
	Canny( left_edge, left_edge, lowThreshold, lowThreshold*ratio, kernel_size );
	Canny( right_edge, right_edge, lowThreshold, lowThreshold*ratio, kernel_size );
	cwz_timer::time_display("Canny edge of left and right");
	show_cv_img("Left Canny Edge", left_edge.data, left_edge.rows, left_edge.cols, 1, false);
	show_cv_img("Right Canny Edge", right_edge.data, right_edge.rows, right_edge.cols, 1, false);
	//
}
void apply_opencv_stereoSGBM(cv::Mat &left, cv::Mat &right, cv::Mat &refinedDMap, match_info info){
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
void enhanceDMap(uchar *dmap, match_info &info){
	for(int idx=0 ; idx<info.node_c ; idx++)
	{
		dmap[idx] = dmap[idx] * (double) IntensityLimit / (double)info.max_x_d;
	}
}
void show_img_diff_with_former(cv::Mat &lastLm, cv::Mat &lastRm, cv::Mat &left, cv::Mat &right, match_info &info){
	cv::Mat diffLm = cv::Mat(info.img_height, info.img_width, CV_8UC3);
	cv::Mat diffRm = cv::Mat(info.img_height, info.img_width, CV_8UC3);
	uchar *diff_gray_left  = new uchar[info.img_height*info.img_width * 3];
	uchar *diff_gray_right = new uchar[info.img_height*info.img_width * 3];
	for(int i=0 ; i<info.node_c*3 ; i++){
		diffLm.data[i] = std::abs(lastLm.data[i] - left.data[i]);
		diffRm.data[i] = std::abs(lastRm.data[i] - right.data[i]);
	}
	for(int i=0 ; i<info.node_c ; i++){
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
}