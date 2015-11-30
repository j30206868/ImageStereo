#ifndef CWZ_DISPARITY_GENERATION
#define CWZ_DISPARITY_GENERATION

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "cwz_cl_data_type.h"
#include "cwz_cl_cpp_functions.h"
#include "cwz_mst.h"

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

#endif