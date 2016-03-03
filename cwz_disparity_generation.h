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

class dmap_gen{
private:
	match_info info;
	int channel;

	cl_context context;
	cl_device_id device;
	cl_program program; 
	cl_int err;

	cl_match_elem *left_cwz_img;
	cl_match_elem *right_cwz_img;

	float *left_1d_gradient;
	float *right_1d_gradient;

	int *left_color_1d_arr;
	int *right_color_1d_arr;

	uchar *left_gray_1d_arr;
	uchar *right_gray_1d_arr;
	uchar **left_gray_2d_arr;
	uchar **right_gray_2d_arr;

	int * left_color_mdf_1d_arr;
	int *right_color_mdf_1d_arr;

	//for mst
	uchar *left_img_arr_for_mst;
	uchar *right_img_arr_for_mst;
	//for channel = 3 mst
	int *left_color_1d_for_3_mst;
	int *right_color_1d_for_3_mst;
public:
	uchar *left_dmap;
	uchar *right_dmap;

	cwz_mst mst_L;
	cwz_mst mst_R;

	void init(cl_context &context, cl_device_id &device, cl_program &program, cl_int &err, 
		      cv::Mat left, cv::Mat right, match_info _info);
	void set_left_right(cv::Mat left, cv::Mat right);
	void filtering();
	void compute_cwz_img();
	uchar *generate_left_dmap();
	uchar *generate_right_dmap();
};
void dmap_gen::init(cl_context &_context, cl_device_id &_device, cl_program &_program, cl_int &_err,
					cv::Mat left, cv::Mat right, match_info _info){
	info = _info;
	int h = info.img_height;
	int w = info.img_width;
	int node_c = info.node_c;

	channel = mst_channel;

	context = _context;
	device  = _device;
	program = _program;
	err     = _err;
	
	left_color_1d_arr  = c3_mat_to_1d_int_arr(left , h, w);
	right_color_1d_arr = c3_mat_to_1d_int_arr(right, h, w);

	left_gray_1d_arr = new uchar[info.node_c];
	right_gray_1d_arr = new uchar[info.node_c];

	left_gray_2d_arr  = map_1d_arr_to_2d_arr<uchar>(left_gray_1d_arr , info.img_height, info.img_width);
	right_gray_2d_arr = map_1d_arr_to_2d_arr<uchar>(right_gray_1d_arr, info.img_height, info.img_width);

	left_1d_gradient  = new float[node_c];
	right_1d_gradient = new float[node_c];

	left_cwz_img  = new cl_match_elem();
	right_cwz_img = new cl_match_elem();

	left_dmap = new uchar[info.node_c];
	right_dmap = new uchar[info.node_c];

	if( channel == 1 ){
		left_img_arr_for_mst  = new uchar[info.node_c];
		right_img_arr_for_mst = new uchar[info.node_c];
	}else{
		left_img_arr_for_mst  = new uchar[info.node_c * 3];
		right_img_arr_for_mst = new uchar[info.node_c * 3];

		left_color_1d_for_3_mst = new int[info.node_c];
		right_color_1d_for_3_mst = new int[info.node_c];
	}

	mst_L.init(h, w, channel, info.max_x_d, info.max_y_d);
	mst_R.init(h, w, channel, info.max_x_d, info.max_y_d);
}
void dmap_gen::set_left_right(cv::Mat left, cv::Mat right){
	left_color_1d_arr  = c3_mat_to_1d_int_arr(left , info.img_height, info.img_width);
	right_color_1d_arr = c3_mat_to_1d_int_arr(right, info.img_height, info.img_width);
}
void dmap_gen::filtering(){
	/************************************************************************
		比較用原圖也不應該做median filtering, 否則也會導致
		深度圖的精確度大大降低
		apply_cl_color_img_mdf<int>(..., bool is_apply_median_filtering_or_not)
	************************************************************************/
	left_color_mdf_1d_arr = apply_cl_color_img_mdf<int>(context, device, program, err,  left_color_1d_arr, info, img_pre_mdf);
	right_color_mdf_1d_arr = apply_cl_color_img_mdf<int>(context, device, program, err, right_color_1d_arr, info, img_pre_mdf);

	if( channel == 1 ){
		if( !(apply_cl_color_img_mdf<uchar>(context, device, program, err, left_gray_1d_arr, left_img_arr_for_mst, info, mst_pre_mdf)) )
		{ printf("left_gray_1d_arr_for_mst median filtering failed.\n"); system("PAUSE"); }

		if( !(apply_cl_color_img_mdf<uchar>(context, device, program, err, right_gray_1d_arr, right_img_arr_for_mst, info, mst_pre_mdf)) )
		{ printf("left_gray_1d_arr_for_mst median filtering failed.\n"); system("PAUSE"); }

	}else{
		apply_cl_color_img_mdf<int>(context, device, program, err,  left_color_1d_arr, left_color_1d_for_3_mst, info, mst_pre_mdf);
		int_1d_color_to_uchar_1d_color(left_color_1d_for_3_mst, left_img_arr_for_mst, info.node_c);

		apply_cl_color_img_mdf<int>(context, device, program, err,  right_color_1d_arr, right_color_1d_for_3_mst, info, mst_pre_mdf);
		int_1d_color_to_uchar_1d_color(right_color_1d_for_3_mst, right_img_arr_for_mst, info.node_c);
	}
}
void dmap_gen::compute_cwz_img(){
	left_cwz_img->node_c = info.node_c;
	left_cwz_img->gradient = left_1d_gradient;
	left_cwz_img->rgb = left_color_mdf_1d_arr;

	right_cwz_img->node_c = info.node_c;
	right_cwz_img->gradient = right_1d_gradient;
	right_cwz_img->rgb = right_color_mdf_1d_arr;

	int_1d_to_gray_arr(left_color_1d_arr , left_gray_1d_arr , info.node_c);
	int_1d_to_gray_arr(right_color_1d_arr, right_gray_1d_arr, info.node_c);

	/************************************************************************
				用來產生gradient的灰階圖不要做median filtering
				否則模糊後邊界會失真
	************************************************************************/
	compute_gradient(left_cwz_img->gradient , left_gray_2d_arr , info.img_height, info.img_width);
	compute_gradient(right_cwz_img->gradient, right_gray_2d_arr, info.img_height, info.img_width);
}
uchar *dmap_gen::generate_left_dmap(){
	/************************************************************************
		用來做 mst 的灰階影像可以做Median filtering
		apply_cl_color_img_mdf<uchar>(..., bool is_apply_median_filtering_or_not)
	************************************************************************/
	mst_L.set_img(left_img_arr_for_mst);

	mst_L.profile_mst();
	//mst_L.mst();

	int match_result_len = info.img_height * info.img_width * info.max_x_d;
	float *matching_result = mst_L.get_agt_result();

	/*******************************************************
							Matching cost
	*******************************************************/
	if( !apply_cl_cost_match(context, device, program, err, 
							left_cwz_img, right_cwz_img, matching_result, match_result_len, info, false) )
	{ printf("generate_left_dmap: apply_cl_cost_match failed.\n"); }

	mst_L.cost_agt();

	uchar *best_disparity = mst_L.pick_best_dispairty();

	/************************************************************************
		取得深度圖後可以做median filtering
		apply_cl_color_img_mdf<uchar>(..., bool is_apply_median_filtering_or_not)
	************************************************************************/
	if( !(apply_cl_color_img_mdf<uchar>(context, device, program, err, best_disparity, left_dmap, info, depth_post_mdf)) )
	{ printf("dmap median filtering failed.\n"); return 0; }
	return left_dmap;
}
uchar *dmap_gen::generate_right_dmap(){
	/************************************************************************
		用來做 mst 的灰階影像可以做Median filtering
		apply_cl_color_img_mdf<uchar>(..., bool is_apply_median_filtering_or_not)
	************************************************************************/
	mst_R.set_img(right_img_arr_for_mst);

	//mst.profile_mst();
	mst_R.mst();

	int match_result_len = info.img_height * info.img_width * info.max_x_d;
	float *matching_result = mst_R.get_agt_result();

	/*******************************************************
							Matching cost
	*******************************************************/
	if( !apply_cl_cost_match(context, device, program, err, 
							right_cwz_img, left_cwz_img, matching_result, match_result_len, info, true) )
	{ printf("generate_right_dmap: apply_cl_cost_match failed.\n"); }

	mst_R.cost_agt();

	uchar *best_disparity = mst_R.pick_best_dispairty();

	/************************************************************************
		取得深度圖後可以做median filtering
		apply_cl_color_img_mdf<uchar>(..., bool is_apply_median_filtering_or_not)
	************************************************************************/
	if( !(apply_cl_color_img_mdf<uchar>(context, device, program, err, best_disparity, right_dmap, info, depth_post_mdf)) )
	{ printf("dmap median filtering failed.\n"); return 0; }
	return right_dmap;
}

class dmap_upsam{
private:
	int channel;
	match_info info;
	match_info sub_info;

	cl_context context;
	cl_device_id device;
	cl_program program; 
	cl_int err;

	int down_sample_pow;
	uchar *sub_disparity_map;

	bool do_mst_mdf;
	bool do_dmap_mdf;

	int *img_color_1d_arr;

	//for channel = 1
	uchar *img_gray_1d_arr;
	uchar *left_gray_1d_arr_for_mst;
	//for channel = 3
	int *left_color_1d_for_3_mst;
	uchar *left_color_1d_arr_uchar;

	uchar *upsampled_dmap;

public:
	cwz_mst mst_b;

	void init(cl_context &_context, cl_device_id &_device, cl_program &_program, cl_int &_err,
		       int _down_sample_pow, cv::Mat img_b, match_info &_info, match_info &_sub_info, uchar *_sub_disparity_map);
	void setup_mst_img();
	void set_sub_disparity_map(uchar *_sub_disparity_map);
	uchar * upsampling();
	
};
void dmap_upsam::init(cl_context &_context, cl_device_id &_device, cl_program &_program, cl_int &_err,
					  int _down_sample_pow, cv::Mat img_b, match_info &_info, match_info &_sub_info, uchar *_sub_disparity_map)
{
	do_mst_mdf = true;
	do_dmap_mdf = true;

	down_sample_pow = _down_sample_pow;

	sub_disparity_map = _sub_disparity_map;

	info = _info;
	sub_info = _sub_info;
	int h = info.img_height;
	int w = info.img_width;
	int node_c = info.node_c;

	channel = upsampling_mst_channel;

	mst_b.init(h, w, channel, info.max_x_d, info.max_y_d);
	//mst_b.updateSigma( cwz_mst::sigma );

	context = _context;
	device  = _device;
	program = _program;
	err     = _err;

	img_color_1d_arr = c3_mat_to_1d_int_arr(img_b , info.img_height, info.img_width);

	if(channel == 1){
		//for channel = 1
		img_gray_1d_arr = new uchar[node_c];
		left_gray_1d_arr_for_mst = new uchar[node_c];
	}else{
		//for channel = 3
		left_color_1d_for_3_mst = new int[node_c];
		left_color_1d_arr_uchar = new uchar[node_c * 3];
	}
}
void dmap_upsam::set_sub_disparity_map(uchar *_sub_disparity_map){
	sub_disparity_map = _sub_disparity_map;
}
void dmap_upsam::setup_mst_img(){
	if(channel == 1){
		int_1d_to_gray_arr(img_color_1d_arr, img_gray_1d_arr, info.node_c);

		if( !(apply_cl_color_img_mdf<uchar>(context, device, program, err, img_gray_1d_arr, left_gray_1d_arr_for_mst, info, do_mst_mdf)) )
		{ printf("left_gray_1d_arr_for_mst median filtering failed.\n"); system("PAUSE"); }
		mst_b.set_img(left_gray_1d_arr_for_mst);
	}else{
		apply_cl_color_img_mdf<int>(context, device, program, err,  img_color_1d_arr, left_color_1d_for_3_mst, info, do_mst_mdf);
		int_1d_color_to_uchar_1d_color(left_color_1d_for_3_mst, left_color_1d_arr_uchar, info.node_c);
		mst_b.set_img(left_color_1d_arr_uchar);
	}
}
uchar *dmap_upsam::upsampling(){
	cwz_timer::start();
	mst_b.mst();
	cwz_timer::time_display("upsampling do MST");

	cwz_timer::start();
	int cost_len = info.img_width * info.img_height * info.max_x_d;
	float *agt_cost = mst_b.get_agt_result();
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
				int best_d = sub_disparity_map[ s_i ] * (info.max_x_d / sub_info.max_x_d);

				if(best_d > 2)
					for(int d=0 ; d < info.max_x_d ; d++){
						agt_cost[idx+d] = std::abs(d - best_d);
					}
		}
	}
	cwz_timer::time_display("upsampling calculate new cost volume");
	
	cwz_timer::start();
	mst_b.cost_agt();
	cwz_timer::time_display("upsampling cost_agt()");
	
	cwz_timer::start();
	upsampled_dmap = mst_b.pick_best_dispairty();
	cwz_timer::time_display("upsampling pick_best_disparity()");

	if( !(apply_cl_color_img_mdf<uchar>(context, device, program, err, upsampled_dmap, upsampled_dmap, info, do_dmap_mdf)) )
	{ printf("left_gray_1d_arr_for_mst median filtering failed.\n"); return 0; }

	return upsampled_dmap;
}


class dmap_refine{
private:
	int occlusion_threshold;

	cwz_mst mst;
	match_info info;

	int h;
	int w;
	int node_amt;

	bool *left_mask_1d;
	bool **left_mask_2d;
	uchar *left_dmap_1d;
	uchar *right_dmap_1d;
	uchar **left_dmap_2d;
	uchar **right_dmap_2d;

public:
	static int MODE_NO;
	static int MODE_TREE;
	static int MODE_SCANLINE_FILL;

	void init(cwz_mst &_mst, match_info &_info, uchar *_left_dmap_1d, uchar *_right_dmap_1d, int _occlusion_threshold = defaultOcclusionTh);
	void init(cwz_mst &_mst, match_info &_info, int _occlusion_threshold = 0);
	void set_left_right_dmap_value(uchar *left_dmap, uchar *right_dmap);

	void detect_occlusion();
	void calc_new_cost_after_left_right_check();
	uchar *refinement(int mode = doTreeRefinement);
};
int dmap_refine::MODE_NO = 0;
int dmap_refine::MODE_TREE = 1; 
int dmap_refine::MODE_SCANLINE_FILL = 2;

void dmap_refine::init(cwz_mst &_mst, match_info &_info, uchar *_left_dmap_1d, uchar *_right_dmap_1d, int _occlusion_threshold){
	occlusion_threshold = _occlusion_threshold;

	mst = _mst;
	info = _info;

	h = info.img_height;
	w = info.img_width;
	node_amt = info.node_c;

	left_mask_1d = new bool[node_amt];
	left_mask_2d = map_1d_arr_to_2d_arr<bool>(left_mask_1d , h, w);

	left_dmap_1d = _left_dmap_1d;
	right_dmap_1d = _right_dmap_1d;

	left_dmap_2d  = map_1d_arr_to_2d_arr<uchar>(left_dmap_1d , h, w);
	right_dmap_2d = map_1d_arr_to_2d_arr<uchar>(right_dmap_1d, h, w);
	
}
void dmap_refine::init(cwz_mst &_mst, match_info &_info, int _occlusion_threshold){
	left_dmap_1d = new uchar[node_amt];
	right_dmap_1d = new uchar[node_amt];
	init(_mst, _info, left_dmap_1d, right_dmap_1d);
}
void dmap_refine::set_left_right_dmap_value(uchar *left_dmap, uchar *right_dmap){
	for(int i=0 ; i<node_amt; i++)
		left_dmap_1d[i] = left_dmap[i];
	for(int i=0 ; i<node_amt; i++)
		right_dmap_1d[i] = right_dmap[i];
}
void dmap_refine::detect_occlusion(){
	memset(left_mask_1d, true, sizeof(bool) * node_amt);

	for(int y=0 ; y<h ; y++){
		for(int x=0 ; x<w ; x++){
			int d = left_dmap_2d[y][x];
			int rx = x-d;
			if( rx > 0 ){
				if( std::abs(left_dmap_2d[y][x] - right_dmap_2d[y][rx]) > occlusion_threshold ){
					left_mask_2d[y][x] = false;
				}
			}else{
				left_mask_2d[y][x] = false;
			}
		}
	}
}
void dmap_refine::calc_new_cost_after_left_right_check(){
	int total_len = info.node_c * info.max_x_d;
	float *agt = mst.get_agt_result();
	memset(agt, 0, sizeof(float) * total_len);

	for(int i=0 ; i<total_len ; i+=info.max_x_d) if(left_mask_1d[i/info.max_x_d]){
		for(int d=0; d<info.max_x_d ; d++){
			agt[i+d] = std::abs(d - left_dmap_1d[i/info.max_x_d]);
		}
	}
}
uchar *dmap_refine::refinement(int mode){
	int w = info.img_width;
	int h = info.img_height;

	detect_occlusion();
	if(mode == dmap_refine::MODE_TREE){
		calc_new_cost_after_left_right_check();
		mst.cost_agt();
		return mst.pick_best_dispairty();
	}else if(mode == dmap_refine::MODE_SCANLINE_FILL){
	
	}else{
		uchar *refined_dmap = new uchar[w*h];
		for(int i=0 ; i<w*h ; i++){
			if(left_mask_1d[i] == false)
				refined_dmap[i] = 0;
			else
				refined_dmap[i] = left_dmap_1d[i];
		}
		return refined_dmap;
	}
}

uchar *cwz_dmap_generate(cl_context &context, cl_device_id &device, cl_program &program, cl_int &err,
						 cv::Mat left,  cv::Mat right, cwz_mst &mst, match_info &info, bool inverse = false)
{
	time_t img_init_s = clock();

	int w = info.img_width;
	int h = info.img_height;
	int node_c = info.node_c;
	int channel = mst_channel;
	
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
	
	if( channel == 1 ){
		uchar *left_gray_1d_arr_for_mst;
		if( !(left_gray_1d_arr_for_mst = apply_cl_color_img_mdf<uchar>(context, device, program, err, left_gray_1d_arr, info, mst_pre_mdf)) )
		{ printf("left_gray_1d_arr_for_mst median filtering failed.\n"); return 0; }
		mst.set_img(left_gray_1d_arr_for_mst);
	}else{
		int *left_color_1d_for_3_mst = apply_cl_color_img_mdf<int>(context, device, program, err,  left_color_1d_arr, info, mst_pre_mdf);
		uchar *left_color_1d_arr_uchar = int_1d_color_to_uchar_1d_color(left_color_1d_for_3_mst, node_c);
		mst.set_img(left_color_1d_arr_uchar);
	}
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
	int channel = upsampling_mst_channel;
	//mstL_b.updateSigma( cwz_mst::sigma );
	//建原本size大小的tree
	int *left_color_1d_arr  = c3_mat_to_1d_int_arr(left_b , info.img_height, info.img_width);

	if(channel == 1){
		uchar *left_gray_1d_arr  = int_1d_arr_to_gray_arr(left_color_1d_arr , info.node_c);
		uchar *left_gray_1d_arr_for_mst;
		if( !(left_gray_1d_arr_for_mst = apply_cl_color_img_mdf<uchar>(context, device, program, err, left_gray_1d_arr, info, do_mst_mdf)) )
		{ printf("left_gray_1d_arr_for_mst median filtering failed.\n"); return 0; }
		mstL_b.set_img(left_gray_1d_arr_for_mst);
	}else{
		int *left_color_1d_for_3_mst = apply_cl_color_img_mdf<int>(context, device, program, err,  left_color_1d_arr, info, do_mst_mdf);
		uchar *left_color_1d_arr_uchar = int_1d_color_to_uchar_1d_color(left_color_1d_for_3_mst, info.node_c);
		mstL_b.set_img(left_color_1d_arr_uchar);
	}

	cwz_timer::start();
	mstL_b.mst();
	cwz_timer::time_display("original size left mst");
	//用subsampled depth map算cost
	cwz_timer::start();
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
	cwz_timer::time_display("upsampling calculate new cost volume");
	//cost aggregate
	uchar *upsampled_dmap;
	mstL_b.cost_agt();
	cwz_timer::start();
	upsampled_dmap = mstL_b.pick_best_dispairty();
	cwz_timer::time_display("upsampling pick_best_dispairty");
	if( !(upsampled_dmap = apply_cl_color_img_mdf<uchar>(context, device, program, err, upsampled_dmap, info, do_dmap_mdf)) )
	{ printf("left_gray_1d_arr_for_mst median filtering failed.\n"); return 0; }
	return upsampled_dmap;
}

#endif