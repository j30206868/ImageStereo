#ifndef CWZ_DISPARITY_GENERATION
#define CWZ_DISPARITY_GENERATION

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "cwz_mst.h"
#include "PxlMatch/cwz_cl_data_type.h"
#include "PxlMatch/cwz_cl_cpp_functions.h"
#include "GuidedFilter/cwz_integral_img.h"

#define DMAP_GEN_GIF_TYPE float
class dmap_gen{
private:
	int channel;

	cl_context context;
	cl_device_id device;
	cl_program program; 
	cl_int err;

	float *left_1d_gradient;
	float *right_1d_gradient;

	int *left_color_1d_arr;
	int *right_color_1d_arr;

	uchar **left_gray_2d_arr;
	uchar **right_gray_2d_arr;

	//for grayscale guided image filtering
	guided_img<DMAP_GEN_GIF_TYPE, DMAP_GEN_GIF_TYPE> *gfilter;
	DMAP_GEN_GIF_TYPE *normalized_left_gray_img;
	DMAP_GEN_GIF_TYPE *normalized_right_gray_img;
	//

	int * left_color_mdf_1d_arr;
	int *right_color_mdf_1d_arr;

	//for mst
	uchar *left_img_arr_for_mst;
	uchar *right_img_arr_for_mst;
	//for channel = 3 mst
	int *left_color_1d_for_3_mst;
	int *right_color_1d_for_3_mst;
public:
	match_info *info;

	uchar *left_gray_1d_arr;
	uchar *right_gray_1d_arr;

	bool doGuildFiltering;
	CWZDISPTYPE *left_dmap;
	CWZDISPTYPE *right_dmap;

	cl_match_elem *left_cwz_img;
	cl_match_elem *right_cwz_img;

	cwz_mst mst_L;
	cwz_mst mst_R;

	void init(cl_context &context, cl_device_id &device, cl_program &program, cl_int &err, 
		      cv::Mat left, cv::Mat right, match_info &_info);
	void set_left_right(cv::Mat left, cv::Mat right);
	void filtering();
	void compute_cwz_img();
	CWZDISPTYPE *generate_left_dmap();
	CWZDISPTYPE *generate_right_dmap();
};


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
	CWZDISPTYPE *sub_disparity_map;

	bool do_mst_mdf;
	bool do_dmap_mdf;

	int *img_color_1d_arr;

	//for channel = 1
	uchar *img_gray_1d_arr;
	uchar *left_gray_1d_arr_for_mst;
	//for channel = 3
	int *left_color_1d_for_3_mst;
	uchar *left_color_1d_arr_uchar;

	CWZDISPTYPE *upsampled_dmap;

public:
	cwz_mst mst_b;

	void init(cl_context &_context, cl_device_id &_device, cl_program &_program, cl_int &_err,
		       int _down_sample_pow, cv::Mat img_b, match_info &_info, match_info &_sub_info, CWZDISPTYPE *_sub_disparity_map);
	void setup_mst_img();
	void set_sub_disparity_map(CWZDISPTYPE *_sub_disparity_map);
	CWZDISPTYPE * upsampling();
	
};


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
	CWZDISPTYPE *left_dmap_1d;
	CWZDISPTYPE *right_dmap_1d;
	CWZDISPTYPE **left_dmap_2d;
	CWZDISPTYPE **right_dmap_2d;
	
	uchar *left_edge; 

public:
	static int MODE_NO;
	static int MODE_TREE;
	static int MODE_SCANLINE_FILL;

	void init(cwz_mst &_mst, match_info &_info, CWZDISPTYPE *_left_dmap_1d, CWZDISPTYPE *_right_dmap_1d, int _occlusion_threshold = defaultOcclusionTh);
	void init(cwz_mst &_mst, match_info &_info, int _occlusion_threshold = 0);
	void set_left_edge_map(uchar *_left_edge);
	void set_left_right_dmap_value(CWZDISPTYPE *left_dmap, CWZDISPTYPE *right_dmap);

	void detect_occlusion();
	void calc_new_cost_after_left_right_check();
	CWZDISPTYPE *refinement(int mode = doTreeRefinement);
};

#endif