//For inputkey loop control usage
namespace cwz_loop_ctrl{
	//控制matching cost threshold
	static float Match_Cost_Th = 5;
	static const float Match_Cost_Step = 0.5;
	static float Match_Cost_Least_W = 0.01;
	static float Match_Cost_Least_W_Step = 0.01;

	//控制目前使用的方法
	static int M_Key_counter = 0;
	static int       Mode = 0;
	static const int METHOD_TREE = 0;
	static const int METHOD_TREE_NO_REFINE = 1;
	static const int METHOD_FILL_SCANLINE = 2;
	static const int MEDTHO_CV_SGNM = 3;
	static const int M_Key_total = 4;

	//控制濾波
	static bool Do_Guided_Filer = true;

	//CV image inputkey state
	static const int CV_IMG_STATUS_KEEPGOING = 0;
	static const int CV_IMG_STATUS_FRAME_BY_FRAME = 1;
	static const int CV_IMG_STATUS_MODIFY_PARAM = 2;
	static const int CV_IMG_STATUS_EXIT = 999;
};

//all used by main function
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <opencv2\opencv.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include "PxlMatch/cwz_cl_data_type.h"
#include "PxlMatch/cwz_cl_cpp_functions.h"
#include "TreeFilter/cwz_mst.h"
#include "TreeFilter/cwz_disparity_generation.h"
#include "GuidedFilter/cwz_integral_img.h"
#include "EdgeMatch/cwz_edge_detect.h"