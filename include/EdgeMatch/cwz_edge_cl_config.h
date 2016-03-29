#ifndef CWZ_EDGE_CL_CONFIG_H
#define CWZ_EDGE_CL_CONFIG_H

/**
	如果修改此檔案
	則也要改變edge_detect.cl
	不然就要清除之前compile完的檔案
	否則程式會使用edge_detect.cl之前compile的結果
	不知道有修改 所以不會重新compile
**/

#define EDGE_DETECT_RESULT_TYPE uchar

/*
//執行此kernel的prerequisite
	if(this->w % 2 != 0) printf("cwz_edge_detector::Error image width should be multiple of 2.\n");
	if(this->dim_max_items[0] * 2 < this->w) printf("cwz_edge_detector::Error image width is too large.\n");
edgeDetect_1d(...)
	work_dim = 1d

	影像寬度假設560
	此kernel的work group size為 560/2
	一個scanline會被切成 2 個work group處理
	每個work group會有280個work item (item_num = 280)
	第一個work group需要load進local memory的global idx有 [0] ~ [item_num + kernel_right_len - 1]
	第二個work group需要load進local memory的global idx有 [item_num - kernel_left_len] ~ [image_width - 1]

	gradient kernel
	{-1, 1}							kernel_left_len = 1 | kernel_right_len = 0
	{-2^0.5, -2^0.5, 2^0.5, 2^0.5}	kernel_left_len = 2 | kernel_right_len = 1
	{-2, -2, -2, -2, 2, 2, 2, 2}	kernel_left_len = 4 | kernel_right_len = 3
*/
#define _1D_K1_Left_len  1
#define _1D_K2_Left_len  2
#define _1D_K3_Left_len  4
#define _1D_K1_Right_len 0
#define _1D_K2_Right_len 1
#define _1D_K3_Right_len 3
#define _1DThreshold 3

//device side跟host side的struct定義要同步
typedef struct{
	#ifdef CWZ_CL_HOST_SIDE
		cl_int width;
		cl_int half_width;
		cl_int height;
		cl_int xoffset;
		cl_float k1[2];
		cl_float k2[4];
		cl_float k3[8];
	#else CWZ_CL_DEVICE_SIDE
		int width;
		int half_width;
		int height;
		int xoffset;
		float k1[2];
		float k2[4];
		float k3[8];
	#endif
} EdgeKernelInfo_1D;

#endif