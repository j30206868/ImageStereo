#ifndef CWZ_EDGE_CL_CONFIG_H
#define CWZ_EDGE_CL_CONFIG_H

/**
	�p�G�ק惡�ɮ�
	�h�]�n����edge_detect.cl
	���M�N�n�M�����ecompile�����ɮ�
	�_�h�{���|�ϥ�edge_detect.cl���ecompile�����G
	�����D���ק� �ҥH���|���scompile
**/

#define EDGE_DETECT_RESULT_TYPE uchar

/*
//���榹kernel��prerequisite
	if(this->w % 2 != 0) printf("cwz_edge_detector::Error image width should be multiple of 2.\n");
	if(this->dim_max_items[0] * 2 < this->w) printf("cwz_edge_detector::Error image width is too large.\n");
edgeDetect_1d(...)
	work_dim = 1d

	�v���e�װ��]560
	��kernel��work group size�� 560/2
	�@��scanline�|�Q���� 2 ��work group�B�z
	�C��work group�|��280��work item (item_num = 280)
	�Ĥ@��work group�ݭnload�ilocal memory��global idx�� [0] ~ [item_num + kernel_right_len - 1]
	�ĤG��work group�ݭnload�ilocal memory��global idx�� [item_num - kernel_left_len] ~ [image_width - 1]

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

//device side��host side��struct�w�q�n�P�B
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