//Result type也被定義在cwz_edge_detector要同時改
#define EDGE_DETECT_RESULT_TYPE float

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
/*
const float _1DK1[2] = {-1, 1};
const float _1DK2[4] = {-1.41421, -1.41421, 1.41421, 1.41421};
const float _1DK3[8] = {-2, -2, -2, -2, 2, 2, 2, 2};
*/
//struct EdgeKernelInfo_1D也被定義在cwz_edge_detector要同時改
typedef struct EdgeKernelInfo_1D{
	int width;
	int half_width;
	int height;
	int xoffset;
	float k1[2];
	float k2[4];
	float k3[8];
} EdgeKernelInfo_1D;
// end of 1D kernel definition

__kernel void edgeDetect_1d(__global EDGE_DETECT_RESULT_TYPE *result, 
							__global uchar *img, 
							__constant struct EdgeKernelInfo_1D *info,
							__local uchar *block){
	// read the matrix tile into shared memory
	const unsigned int idx  = get_global_id(0);
	const unsigned int xidx = idx % info->width;
	const unsigned int yidx = idx / info->width;
	unsigned int local_idx = get_local_id(0);

	if( xidx < info->half_width )
	{//group 1
		
		block[local_idx] = img[idx];

		if(local_idx == (info->half_width - 1))
		{//gropu 1的最後一個負責加右邊3格的值
			for(int i=1 ; i <= _1D_K3_Right_len ; i++){
				block[local_idx+i] = img[idx+i];
			}
		}
	}else
	{//group 2
		unsigned int shift_local_idx = local_idx + _1D_K3_Left_len;
		if(local_idx == 0)
		{//group 2的第一個負責加左邊4格的值
			for(int i=0 ; i<_1D_K3_Left_len ; i++){
				block[i] = img[idx-_1D_K3_Left_len+i];
			}
		}

		block[shift_local_idx] = img[idx];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int group_id = (xidx - local_idx) / info->half_width;

	/*if( group_id == 0 )
	{
		result[idx] = 0;
	}else if( group_id == 1 ) {
		result[idx] = 128;
	}else{
		result[idx] = 255;
	}*/
	result[0] = info->width;
	result[1] = info->half_width;
	result[2] = info->height;
	result[3] = info->xoffset;
	result[4] = info->k1[0];
	result[5] = info->k1[1];
	result[6] = info->k2[0];
	result[7] = info->k2[1];
	result[8] = info->k2[2];
	result[9] = info->k2[3];

	result[10] = info->k3[0];
	result[11] = info->k3[1];
	result[12] = info->k3[2];
	result[13] = info->k3[3];
	result[14] = info->k3[4];
	result[15] = info->k3[5];
	result[16] = info->k3[6];
	result[17] = info->k3[7];
}