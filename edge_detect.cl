#define CWZ_CL_DEVICE_SIDE
#include "cwz_edge_cl_config.h"
#define _1DThreshold_L1 10
#define _1DThreshold_L2 20
#define _1DThreshold_L3 40
/*
const float _1DK1[2] = {-1, 1};
const float _1DK2[4] = {-1.41421, -1.41421, 1.41421, 1.41421};
const float _1DK3[8] = {-2, -2, -2, -2, 2, 2, 2, 2};
*/

bool isEdge(float cost, __local uchar *block){
	if(cost > _1DThreshold)
		return true;
	else
		return false;
}

inline void isEdge_3L(__global EDGE_DETECT_RESULT_TYPE *result, 
					  __local uchar *block, 
					  __global const EdgeKernelInfo_1D *info, 
					  unsigned int block_idx,
					  unsigned int idx)
{
	//第一層
	float cost = fabs(block[block_idx - 1] * info->k1[0] + 
					  block[block_idx    ] * info->k1[1] );
	if(cost > _1DThreshold_L1){
		//result[idx] = round(cost);
		result[idx] = 255;
	}else{
		//第二層
		cost = fabs(block[block_idx - 2] * info->k2[0] + 
					block[block_idx - 1] * info->k2[1] + 
					block[block_idx    ] * info->k2[2] +
					block[block_idx + 1] * info->k2[3] );
		if(cost > _1DThreshold_L2){
			//result[idx] = round(cost);
			result[idx] = 255;
		}else{
			//第三層
			cost = fabs(block[block_idx - 4] * info->k3[0] + 
						block[block_idx - 3] * info->k3[1] + 
						block[block_idx - 2] * info->k3[2] +
						block[block_idx - 1] * info->k3[3] +
						block[block_idx    ] * info->k3[4] +
						block[block_idx + 1] * info->k3[5] + 
						block[block_idx + 2] * info->k3[6] +
						block[block_idx + 3] * info->k3[7] );
			if(cost > _1DThreshold_L3){
				//result[idx] = round(cost);
				result[idx] = 255;
			}
		}
	}
}

__kernel void edgeDetect_1d_exp(__global EDGE_DETECT_RESULT_TYPE *result, 
							__global uchar *img, 
							__global const EdgeKernelInfo_1D *info,
							 //__constant EdgeKernelInfo_1D *info, ---->會讀到錯誤的值, 原因找很久仍不明
							 __local uchar *block)
{
	// read the matrix tile into shared memory
	const unsigned int idx  = get_global_id(0);
	const unsigned int xidx = idx % info->width;
	//const unsigned int yidx = idx / info->width;
	unsigned int local_idx = get_local_id(0);
	float cost;
	if( xidx < info->half_width )
	{//group 1
		
		block[local_idx] = img[idx];

		if(local_idx == (info->half_width - 1))
		{//gropu 1的最後一個負責加右邊3格的值
			for(int i=1 ; i <= _1D_K3_Right_len ; i++){
				block[local_idx+i] = img[idx+i];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		//第一組的前幾個不用做
		if(local_idx >= _1D_K3_Left_len){
			isEdge_3L(result, block, info, local_idx, idx);			
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

		barrier(CLK_LOCAL_MEM_FENCE);

		//第二組的最後面幾個不用做
		if(local_idx <= info->half_width-1-_1D_K3_Right_len){
			isEdge_3L(result, block, info, shift_local_idx, idx);	
		}
	}
}

__kernel void edgeDetect_1d(__global EDGE_DETECT_RESULT_TYPE *result, 
							__global uchar *img, 
							__global const EdgeKernelInfo_1D *info,
						  //__constant EdgeKernelInfo_1D *info, ---->會讀到錯誤的值, 原因找很久仍不明
							__local uchar *block){
	// read the matrix tile into shared memory
	const unsigned int idx  = get_global_id(0);
	const unsigned int xidx = idx % info->width;
	//const unsigned int yidx = idx / info->width;
	unsigned int local_idx = get_local_id(0);
	float cost;
	if( xidx < info->half_width )
	{//group 1
		
		block[local_idx] = img[idx];

		if(local_idx == (info->half_width - 1))
		{//gropu 1的最後一個負責加右邊3格的值
			for(int i=1 ; i <= _1D_K3_Right_len ; i++){
				block[local_idx+i] = img[idx+i];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		//第一層
		if(local_idx >= _1D_K1_Left_len){

			cost = fabs(block[local_idx - 1] * info->k1[0] + block[local_idx] * info->k1[1]); 
			if( isEdge(cost, block) ){
				//result[idx] = round(cost);
				result[idx] = 255;
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

		barrier(CLK_LOCAL_MEM_FENCE);

		//第一層
		cost = fabs(block[shift_local_idx - 1] * info->k1[0] + block[shift_local_idx] * info->k1[1]); 
		if( isEdge(cost, block) ){
			//result[idx] = round(cost);
			result[idx] = 255;
		}
	}
}