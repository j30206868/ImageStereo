//Result type也被定義在cwz_edge_detector要同時改
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

bool isEdge(float cost, __local uchar *block){
	if(cost > _1DThreshold)
		return true;
	else
		return false;
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