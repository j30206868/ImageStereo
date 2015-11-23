//__constant float color_ratio       = 0.11;
__constant float color_ratio       = 0.11;
__constant float gradient_ratio    = 0.89;
__constant float max_color_cost    = 7.0;
__constant float max_gradient_cost = 2.0;

__constant float lips_10b_max_color_cost    = 84.0;
__constant float lips_10b_max_gradient_cost = 8.0;
//__constant float lips_10b_max_color_cost    = 7;
//__constant float lips_10b_max_gradient_cost = 2.0;

__constant int mask_b = 0xFF;
__constant int mask_g = 0xFF00;
__constant int mask_r = 0xFF0000;

typedef struct {
	int max_d;
	int img_width;
	int img_height;
	int node_c;
} match_info;

__kernel void matching_cost(__global const int* l_rgb, __global const float *l_gradient, 
							__global const int* r_rgb, __global const float *r_gradient, 
							__global float* result, __global match_info *info)
{
	// 450 -> width of image
	// 60  -> max_disparity

	const int idx = get_global_id(0);
	const int x = idx % info->img_width;

	int ridx = idx - x;
	for(int d = info->max_d-1 ; d >= 0  ; d--){
		if(x > d)
			ridx = idx-d;

		float color_cost = abs(  ((l_rgb[idx]&mask_b) - (r_rgb[ridx]&mask_b))      ) +
					       abs( (((l_rgb[idx]&mask_g) - (r_rgb[ridx]&mask_g)) >> 8)) +
					       abs( (((l_rgb[idx]&mask_r) - (r_rgb[ridx]&mask_r)) >> 16));

		color_cost = fmin(color_cost/3.0, max_color_cost);

		float gradient_cost = fmin( fabs(l_gradient[idx] - r_gradient[ridx]), max_gradient_cost);

		//result[d*info->node_c + idx] = color_cost*color_ratio + gradient_cost*gradient_ratio;
		result[(idx * info->max_d) + d] = color_cost*color_ratio + gradient_cost*gradient_ratio;
	}
}

__kernel void lips_10b_matching_cost(__global const short* l_rgb, __global const float *l_gradient, 
									 __global const short* r_rgb, __global const float *r_gradient, 
									 __global float* result, __global match_info *info)
{
	// 450 -> width of image
	// 60  -> max_disparity

	const int idx = get_global_id(0);
	const int x = idx % info->img_width;

	int ridx = idx - x;
	for(int d = info->max_d-1 ; d >= 0  ; d--){
		if(x > d)
			ridx = idx-d;

		float color_cost = abs(l_rgb[idx] - r_rgb[ridx]);

		color_cost = fmin(color_cost, lips_10b_max_color_cost);

		float gradient_cost = fmin( fabs(l_gradient[idx] - r_gradient[ridx]), lips_10b_max_gradient_cost);

		//result[d*info->node_c + idx] = color_cost*color_ratio + gradient_cost*gradient_ratio;
		result[(idx * info->max_d) + d] = color_cost*color_ratio + gradient_cost*gradient_ratio;
	}
}

__kernel void lips_10b_matching_cost_inverse(__global const short* l_rgb, __global const float *l_gradient, 
											 __global const short* r_rgb, __global const float *r_gradient, 
											 __global float* result, __global match_info *info)
{
	// 450 -> width of image
	// 60  -> max_disparity

	const int idx = get_global_id(0);
	const int x = idx % info->img_width;

	int ridx = idx - x + info->img_width;
	for(int d = info->max_d - 1 ; d >= 0 ; d--){
		if( (x + d) < info->img_width )
			ridx = idx + d;

		float color_cost = abs(l_rgb[idx] - r_rgb[ridx]);

		color_cost = fmin(color_cost, lips_10b_max_color_cost);

		float gradient_cost = fmin( fabs(l_gradient[idx] - r_gradient[ridx]), lips_10b_max_gradient_cost);

		//result[d*info->node_c + idx] = color_cost*color_ratio + gradient_cost*gradient_ratio;
		result[(idx * info->max_d) + d] = color_cost*color_ratio + gradient_cost*gradient_ratio;
	}
}

__kernel void MedianFilterBitonic(const __global uint* pSrc, __global uint* pDst, __global match_info *info)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int iOffset = y * info->img_width;

	//若邊緣沒有擋住 會造成不可預期的錯誤 很難debug
	if( (x > 0 && y > 0) && (x < info->img_width-1 && y < info->img_height-1) ){
		const int iPrev = iOffset - info->img_width;
		const int iNext = iOffset + info->img_width;

		uint uiRGBA[9];
			
		//get pixels within aperture
		uiRGBA[0] = pSrc[iPrev + x - 1];
		uiRGBA[1] = pSrc[iPrev + x];
		uiRGBA[2] = pSrc[iPrev + x + 1];

		uiRGBA[3] = pSrc[iOffset + x - 1];
		uiRGBA[4] = pSrc[iOffset + x];
		uiRGBA[5] = pSrc[iOffset + x + 1];

		uiRGBA[6] = pSrc[iNext + x - 1];
		uiRGBA[7] = pSrc[iNext + x];
		uiRGBA[8] = pSrc[iNext + x + 1];

		uint uiResult = 0;
		uint uiMask = 0xFF;

		for(int ch = 0; ch < 3; ch++)
		{

			//extract next color channel
			uint r0,r1,r2,r3,r4,r5,r6,r7,r8;
			r0=uiRGBA[0]& uiMask;
			r1=uiRGBA[1]& uiMask;
			r2=uiRGBA[2]& uiMask;
			r3=uiRGBA[3]& uiMask;
			r4=uiRGBA[4]& uiMask;
			r5=uiRGBA[5]& uiMask;
			r6=uiRGBA[6]& uiMask;
			r7=uiRGBA[7]& uiMask;
			r8=uiRGBA[8]& uiMask;
		
			//perform partial bitonic sort to find current channel median
			uint uiMin = min(r0, r1);
			uint uiMax = max(r0, r1);
			r0 = uiMin;
			r1 = uiMax;

			uiMin = min(r3, r2);
			uiMax = max(r3, r2);
			r3 = uiMin;
			r2 = uiMax;

			uiMin = min(r2, r0);
			uiMax = max(r2, r0);
			r2 = uiMin;
			r0 = uiMax;

			uiMin = min(r3, r1);
			uiMax = max(r3, r1);
			r3 = uiMin;
			r1 = uiMax;

			uiMin = min(r1, r0);
			uiMax = max(r1, r0);
			r1 = uiMin;
			r0 = uiMax;

			uiMin = min(r3, r2);
			uiMax = max(r3, r2);
			r3 = uiMin;
			r2 = uiMax;

			uiMin = min(r5, r4);
			uiMax = max(r5, r4);
			r5 = uiMin;
			r4 = uiMax;

			uiMin = min(r7, r8);
			uiMax = max(r7, r8);
			r7 = uiMin;
			r8 = uiMax;

			uiMin = min(r6, r8);
			uiMax = max(r6, r8);
			r6 = uiMin;
			r8 = uiMax;

			uiMin = min(r6, r7);
			uiMax = max(r6, r7);
			r6 = uiMin;
			r7 = uiMax;

			uiMin = min(r4, r8);
			uiMax = max(r4, r8);
			r4 = uiMin;
			r8 = uiMax;

			uiMin = min(r4, r6);
			uiMax = max(r4, r6);
			r4 = uiMin;
			r6 = uiMax;

			uiMin = min(r5, r7);
			uiMax = max(r5, r7);
			r5 = uiMin;
			r7 = uiMax;

			uiMin = min(r4, r5);
			uiMax = max(r4, r5);
			r4 = uiMin;
			r5 = uiMax;

			uiMin = min(r6, r7);
			uiMax = max(r6, r7);
			r6 = uiMin;
			r7 = uiMax;

			uiMin = min(r0, r8);
			uiMax = max(r0, r8);
			r0 = uiMin;
			r8 = uiMax;

			r4 = max(r0, r4);
			r5 = max(r1, r5);

			r6 = max(r2, r6);
			r7 = max(r3, r7);

			r4 = min(r4, r6);
			r5 = min(r5, r7);

			//store found median into result
			uiResult |= min(r4, r5);

			//update channel mask
			uiMask <<= 8;
		}

		//store result into memory
		pDst[iOffset + x] = uiResult;
	}else{
		pDst[iOffset + x] = pSrc[iOffset + x];
	}
}

__kernel void MedianFilterGrayScale(const __global uchar* pSrc, __global uchar* pDst, __global match_info *info)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	const int iOffset = y * info->img_width;
	
	//若邊緣沒有擋住 會造成不可預期的錯誤 很難debug
	if((x > 0 && y > 0) && (x < info->img_width-1 && y < info->img_height-1)){
		const int iPrev = iOffset - info->img_width;
		const int iNext = iOffset + info->img_width;
		//extract next color channel
		uchar r0,r1,r2,r3,r4,r5,r6,r7,r8;
		r0 = pSrc[iPrev + x - 1];
		r1 = pSrc[iPrev + x];
		r2 = pSrc[iPrev + x + 1];

		r3 = pSrc[iOffset + x - 1];
		r4 = pSrc[iOffset + x];
		r5 = pSrc[iOffset + x + 1];

		r6 = pSrc[iNext + x - 1];
		r7 = pSrc[iNext + x];
		r8 = pSrc[iNext + x + 1];


		//perform partial bitonic sort to find current channel median
		uchar uiMin = min(r0, r1);
		uchar uiMax = max(r0, r1);
		r0 = uiMin;
		r1 = uiMax;

		uiMin = min(r3, r2);
		uiMax = max(r3, r2);
		r3 = uiMin;
		r2 = uiMax;

		uiMin = min(r2, r0);
		uiMax = max(r2, r0);
		r2 = uiMin;
		r0 = uiMax;

		uiMin = min(r3, r1);
		uiMax = max(r3, r1);
		r3 = uiMin;
		r1 = uiMax;

		uiMin = min(r1, r0);
		uiMax = max(r1, r0);
		r1 = uiMin;
		r0 = uiMax;

		uiMin = min(r3, r2);
		uiMax = max(r3, r2);
		r3 = uiMin;
		r2 = uiMax;

		uiMin = min(r5, r4);
		uiMax = max(r5, r4);
		r5 = uiMin;
		r4 = uiMax;

		uiMin = min(r7, r8);
		uiMax = max(r7, r8);
		r7 = uiMin;
		r8 = uiMax;

		uiMin = min(r6, r8);
		uiMax = max(r6, r8);
		r6 = uiMin;
		r8 = uiMax;

		uiMin = min(r6, r7);
		uiMax = max(r6, r7);
		r6 = uiMin;
		r7 = uiMax;

		uiMin = min(r4, r8);
		uiMax = max(r4, r8);
		r4 = uiMin;
		r8 = uiMax;

		uiMin = min(r4, r6);
		uiMax = max(r4, r6);
		r4 = uiMin;
		r6 = uiMax;

		uiMin = min(r5, r7);
		uiMax = max(r5, r7);
		r5 = uiMin;
		r7 = uiMax;

		uiMin = min(r4, r5);
		uiMax = max(r4, r5);
		r4 = uiMin;
		r5 = uiMax;

		uiMin = min(r6, r7);
		uiMax = max(r6, r7);
		r6 = uiMin;
		r7 = uiMax;

		uiMin = min(r0, r8);
		uiMax = max(r0, r8);
		r0 = uiMin;
		r8 = uiMax;

		r4 = max(r0, r4);
		r5 = max(r1, r5);

		r6 = max(r2, r6);
		r7 = max(r3, r7);

		r4 = min(r4, r6);
		r5 = min(r5, r7);

		//store result into memory
		pDst[iOffset + x] = min(r4, r5);
	}else{
		pDst[iOffset + x] = pSrc[iOffset + x];
	}
}

__kernel void Lips_10b_MedianFilterGrayScale(const __global ushort* pSrc, __global ushort* pDst, __global match_info *info)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	const int iOffset = y * info->img_width;
	
	//若邊緣沒有擋住 會造成不可預期的錯誤 很難debug
	if((x > 0 && y > 0) && (x < info->img_width-1 && y < info->img_height-1)){
		const int iPrev = iOffset - info->img_width;
		const int iNext = iOffset + info->img_width;
		//extract next color channel
		ushort r0,r1,r2,r3,r4,r5,r6,r7,r8;
		r0 = pSrc[iPrev + x - 1];
		r1 = pSrc[iPrev + x];
		r2 = pSrc[iPrev + x + 1];

		r3 = pSrc[iOffset + x - 1];
		r4 = pSrc[iOffset + x];
		r5 = pSrc[iOffset + x + 1];

		r6 = pSrc[iNext + x - 1];
		r7 = pSrc[iNext + x];
		r8 = pSrc[iNext + x + 1];


		//perform partial bitonic sort to find current channel median
		ushort uiMin = min(r0, r1);
		ushort uiMax = max(r0, r1);
		r0 = uiMin;
		r1 = uiMax;

		uiMin = min(r3, r2);
		uiMax = max(r3, r2);
		r3 = uiMin;
		r2 = uiMax;

		uiMin = min(r2, r0);
		uiMax = max(r2, r0);
		r2 = uiMin;
		r0 = uiMax;

		uiMin = min(r3, r1);
		uiMax = max(r3, r1);
		r3 = uiMin;
		r1 = uiMax;

		uiMin = min(r1, r0);
		uiMax = max(r1, r0);
		r1 = uiMin;
		r0 = uiMax;

		uiMin = min(r3, r2);
		uiMax = max(r3, r2);
		r3 = uiMin;
		r2 = uiMax;

		uiMin = min(r5, r4);
		uiMax = max(r5, r4);
		r5 = uiMin;
		r4 = uiMax;

		uiMin = min(r7, r8);
		uiMax = max(r7, r8);
		r7 = uiMin;
		r8 = uiMax;

		uiMin = min(r6, r8);
		uiMax = max(r6, r8);
		r6 = uiMin;
		r8 = uiMax;

		uiMin = min(r6, r7);
		uiMax = max(r6, r7);
		r6 = uiMin;
		r7 = uiMax;

		uiMin = min(r4, r8);
		uiMax = max(r4, r8);
		r4 = uiMin;
		r8 = uiMax;

		uiMin = min(r4, r6);
		uiMax = max(r4, r6);
		r4 = uiMin;
		r6 = uiMax;

		uiMin = min(r5, r7);
		uiMax = max(r5, r7);
		r5 = uiMin;
		r7 = uiMax;

		uiMin = min(r4, r5);
		uiMax = max(r4, r5);
		r4 = uiMin;
		r5 = uiMax;

		uiMin = min(r6, r7);
		uiMax = max(r6, r7);
		r6 = uiMin;
		r7 = uiMax;

		uiMin = min(r0, r8);
		uiMax = max(r0, r8);
		r0 = uiMin;
		r8 = uiMax;

		r4 = max(r0, r4);
		r5 = max(r1, r5);

		r6 = max(r2, r6);
		r7 = max(r3, r7);

		r4 = min(r4, r6);
		r5 = min(r5, r7);

		//store result into memory
		pDst[iOffset + x] = min(r4, r5);
	}else{
		pDst[iOffset + x] = pSrc[iOffset + x];
	}
}