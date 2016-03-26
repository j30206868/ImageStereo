#ifndef CWZ_INTEGRAL_IMG
#define CWZ_INTEGRAL_IMG

#include <iostream>
#include <math.h>

#include "common_func.h"

void guild_filter_example(const char *LeftIMGName, int max_intensity);

inline int get_1d_idx(int x, int y, int w){return y * w + x;}
template<class T> inline T getArea(T *acu_img, int x1, int y1, int x2, int y2, int w, int h){
	if(x1 > 0)
		if(y1 > 0)
			return acu_img[get_1d_idx(x2, y2, w)] - acu_img[get_1d_idx(x1-1, y2, w)] - acu_img[get_1d_idx(x2, y1-1, w)] + acu_img[get_1d_idx(x1-1, y1-1, w)];
		else
			return acu_img[get_1d_idx(x2, y2, w)] - acu_img[get_1d_idx(x1-1, y2, w)];
	else if(y1 > 0)
		return acu_img[get_1d_idx(x2, y2, w)] - acu_img[get_1d_idx(x2, y1-1, w)];
	else
		return acu_img[get_1d_idx(x2, y2, w)];
}

template<class A, class B, class O> inline void array_add (A *array_a, B *array_b, O *array_out, int len){for(int i=0; i<len ; i++){ array_out[i] = array_a[i] + array_b[i]; }}
template<class A, class B, class O> inline void array_sub (A *array_a, B *array_b, O *array_out, int len){for(int i=0; i<len ; i++){ array_out[i] = array_a[i] - array_b[i]; }}
template<class A, class B, class O> inline void array_mult(A *array_a, B *array_b, O *array_out, int len){for(int i=0; i<len ; i++){ array_out[i] = array_a[i] * array_b[i]; }}
template<class A, class B, class O> inline void array_div (A *array_a, B *array_b, O *array_out, int len){for(int i=0; i<len ; i++){ array_out[i] = array_a[i] / (O)array_b[i]; }}

template<class T, class O> void buildIntegralImg(T *img, O *acu_img, int w, int h);
template<class A, class B, class O> void buildAtimesBIntegralImg(A *img_a, B *img_b, O *acu_img, int w, int h);

template<class P, class I, class O>void validate_box_filter(P *img,I *input_integral_img, O *result, int kw, int kh, int w, int h);

template<class I, class O>void box_filter(I *input_integral_img, O *result, int kw, int kh, int w, int h);

void showImg(unsigned char *img, int h, int w);
void showIntegralImg(int *img, int h, int w);
void showIntegralImg(double *img, int h, int w);
void showImgRange(int *img, int x1, int y1, int x2, int y2, int w);
void showImgRange(double *img, int x1, int y1, int x2, int y2, int w);

template <class T> inline void normalize_arr(T *arr, double *n_arr, int len);
template <class T> inline void normalize_arr(T *arr, float *n_arr, int len);
template <class T> inline void normalize_arr(T *arr, double *n_arr, int len, T max_v);

template <class I, class O> inline void normal_to_gray_img(I *n_arr, O *arr, int len, int max_intensity = IntensityLimit-1);

//for Fast cost volume 
//#define GUI_KERNEL_WIDTH  9
//#define GUI_KERNEL_HEIGHT 9
//#define GUI_EPSILON 0.0001 
//
#define GUI_KERNEL_WIDTH  2
#define GUI_KERNEL_HEIGHT 2
#define GUI_EPSILON 0.01

//T  -> guided imageªºtype
//P  -> imageªºtype
//IP -> image_ipªºtype
#define IP_INT_IMG_TYPE double
#define P_INT_IMG_TYPE double
#define I_INT_IMG_TYPE double
#define RESULT_TYPE float
template<class T, class P>
class guided_img{
private:
	int w, h;
	int kw, kh;
	double epsilon;

	I_INT_IMG_TYPE *int_img_i;
	P_INT_IMG_TYPE *int_img_p;
	IP_INT_IMG_TYPE *int_img_ip;
	I_INT_IMG_TYPE *int_img_ii;

	double *int_img_a;
	double *int_img_b;

	double *mean_i;
	double *mean_p;
	double *mean_ip;
	double *mean_ii;

	double *cov_ip;
	double *var_i;// variance of image i

	double *mean_a;
	double *mean_b;
public:
	void init(T *_img_i, P *_img_p, int _w, int _h);
	RESULT_TYPE *filtering();
	RESULT_TYPE *filtering_with_self(T *_img_i);

	T *img_i;
	P *img_p;
	RESULT_TYPE *filter_result;
};

template <class N, class I, class O> void apply_gray_guided_img_filtering(uchar *img, int h, int w, N *nor_img, guided_img<I, O> gfilter);

#include "cwz_integral_img.inl"

#endif