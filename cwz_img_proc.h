#ifndef CWZ_IMG_PROC_H
#define CWZ_IMG_PROC_H

#include "common_func.h"

//#define CWZ_EXPAND_GRAY_IMG_BORDER_DEBUG true
void expandGrayImgBorder(uchar *img, uchar *tar_img, int w, int h, int kw, int kh){
	int new_w = w + kw + kw;
	int new_h = h + kh + kh;
	int tar_base_i=0;
	//top
	int src_base_i=0;
	for(int i=0 ; i<kh ; i++){
		tar_base_i = i * new_w;
		memset(&tar_img[tar_base_i]        , img[src_base_i]    , kw * sizeof(uchar));
		memcpy(&tar_img[tar_base_i+kw]     , &img[src_base_i]   , w  * sizeof(uchar));
		memset(&tar_img[tar_base_i+kw+w]   , img[src_base_i+w-1], kw * sizeof(uchar));
	}
	//middle
	for(int i=kh ; i<h+kh ; i++){
		tar_base_i = i * new_w;
		src_base_i = (i-kh) * w;
		memset(&tar_img[tar_base_i]        , img[src_base_i]    , kw * sizeof(uchar));
		memcpy(&tar_img[tar_base_i+kw]     , &img[src_base_i]   , w  * sizeof(uchar));
		memset(&tar_img[tar_base_i+kw+w]   , img[src_base_i+w-1], kw * sizeof(uchar));
	}
	//bottom
	src_base_i = (h-1) * w;
	for(int i=h+kh ; i<new_h ; i++){
		tar_base_i = i * new_w;
		memset(&tar_img[tar_base_i]        , img[src_base_i]    , kw * sizeof(uchar));
		memcpy(&tar_img[tar_base_i+kw]     , &img[src_base_i]   , w  * sizeof(uchar));
		memset(&tar_img[tar_base_i+kw+w]   , img[src_base_i+w-1], kw * sizeof(uchar));
	}
#ifdef CWZ_EXPAND_GRAY_IMG_BORDER_DEBUG
	show_cv_img("expandGrayImgBorder", tar_img, new_h, new_w, 1, true);
#endif
}

#define CWZ_T_ANALY_KW 3
#define CWZ_T_ANALY_KH 3

class cwz_texture_analyzer{
private:
	int img_w, img_h;
	int th_kw, th_kh;
	int expand_kw, expand_kh;
	uchar *img, *expand_img;
public:
	int exp_w, exp_h;

	void init(int _w, int _h);
	uchar *expandImgBorder(uchar *img);
};

void cwz_texture_analyzer::init(int _w, int _h){
	this->img_w = _w;
	this->img_h = _h;

	this->th_kw = CWZ_T_ANALY_KW;
	this->th_kh = CWZ_T_ANALY_KH;
	this->expand_kw = CWZ_T_ANALY_KW;
	this->expand_kh = CWZ_T_ANALY_KH;

	this->exp_w = img_w+expand_kw+expand_kw;
	this->exp_h = img_h+expand_kh+expand_kh;

	expand_img = new uchar[exp_w * exp_h];
}

uchar *cwz_texture_analyzer::expandImgBorder(uchar *_img){
	this->img = _img;
	expandGrayImgBorder(this->img, this->expand_img, this->img_w, this->img_h, this->expand_kw, this->expand_kh);
	return this->expand_img;
}

#endif //CWZ_IMG_PROC_H