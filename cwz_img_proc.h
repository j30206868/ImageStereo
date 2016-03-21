#ifndef CWZ_IMG_PROC_H
#define CWZ_IMG_PROC_H

#include "common_func.h"

//#define CWZ_EXPAND_GRAY_IMG_BORDER_DEBUG true //對expandGrayImgBorder()與getGrayImgFromExpandedImg() 做debug
void expandGrayImgBorder(uchar *img, uchar *tar_img, int w, int h, int kw, int kh);
void getGrayImgFromExpandedImg(uchar *exp_img, uchar *img, int w, int h, int kw, int kh);

void expandGrayImgBorder(uchar *img, uchar *tar_img, int w, int h, int kw, int kh){
#ifdef CWZ_EXPAND_GRAY_IMG_BORDER_DEBUG
	//把kw跟kh放大來看 就知道邊緣有沒有copy錯誤
	kw = 30;
	kh = 30;
	int new_w = w + kw + kw;
	int new_h = h + kh + kh;
	tar_img = new uchar[new_w * new_h];
#else
	int new_w = w + kw + kw;
	int new_h = h + kh + kh;
#endif
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
	uchar *buffer_img = new uchar[w*h];
	getGrayImgFromExpandedImg(tar_img, buffer_img, w, h, kw, kh);
	//從expanded image結出中間的部分(應該要完全等於輸入的原影像, 因此相減後所有值都要是0)
	bool isDiff = false;
	for(int i=0 ; i<w*h; i++){
		int v = buffer_img[i] - img[i];
		if(v!=0){
			isDiff = true;
		}
	}

	//一開始已經將kw跟kh放的非常大 可以用眼睛觀察上下左右的copy有沒有出錯
	//這邊則是將copy後的影像中間挖出來 比對原輸入影像 確認值都是正確的, 也可以順便debug getGrayImgFromExpandedImg()這個function
	if(isDiff){
		printf("expandGrayImgBorder: debug error! extracted image is different from input image.\n");
		printf("function getGrayImgFromExpandedImg() may wrong!\n");
	}else{
		printf("expandGrayImgBorder: debug correct! extracted image is the same as input image.\n");
		printf("function getGrayImgFromExpandedImg() is correct!\n");
	}

	show_cv_img("Expanded image", tar_img, new_h, new_w, 1, false);
	show_cv_img("Origin image", buffer_img, h, w, 1, true);
	delete[] buffer_img;
	delete[] tar_img;
#endif
}
void getGrayImgFromExpandedImg(uchar *exp_img, uchar *img, int w, int h, int kw, int kh){
	int new_w = w + kw + kw;
	int new_h = h + kh + kh;
	int tar_base_i = new_w * kh + kw;
	int src_base_i = 0;
	do{
		memcpy(&img[src_base_i], &exp_img[tar_base_i], w*sizeof(uchar));
		tar_base_i+=new_w;
		src_base_i+=w;
	}while(src_base_i < w*h);
}

#define CWZ_T_ANALY_KW 4
#define CWZ_T_ANALY_KH 4

class cwz_texture_analyzer{
private:
	int img_w, img_h;
	int th_kw, th_kh;
	uchar *img;
public:
	int expand_kw, expand_kh;//額外的kernel寬度跟長度
	int exp_w, exp_h;        //expand後整張影像的長寬

	void init(int _w, int _h);
	uchar *createEmptyExpandImg();
	void expandImgBorder(uchar *img, uchar *expand_img);
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
}
uchar *cwz_texture_analyzer::createEmptyExpandImg(){
	return new uchar[exp_w * exp_h];
}
void cwz_texture_analyzer::expandImgBorder(uchar *_img, uchar *expand_img){
	this->img = _img;
	expandGrayImgBorder(this->img, expand_img, this->img_w, this->img_h, this->expand_kw, this->expand_kh);
}

#endif //CWZ_IMG_PROC_H