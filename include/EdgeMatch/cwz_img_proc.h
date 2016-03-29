#ifndef CWZ_IMG_PROC_H
#define CWZ_IMG_PROC_H

#include "common_func.h"

//#define CWZ_EXPAND_GRAY_IMG_BORDER_DEBUG true //��expandGrayImgBorder()�PgetGrayImgFromExpandedImg() ��debug
void expandGrayImgBorder(uchar *img, uchar *tar_img, int w, int h, int kw, int kh);
void getGrayImgFromExpandedImg(uchar *exp_img, uchar *img, int w, int h, int kw, int kh);

#define CWZ_T_ANALY_KW 4
#define CWZ_T_ANALY_KH 4

class cwz_texture_analyzer{
private:
	int img_w, img_h;
	int th_kw, th_kh;
	uchar *img;
public:
	int expand_kw, expand_kh;//�B�~��kernel�e�׸����
	int exp_w, exp_h;        //expand���i�v�������e

	void init(int _w, int _h);
	uchar *createEmptyExpandImg();
	void expandImgBorder(uchar *img, uchar *expand_img);
};


#endif //CWZ_IMG_PROC_H