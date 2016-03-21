#ifndef CWZ_IMG_PROC_H
#define CWZ_IMG_PROC_H

#include "common_func.h"

//#define CWZ_EXPAND_GRAY_IMG_BORDER_DEBUG true //對expandGrayImgBorder()與getGrayImgFromExpandedImg() 做debug
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
	int expand_kw, expand_kh;//額外的kernel寬度跟長度
	int exp_w, exp_h;        //expand後整張影像的長寬

	void init(int _w, int _h);
	uchar *createEmptyExpandImg();
	void expandImgBorder(uchar *img, uchar *expand_img);
};


#endif //CWZ_IMG_PROC_H