#include "cwz_integral_img.h"

void showImg(unsigned char *img, int h, int w){
	for(int y=0 ; y<h ; y++){
		for(int x=0 ; x<w ; x++){
			printf("%3d  ", img[y*w+x]);
		}
		printf("\n");
	}
}
void showIntegralImg(int *img, int h, int w){
	for(int y=0 ; y<h ; y++){
		for(int x=0 ; x<w ; x++){
			printf("%3d  ", img[y*w+x]);
		}
		printf("\n");
	}
}
void showIntegralImg(double *img, int h, int w){
	for(int y=0 ; y<h ; y++){
		for(int x=0 ; x<w ; x++){
			printf("%3.3f  ", img[y*w+x]);
		}
		printf("\n");
	}
}
void showImgRange(int *img, int x1, int y1, int x2, int y2, int w){
	for(int j=y1 ; j<=y2 ; j++)
	{
		for(int i=x1 ; i<=x2 ; i++){
			printf("%3d  ", img[get_1d_idx(i,j,w)]);
		}
		printf("\n");
	}
}
void showImgRange(double *img, int x1, int y1, int x2, int y2, int w){
	for(int j=y1 ; j<=y2 ; j++)
	{
		for(int i=x1 ; i<=x2 ; i++){
			printf("%3.3f  ", img[get_1d_idx(i,j,w)]);
		}
		printf("\n");
	}
}