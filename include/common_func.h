#ifndef CWZ_COMMON_FUNC_H
#define CWZ_COMMON_FUNC_H

#include <opencv2\opencv.hpp>

#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <fstream>

#include <time.h>


#define default_method 1  //tree=1 ; sgbm=2
#define IntensityLimit 256
#define doTreeRefinement false
#define defaultOcclusionTh 10 //其實應該跟max_x有相對關係
#define setWto1 false
#define max_d_to_img_len_pow 3
#define mst_channel 3
#define upsampling_mst_channel 3
#define DoGuidedFiltering true
const float default_sigma = 0.1;
const bool img_pre_mdf = false;
const bool mst_pre_mdf = true;
const bool depth_post_mdf = true;


#define CWZ_SHOW_LEFT_DMAP false
#define CWZ_SHOW_RIGHT_DMAP false

class cwz_timer{
public:
	static void start();
	static void t_start();
	static double stop();
	static double t_stop();
	static void time_display(char *disp,int nr_frame=1);
	static void t_time_display(char *disp,int nr_frame=1);
private: 
	static double m_pc_frequency; 
	static __int64 m_counter_start;
	static double t_pc_frequency; 
	static __int64 t_counter_start;
};

struct cl_match_elem{
	int *rgb;
	float *gradient;
	int node_c;
	cl_match_elem(){}
	cl_match_elem(int _node_c, int *_rgb, float *_gradient){
		this->node_c = _node_c;
		this->rgb      = _rgb;
		this->gradient = _gradient;
	}
	cl_match_elem(int _node_c){
		this->node_c = _node_c;
		this->rgb      = new int[node_c];
		this->gradient = new float[node_c];
	}
};

void cvmat_subsampling(cv::Mat &origin, cv::Mat &subsampled, int ch, int sub_pow);

template <class T>T *new_1d_arr(int len, T init_value){
	T *arr = new T[len];
	for(int i=0; i<len; i++){
		arr[i] = init_value;
	}
	return arr;
}

template <class T>T **new_2d_arr(int rows, int cols, T init_value){
	T **arr = new T*[rows];
	for(int i=0; i<rows; i++){
		arr[i] = new T[cols];
		for(int j=0; j<cols; j++){
			arr[i][j] = init_value;
		}
	}
	return arr;
}
template <class T>T **new_2d_arr(int rows, int cols){
	T **arr = new T*[rows];
	for(int i=0; i<rows; i++){
		arr[i] = new T[cols];
	}
	return arr;
}

template <class T>T ***new_3d_arr(int h, int w, int b){
	T ***arr = new T**[h];
	for(int i=0; i<h; i++){
		arr[i] = new T*[w];
		for(int j=0 ; j < w ; j++){
			arr[i][j] = new T[b];
		}
	}
	return arr;
}

template <class T>
void free_2d_arr(T **arr, int rows, int cols)
{
	for(int i=0; i<rows; i++)
	{
		delete[] arr[i];
	}
	delete[] arr;
}
template <class T>
void free_3d_arr(T ***arr, int h, int w, int b)
{
	for(int i=0; i<h ; i++)
	{
		for(int j=0; j<w; j++)
		{
			delete[] arr[i][j];
		}
		delete[]  arr[i];
	}
	delete[] arr;
}


//將format string切成陣列
int closestDelimiterPosi(std::string str, std::string *delimiters, int delCount, int &delLength);
std::string *splitInstructions(std::string str, std::string *delimiters, int delCount, int &length);

int *c3_mat_to_1d_int_arr(cv::Mat img, int h, int w);
void c3_mat_to_1d_int_arr(cv::Mat img, int *out, int h, int w);
template<class T>
T **map_1d_arr_to_2d_arr(T *arr, int h, int w){
	T **arr_2d = new T*[h];
	for(int y=0 ; y<h ; y++){
		int offset = y * w;
		arr_2d[y] = &arr[offset];
	}
	return arr_2d;
}

uchar **int_2d_arr_to_gray_arr(int **color_arr, int h, int w);
uchar *int_1d_arr_to_gray_arr(int *color_arr, int node_c);

void int_1d_to_gray_arr(int *color_arr, uchar *gray_arr, int node_c);

uchar *int_1d_color_to_uchar_1d_color(int *in_arr, int node_c);
void int_1d_color_to_uchar_1d_color(int *in_arr, uchar *out_arr, int node_c);
inline uchar rgb_2_gray(uchar*in){return(uchar(0.299*in[0]+0.587*in[1]+0.114*in[2]+0.5));}
inline uchar max_rgb(uchar *in){
	if(in[0] > in[1]){
		if(in[0] > in[2])
			return in[0];
		else
			return in[2];
	}else{
		if(in[1] > in[2])
			return in[1];
		else
			return in[2];
	}
};

//check if type equals
template<typename T, typename U>
struct is_same
{
    static const bool value = false;
};

template<typename T>
struct is_same<T, T>
{
    static const bool value = true;
};

template<typename T, typename U>
bool eqTypes() { return is_same<T, U>::value; }

void write_cv_img(int index, std::string title, cv::Mat &img);
void write_cv_img(int index, std::string title, uchar *pixels, int h, int w, int type);
void show_cv_img(int index, std::string title, uchar *pixels, int h, int w, int c, bool shouldWait = true);
void show_cv_img(int index, std::string title, cv::Mat &img, bool shouldWait = true);
void show_cv_img(std::string title, uchar *pixels, int h, int w, int c, bool shouldWait = true);
void show_cv_img(std::string fname, int c, bool shouldWait = true);

//檔案處理
bool cleanFile(std::string fname);
void writeStrToFile(std::string fname, std::string data);

#endif //CWZ_COMMON_FUNC_H
