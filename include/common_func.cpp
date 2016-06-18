#include "common_func.h"
#include <iostream>
// for change window name

#ifdef __MINGW32__
	#include <windows.h>
#elif _WIN32
	#define _AFXDLL
	#include <afxwin.h> 
#endif

double  cwz_timer::m_pc_frequency = 0; 
__int64 cwz_timer::m_counter_start = 0;
double  cwz_timer::t_pc_frequency = 0; 
__int64 cwz_timer::t_counter_start = 0;

//讀line轉成blocks處理
int closestDelimiterPosi(std::string str, std::string *delimiters, int delCount, int &delLength){
    int minIdx = str.length();
	bool isFound = false;
	int posi;
	for(int i=0 ; i<delCount ; i++){
		if( (posi = str.find(delimiters[i])) != std::string::npos)
		{// delimiter is found
			if( posi < minIdx ){
				minIdx = posi;
				delLength = delimiters[i].length();
			}
			isFound = true;
		}
	}

	if(isFound)
		return minIdx;
    else
		return -1;
}
std::string *splitInstructions(std::string str, std::string *delimiters, int delCount, int &length){
    std::string buffer[10];
	int idx = 0;
	
	int posi=0;
	std::string tmp = "";

	int delimiterLen = 0; // match 到的delimiter字串長度
	while( (posi = closestDelimiterPosi(str, delimiters, delCount, delimiterLen)) != -1 )
	{
		tmp = str.substr(0, posi);
		//cout <<" " << tmp << " ";
		buffer[idx] = tmp;
		//cout <<" " << result[idx] << " ";
		str.erase(0, posi + delimiterLen);
		if(tmp.length() >= 1)
		{//要有東西才算一個
			idx++;
		}
	}
	
	tmp = str.substr(0, str.length());
	buffer[idx] = tmp;
	//cout <<" "<< result[idx] << " ";
	if(tmp.length() >= 1)
	{//要有東西才算一個
		idx++;
	}

    length = idx;
	std::string *result = new std::string[length];
	for(int i=0 ; i<length ;i++)
	{
    	result[i] = buffer[i];
	}

	return result;
}

int *c3_mat_to_1d_int_arr(cv::Mat img, int h, int w){
	int *arr = new int [h*w];
	uchar *img_arr = img.data;
	int img_idx = 0;
	for(int idx=0 ; idx<h*w; idx++)
	{
		arr[idx] =   img_arr[img_idx]       |
				    (img_arr[img_idx+1] << 8) |
					(img_arr[img_idx+2] << 16);
		img_idx+=3;
	}
	
	return arr;
}
void c3_mat_to_1d_int_arr(cv::Mat img, int *out, int h, int w){
	uchar *img_arr = img.data;
	int img_idx = 0;
	for(int idx=0 ; idx<h*w; idx++)
	{
		out[idx] =   img_arr[img_idx]       |
				    (img_arr[img_idx+1] << 8) |
					(img_arr[img_idx+2] << 16);
		img_idx+=3;
	}
}

uchar **int_2d_arr_to_gray_arr(int **color_arr, int h, int w){
	uchar **arr = new_2d_arr<uchar>(h, w);
	uchar *color = new uchar[3];
	int mask_b = 0xFF;
	int mask_g = mask_b << 8;
	int mask_r = mask_g << 8;
	for(int y=0; y<h ; y++)
	for(int x=0; x<w ; x++)
	{
		color[2] = (color_arr[y][x]&mask_b);
		color[1] = ((color_arr[y][x]&mask_g) >> 8);
		color[0] = ((color_arr[y][x]&mask_r) >> 16);
		arr[y][x] = rgb_2_gray( color );
	}
	delete[] color;
	return arr;
}
uchar *int_1d_arr_to_gray_arr(int *color_arr, int node_c){
	uchar *arr = new uchar[node_c];
	uchar *color = new uchar[3];
	int mask_b = 0xFF;
	int mask_g = mask_b << 8;
	int mask_r = mask_g << 8;
	for(int i=0; i<node_c ; i++)
	{
		color[2] = (color_arr[i]&mask_b);
		color[1] = ((color_arr[i]&mask_g) >> 8);
		color[0] = ((color_arr[i]&mask_r) >> 16);
		arr[i] = rgb_2_gray( color );
	}
	delete[] color;
	return arr;
}
void int_1d_to_gray_arr(int *color_arr, uchar *gray_arr, int node_c){
	uchar *color = new uchar[3];
	int mask_b = 0xFF;
	int mask_g = mask_b << 8;
	int mask_r = mask_g << 8;
	for(int i=0; i<node_c ; i++)
	{
		color[2] = (color_arr[i]&mask_b);
		color[1] = ((color_arr[i]&mask_g) >> 8);
		color[0] = ((color_arr[i]&mask_r) >> 16);
		gray_arr[i] = rgb_2_gray( color );
	}
	delete[] color;
}

uchar *int_1d_color_to_uchar_1d_color(int *in_arr, int node_c){
	uchar *uchar_arr = new uchar[node_c*3];
	int mask_b = 0xFF;
	int mask_g = 0xFF00;
	int mask_r = 0xFF0000;
	for(int i=0; i < node_c ; i++){
		int u_idx = i * 3;
		uchar_arr[u_idx  ] =    in_arr[i]&mask_b;
		uchar_arr[u_idx+1] = (((in_arr[i]&mask_g)) >> 8);
		uchar_arr[u_idx+2] = (((in_arr[i]&mask_r)) >> 16);
	}
	return uchar_arr;
}
void int_1d_color_to_uchar_1d_color(int *in_arr, uchar *out_arr, int node_c){
	int mask_b = 0xFF;
	int mask_g = 0xFF00;
	int mask_r = 0xFF0000;
	for(int i=0; i < node_c ; i++){
		int u_idx = i * 3;
		out_arr[u_idx  ] =    in_arr[i]&mask_b;
		out_arr[u_idx+1] = (((in_arr[i]&mask_g)) >> 8);
		out_arr[u_idx+2] = (((in_arr[i]&mask_r)) >> 16);
	}
}

void cwz_timer::start()
{
	//m_begin=clock();
	
    LARGE_INTEGER li;
    if(!QueryPerformanceFrequency(&li))
        std::cout << "QueryPerformanceFrequency failed!\n";

    m_pc_frequency = double(li.QuadPart);///1000.0;

    QueryPerformanceCounter(&li);
    m_counter_start = li.QuadPart;
}
void cwz_timer::t_start()
{
	//m_begin=clock();
	
    LARGE_INTEGER li;
    if(!QueryPerformanceFrequency(&li))
        std::cout << "QueryPerformanceFrequency failed!\n";

    t_pc_frequency = double(li.QuadPart);///1000.0;

    QueryPerformanceCounter(&li);
    t_counter_start = li.QuadPart;
}
double cwz_timer::t_stop(){
	LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return double(li.QuadPart-t_counter_start)/t_pc_frequency;
}
double cwz_timer::stop()
{
	//m_end=clock(); return ( double(m_end-m_begin)/CLOCKS_PER_SEC );
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return double(li.QuadPart-m_counter_start)/m_pc_frequency;
}
void cwz_timer::time_display(char *disp,int nr_frame){ printf("Running time (%s) is: %5.5f Seconds.\n",disp,stop()/nr_frame);}
void cwz_timer::t_time_display(char *disp,int nr_frame){ printf("Running time (%s) is: %5.5f Seconds.\n",disp,t_stop()/nr_frame);}

void cvmat_subsampling(cv::Mat &origin, cv::Mat &subsampled, int ch, int sub_pow){
	int s_h = origin.rows / sub_pow;
	int s_w = origin.cols / sub_pow;
	if(ch == 1)
		subsampled = cv::Mat(s_h, s_w, CV_8UC1);
	else
		subsampled = cv::Mat(s_h, s_w, CV_8UC3);

	for(int s_y = 0 ; s_y < s_h ; s_y++){
		int y = s_y * sub_pow;
		for(int s_x = 0 ; s_x < s_w ; s_x++){
			int x = s_x * sub_pow;
			if(ch == 1){
				subsampled.at<uchar>(s_y,s_x) = origin.at<uchar>(y,x);
			}else{
				int _s_x = s_x * 3;
				int _x   =   x * 3;
				subsampled.at<uchar>(s_y,_s_x  ) = origin.at<uchar>(y, _x  );
				subsampled.at<uchar>(s_y,_s_x+1) = origin.at<uchar>(y, _x+1);
				subsampled.at<uchar>(s_y,_s_x+2) = origin.at<uchar>(y, _x+2);
			}
		}
	}
}

void write_cv_img(int index, std::string title, uchar *pixels, int h, int w, int type){
	std::stringstream sstm;
	sstm.str("");
    sstm << title << index << ".bmp";
	cv::Mat img = cv::Mat(h, w, type);
	img.data = pixels;
    cv::imwrite(sstm.str().c_str(), img);
	img.release();
}
void write_cv_img(int index, std::string title, cv::Mat &img){
	std::stringstream sstm;
	sstm.str("");
    sstm << title << index << ".bmp";
    cv::imwrite(sstm.str().c_str(), img);
}

static struct img_and_title{
	cv::Mat img;
	std::string title;
};
static void MouseCallBackFunc(int event, int x, int y, int flags, void* userdata)
{
     /*if  ( event == cv::EVENT_LBUTTONDOWN )
     {
          std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
     }
     else if  ( event == cv::EVENT_RBUTTONDOWN )
     {
          std::cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
     }
     else if  ( event == cv::EVENT_MBUTTONDOWN )
     {
          std::cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
     }
     else if ( event == cv::EVENT_MOUSEMOVE )
     {
          std::cout << "Mouse move over the window - position (" << x << ", " << y << ")" << std::endl;
     }*/
	if ( event == cv::EVENT_MOUSEMOVE )
     {
          //std::cout << "Mouse move over the window - position (" << x << ", " << y << ")" << std::endl;
		  //printf("Pixel: %d\n", lastIndexedImg.at<uchar>(y, x));

		 img_and_title *mydata = ((img_and_title *)userdata);
		 cv::Mat cloneImg = mydata->img.clone();

		 int pixelcolor = cloneImg.at<uchar>(y, x);
		 char pixel[5];
		 sprintf(pixel, "%i", pixelcolor);
		 cv::putText(cloneImg, pixel, cv::Point(x+10, y+10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255), 2);
		 std::stringstream sstm;
		 sstm << "(" << x << "," << y << ")";
		 cv::putText(cloneImg, sstm.str().c_str(), cv::Point(x+10, y+35), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255), 2);
		 cv::imshow(mydata->title, cloneImg);
     }
}
static img_and_title *mouseCallBackUserData = NULL;
void show_cv_img(int index, std::string title, uchar *pixels, int h, int w, int c, bool shouldWait){
	cv::Mat img;
	if(c == 3)
		img = cv::Mat(h, w, CV_8UC3);
	else if(c == 1)
		img = cv::Mat(h, w, CV_8UC1);
	img.data = pixels;
	
	if(mouseCallBackUserData != NULL){
		free( mouseCallBackUserData );
		mouseCallBackUserData = NULL;
	}
	mouseCallBackUserData = new img_and_title();
	mouseCallBackUserData->img   = img;
	mouseCallBackUserData->title = title;

	std::stringstream sstm;
	sstm << title << "(" << index << ")";
#ifdef __MINGW32__
	cv::namedWindow(sstm.str().c_str(), CV_WINDOW_FREERATIO);
	cv::imshow(sstm.str().c_str(), img);
#elif _WIN32
	cv::namedWindow(title, CV_WINDOW_FREERATIO);
	cv::setMouseCallback(title, MouseCallBackFunc, mouseCallBackUserData);
	HWND hWnd = (HWND)cvGetWindowHandle(title.c_str());
	CWnd *wnd = CWnd::FromHandle(hWnd);
	CWnd *wndP = wnd->GetParent();
	wndP->SetWindowText((const char *) sstm.str().c_str()); 
	cv::imshow(title, img);
	
#endif
	if(shouldWait)
		cvWaitKey(0);
	else
		cvWaitKey(10);
}
void show_cv_img(int index, std::string title, cv::Mat &img, bool shouldWait){
	std::stringstream sstm;
	sstm << title << "(" << index << ")";

#ifdef __MINGW32__
	cv::namedWindow(sstm.str().c_str(), CV_WINDOW_KEEPRATIO);
	cv::imshow(sstm.str().c_str(), img);
#elif _WIN32
	cv::namedWindow(title, CV_WINDOW_KEEPRATIO);
	HWND hWnd = (HWND)cvGetWindowHandle(title.c_str());
	CWnd *wnd = CWnd::FromHandle(hWnd);
	CWnd *wndP = wnd->GetParent();
	wndP->SetWindowText((const char *) sstm.str().c_str()); 
	cv::imshow(title, img);
#endif
	if(shouldWait)
		cvWaitKey(0);
	else
		cvWaitKey(10);
}
void show_cv_img(std::string title, uchar *pixels, int h, int w, int c, bool shouldWait){
	cv::Mat img;
	if(c == 3)
		img = cv::Mat(h, w, CV_8UC3);
	else if(c == 1)
		img = cv::Mat(h, w, CV_8UC1);
	img.data = pixels;
	cv::namedWindow(title, CV_WINDOW_FREERATIO);
	cv::imshow(title, img);
	if(shouldWait)
		cvWaitKey(0);
	else
		cvWaitKey(10);
}
void show_cv_img(std::string fname, int c, bool shouldWait){
	cv::Mat img;
	if(c==3)
		img = cv::imread(fname, CV_LOAD_IMAGE_COLOR); 
	else
		img = cv::imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

	cv::namedWindow(fname, CV_WINDOW_FREERATIO);
	cv::imshow(fname, img);
	if(shouldWait == true)
		cvWaitKey(0);
	else
		cvWaitKey(10);
}
//檔案處理
bool cleanFile(std::string fname){

    //clean the file
    std::ofstream myfile (fname.c_str());
    myfile << "";
    myfile.close();

    return true;
}
void writeStrToFile(std::string fname, std::string data){
    std::ofstream myfile (fname.c_str(), std::ios::app);
    myfile << data <<"\n";
    myfile.close();
}
