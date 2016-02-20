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

template<class T, class O> void buildIntegralImg(T *img, O *acu_img, int w, int h){
	//set [0][0]
	acu_img[0] = img[0];
	//set [0][x]
	for(int x=1 ; x<w ; x++)
		acu_img[x] = acu_img[x-1] + img[x];
	//set [y][0]
	int preIdx = 0;
	int idx = w;
	for(int y=1 ; y<h ; y++){
		acu_img[idx] = acu_img[preIdx] + img[idx];
		preIdx = idx;
		idx += w;
	}
	//set [1][1] ~ [h-1][w-1]
	int preYIdx = 0;
	int yIdx    = w;
	for(int y=1 ; y<h ; y++)
	{
		for(int x=1 ; x<w ; x++){
			acu_img[yIdx + x] = acu_img[yIdx + x -1] + acu_img[preYIdx + x] - acu_img[preYIdx + x -1] + img[yIdx + x];
		}
		preYIdx = yIdx;
		yIdx +=w;
	}
}
template<class A, class B, class O> void buildAtimesBIntegralImg(A *img_a, B *img_b, O *acu_img, int w, int h){
	//set [0][0]
	acu_img[0] = (img_a[0] * img_b[0]);
	//set [0][x]
	for(int x=1 ; x<w ; x++)
		acu_img[x] = acu_img[x-1] + (img_a[x] * img_b[x]);
	//set [y][0]
	int preIdx = 0;
	int idx = w;
	for(int y=1 ; y<h ; y++){
		acu_img[idx] = acu_img[preIdx] + (img_a[idx] * img_b[idx]);
		preIdx = idx;
		idx += w;
	}
	//set [1][1] ~ [h-1][w-1]
	int preYIdx = 0;
	int yIdx    = w;
	for(int y=1 ; y<h ; y++)
	{
		for(int x=1 ; x<w ; x++)
			acu_img[yIdx + x] = acu_img[yIdx + x -1] + acu_img[preYIdx + x] - acu_img[preYIdx + x -1] 
		                      + (img_a[yIdx + x] * img_b[yIdx + x]);
		
		preYIdx = yIdx;
		yIdx +=w;
	}
}

template <class P>int calcArea(P *img, int x1, int y1, int x2, int y2, int w, int h){
	int sum=0;
	for(int i=x1 ; i<=x2 ; i++)
	for(int j=y1 ; j<=y2 ; j++)
	{
		sum += img[get_1d_idx(i, j, w)];
	}
	return sum;
}
template<class P, class I, class O>void validate_box_filter(P *img,I *input_integral_img, O *result, int kw, int kh, int w, int h){
	for(int y=0 ; y<h ; y++){
		int y1 = std::max(y - kh, 0);
		int y2 = std::min(y + kh, h-1);
		int ylen = y2 - y1 + 1;
		for(int x=0 ; x<w ; x++){
			int x1 = std::max(x - kw, 0);
			int x2 = std::min(x + kw, w-1);
			int xlen = x2 - x1 + 1;
			double n = ylen * xlen;
			int area = getArea<I>(input_integral_img, x1, y1, x2, y2, w, h);

			if(area != calcArea<P>(img, x1, y1, x2, y2, w, h)){
				printf("x1:%3d y1:%3d x2:%3d y2:%3d\n", x1, y1, x2, y2);
				printf("integral area:%d  |  validate area:%d\n", area, calcArea<P>(img, x1, y1, x2, y2, w, h));
				printf("input_integral_img x2 y2 value:%d\n", input_integral_img[get_1d_idx(x2, y2, w)]);
				printf("not equal!\n");
				getchar();
			}

			result[get_1d_idx(x, y, w)] = area / n;
		}
	}
}
template<class I, class O>void box_filter(I *input_integral_img, O *result, int kw, int kh, int w, int h){
	for(int y=0 ; y<h ; y++){
		int y1 = std::max(y - kh, 0);
		int y2 = std::min(y + kh, h-1);
		int ylen = y2 - y1 + 1;
		for(int x=0 ; x<w ; x++){
			int x1 = std::max(x - kw, 0);
			int x2 = std::min(x + kw, w-1);
			int xlen = x2 - x1 + 1;
			double n = ylen * xlen;
			result[get_1d_idx(x, y, w)] = getArea<I>(input_integral_img, x1, y1, x2, y2, w, h) / n;
		}
	}
}

template <class T> inline T get_max_value(T *arr, int len){
	T max_v = arr[0];
	for(int i=1 ; i<len; i++){
		if(arr[i] > max_v)
			max_v = arr[i];
	}
	return max_v;
}
template <class T> inline void normalize_arr(T *arr, double *n_arr, int len){
	T max_v = get_max_value<T>(arr, len);
	for(int i=0 ; i<len; i++)
		n_arr[i] = arr[i] / (double)max_v;
}

template<class T, class P>
void guided_img<T, P>::init(T *_img_i, P *_img_p, int _w, int _h){
	img_i = _img_i;
	img_p = _img_p;
	
	w     = _w;
	h     = _h;

	kw      = GUI_KERNEL_WIDTH;
	kh      = GUI_KERNEL_HEIGHT;
	epsilon = GUI_EPSILON;

	int_img_i  = new I_INT_IMG_TYPE[w * h];
	int_img_p  = new P_INT_IMG_TYPE[w * h];
	int_img_ip = new IP_INT_IMG_TYPE[w * h];
	int_img_ii = new I_INT_IMG_TYPE[w * h];

	int_img_a  = new double[w * h];
	int_img_b  = new double[w * h];

	mean_i  = new double[w * h];
	mean_p  = new double[w * h];
	mean_ip = new double[w * h];
	mean_ii = new double[w * h];

	var_i   = new double[w * h];
	cov_ip  = new double[w * h];

	mean_a  = new double[w * h];
	mean_b  = new double[w * h];

	filter_result = NULL;
	//filter_result = new double[w * h];
}

template<class T, class P>
RESULT_TYPE *guided_img<T, P>::filtering(){
	if(filter_result == NULL)
		filter_result = new RESULT_TYPE[w * h];

	int len = w * h;

	buildIntegralImg<T, I_INT_IMG_TYPE>(img_i, int_img_i, w, h);
	buildIntegralImg<P, P_INT_IMG_TYPE>(img_p, int_img_p, w, h);
	buildAtimesBIntegralImg<T, P, IP_INT_IMG_TYPE>(img_i, img_p, int_img_ip, w, h);
	buildAtimesBIntegralImg<T, T, I_INT_IMG_TYPE> (img_i, img_i, int_img_ii, w, h);

	box_filter<I_INT_IMG_TYPE , double>(int_img_i , mean_i , kw, kh, w, h);//mean_i = box(integral_i)
	box_filter<P_INT_IMG_TYPE , double>(int_img_p , mean_p , kw, kh, w, h);//mean_p = box(integral_p)
	box_filter<IP_INT_IMG_TYPE, double>(int_img_ip, mean_ip, kw, kh, w, h);//mean_ip = box(integral_ip)
	box_filter<I_INT_IMG_TYPE , double>(int_img_ii, mean_ii, kw, kh, w, h);//mean_ii = box(integral_ii)

	//cov_ip = mean_ip - (mean_i * mean_p)
	for(int i=0 ; i<len ; i++){  cov_ip[i] = mean_ip[i] - (mean_i[i] * mean_p[i]);  }

	//var_i = mean_ii - mean_i^2
	for(int i=0 ; i<len ; i++){  var_i[i] = mean_ii[i] - (mean_i[i] * mean_i[i]);  }

	//get a = cov_ip / (  var_i + epsilon  )
	for(int i=0 ; i<len ; i++){  mean_a[i] = cov_ip[i] / (var_i[i] + epsilon); }
	//get b = mean_p - a * mean_i
	for(int i=0 ; i<len ; i++){  mean_b[i] = mean_p[i] - mean_a[i] * mean_i[i];  }

	//get mean_a
	buildIntegralImg<double, double>(mean_a, int_img_a, w, h);
	box_filter<double, double>(int_img_a , mean_a , kw, kh, w, h);
	//get mean_b
	buildIntegralImg<double, double>(mean_b, int_img_b, w, h);
	box_filter<double, double>(int_img_b , mean_b , kw, kh, w, h);

	//get q = mean_a * I + mean_b
	for(int i=0 ; i<len ; i++){  filter_result[i] = mean_a[i] * img_i[i] + mean_b[i];  }

	/*
	printf("show mean_a\n");
	showImgRange(mean_a, 0, 0, 5, 5, w);
	printf("show cov_ip\n");
	showImgRange(cov_ip, 0, 0, 5, 5, w);
	printf("show var_i\n");
	showImgRange(var_i, 0, 0, 5, 5, w);
	getchar();
	*/

	return filter_result;
}

void guild_filter_example(const char *LeftIMGName, int max_intensity){
	cv::Mat my_img = cv::imread(LeftIMGName , CV_LOAD_IMAGE_GRAYSCALE);

	//create noise
		cv::Mat noise = cv::Mat(my_img.size(),CV_64F);
		cv::Mat input;
		normalize(my_img, input, 0.0, 1.0, CV_MINMAX, CV_64F);
		cv::randn(noise, 0, 0.1);
		input = input + noise;
		cv::normalize(input, input, 0.0, 1.0, CV_MINMAX, CV_64F);
		input.convertTo(input, CV_8UC1, 255, 0);
		//cv::imwrite("noisy_face.bmp", input);

	int w = my_img.cols;
	int h = my_img.rows;

	uchar *img = input.data;
	double *nor_img = new double[w * h];

	normalize_arr<uchar>(img, nor_img, w * h);

	guided_img<double, double> gfilter;
	gfilter.init(nor_img, nor_img, w, h);
	RESULT_TYPE *result = gfilter.filtering();

	//將normalized的filtering結果轉回0 ~ max_intensity的正常影像數值
	uchar *my_img_pixel = my_img.data;
	for(int i=0; i<w*h; i++){ 
		gfilter.filter_result[i] = result[i] * max_intensity;
	
		if(gfilter.filter_result[i] < 0)
			my_img_pixel[i] = 0;
		else if(gfilter.filter_result[i] > max_intensity)
			my_img_pixel[i] = max_intensity;
		else
			my_img_pixel[i] = std::floor(gfilter.filter_result[i]);
	}

	cv::namedWindow("noise", CV_WINDOW_KEEPRATIO);
	cv::imshow("noise",input);
	cvWaitKey(5);
	cv::namedWindow("denoised", CV_WINDOW_KEEPRATIO);
	cv::imshow("denoised",my_img);
	cv::waitKey(0);
}

/*void avoid_link_err(){
	guided_img<unsigned char, double> gfilter;
	unsigned char *img;
	double *result;
	int w=0, h=0;
	gfilter.init(img, result, w, h);
	gfilter.filtering();
}*/