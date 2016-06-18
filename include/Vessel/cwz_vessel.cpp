#include "Vessel/cwz_vessel.h"

void cwz_vessel::init(int _w, int _h){
    this->w = _w;
    this->h = _h;
    this->channel = 3;
    this->node_amt = _w * _h;
    this->img       = new uchar[_w * _h * channel];
    this->vessl_img = new uchar[_w * _h];
    memset(vessl_img, 0, node_amt);

    max_vertex_amt = 6 * (h-1) * (w-1);
    v_count = 0;
    vertices = new Vector3f[max_vertex_amt];
    printf("max_vertex_amt:%d\n", max_vertex_amt);
}

void show_cv_img(std::string title, uchar *pixels, int h, int w, int c, bool shouldWait){
    cv::Mat img;
    if(c == 3)
        img = cv::Mat(h, w, CV_8UC3);
    else if(c == 1)
        img = cv::Mat(h, w, CV_8UC1);
    img.data = pixels;
    cv::namedWindow(title, CV_WINDOW_AUTOSIZE);
    cv::imshow(title, img);
    if(shouldWait)
        cvWaitKey(0);
    else
        cvWaitKey(10);
}

void cwz_vessel::loadImg(std::string fname){
    cv::Mat im_mat = cv::imread(fname, CV_LOAD_IMAGE_COLOR);
    memcpy(this->img, im_mat.data, w * h * channel);
    show_cv_img("color img", im_mat.data, this->h, this->w, this->channel, false);
}

void cwz_vessel::getVesselImg(){
    if(channel == 3){
        int i=0;
        for(int _i=0 ; _i<node_amt*3; _i+=3){
            float red_ratio = (float)img[_i+2] / ((float)img[_i] + (float)img[_i+1]);
            if(red_ratio > 2){
                vessl_img[i] = 127;
            }
            i++;
        }
    }
    show_cv_img("vessel img", this->vessl_img, this->h, this->w, 1, false);
}

Vector3f * cwz_vessel::getVertexBuffer(int &len){
    v_count = 0;
    float default_z = -150;
    int half_w = w/2;
    int half_h = h/2;
    for(int idx=0 ; idx<node_amt ; idx++){
        int x = idx % w;
        int y = idx / w;

        //ignore border
        if(x == 0)
            continue;
        if(y == h-1)
            continue;
        //

        int v_id = 0;

        if(vessl_img[idx] > 0)
            v_buffer[v_id++] = Vector3f(x-half_w, y-half_h, default_z);

        if(vessl_img[idx-1] > 0)
            v_buffer[v_id++] = Vector3f(x-1-half_w, y-half_h, default_z);

        if(vessl_img[idx+w-1] > 0)
            v_buffer[v_id++] = Vector3f(x-1-half_w, y+1-half_h, default_z);

        if(vessl_img[idx+w] > 0)
            v_buffer[v_id++] = Vector3f(x-half_w, y+1-half_h, default_z);

        if(v_id < 3){
            //do nothing
        }else{//at least one triangle need to be pushed
            vertices[v_count++] = v_buffer[0];
            vertices[v_count++] = v_buffer[1];
            vertices[v_count++] = v_buffer[2];

            if(v_id == 4){//second triangle need to be pushed
                vertices[v_count++] = v_buffer[0];
                vertices[v_count++] = v_buffer[3];
                vertices[v_count++] = v_buffer[2];
            }
        }
    }
    len = v_count;
    return vertices;
}
