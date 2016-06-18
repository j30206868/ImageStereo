#ifndef CWZ_VESSEL_H
#define CWZ_VESSEL_H

#include <opencv2\opencv.hpp>
#include "Vessel/math_3d.h"

class cwz_vessel{
private:
    int w, h, channel;
    int node_amt;
    uchar *img;
    uchar *vessl_img;
    Vector3f v_buffer[4];
    Vector3f c_buffer[4];

    int max_vertex_amt;
    int v_count;
    Vector3f *vertices;
public:
    void init(int _w, int _h);
    void loadImg(std::string fname);
    void getVesselImg();
    Vector3f * getVertexBuffer(int &len);
};

#endif
