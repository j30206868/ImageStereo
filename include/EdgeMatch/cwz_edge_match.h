#include "common_func.h"
#include "PxlMatch/cwz_cl_data_type.h"
#include <iostream>

#define IS_CWZ_EDGE_MATCHER_DEBUG true
/*static int GLOBAL_DEBUG_INDEX = 0;
static const int GLOBAL_DEBUG_STOP_INDEX = 309;
#define IS_CWZ_EDGE_MATCHER_edgeMatch_DEBUG true
#define IS_CWZ_EDGE_MATCHER_updateOptCost_DEBUG true
#define IS_CWZ_EDGE_MATCHER_buildDPTable_DEBUG true
*/

//only for 1 channel use
struct c1_pxl_interval{
	short sID;
	short eID;
	short cID;
	unsigned short pxl_amt;
	int area;    //summation of intensities of pixels in an interval 
	float mean; //
};//align 16 bytes for 32-bytes program

#define MAX_INTENSITY (IntensityLimit-1)
#define DP_TABLE_COST_TYPE float
#define DP_TABLE_Depth_Type uchar
struct dp_element{
	DP_TABLE_COST_TYPE  opt_cost; // used to keep optimal cost in updateOptCost()
	struct{
		DP_TABLE_COST_TYPE  diagonal;
		DP_TABLE_COST_TYPE  hor;
		DP_TABLE_COST_TYPE  ver;
	};
	dp_element *next_prim_path;
	union{
		DP_TABLE_Depth_Type dia_disp;
		DP_TABLE_Depth_Type prim_path_disp;
	};
	bool                isUpdated;
};

#define Pxl_Flat_Value 0

class cwz_edge_matcher{
private:
	match_info *info;
	
	DP_TABLE_Depth_Type max_disp_plus_1;

	dp_element **table ; //如果沒有要平行化, 每個scanline可以重複使用
	int max_interval_num;
	
	DP_TABLE_COST_TYPE max_cost;

	c1_pxl_interval *l_buf;
	c1_pxl_interval *r_buf;

	int table_x_e_id;
	int table_y_e_id;

	int l_buf_len;
	int r_buf_len;
	int minPxlForAnInterval;

	int w, h;
public:
	void init(match_info *_info);
	void getPxlIntervalFromScanline(uchar *scanline, uchar *scanline_mask, c1_pxl_interval *i_buf, int &buff_idx);
	inline void calcCostForNormalDPElement(int x, int y, int cols, int row);
	void buildDPTable(uchar *l_sline, uchar *r_sline, uchar *l_sline_mask, uchar *r_sline_mask, int &x_end_id, int &y_end_id);
	
	void writeThreeWayCostToFile(std::string fname);
	void writeUpdatedOptCostToFile(std::string fname);
	DP_TABLE_COST_TYPE updateOptCost(int x, int y);
	void edgeMatch(uchar *left_img, uchar *right_img, uchar *left_mask, uchar *right_mask, DP_TABLE_Depth_Type *dmap);
};
