#include "common_func.h"
#include "cwz_cl_data_type.h"

#define IS_CWZ_EDGE_MATCHER_DEBUG true

//only for 1 channel use
struct c1_pxl_interval{
	unsigned short sID;
	unsigned short eID;
	unsigned short cID;
	unsigned short pxl_amt;
	unsigned int area;    //summation of intensities of pixels in an interval 
	unsigned int sq_area; //square of area 因為只是一維 所以int就夠用了
};//align 16 bytes for 32-bytes program

#define DP_TABLE_COST_TYPE float
#define DP_TABLE_Depth_Type uchar
struct dp_element{
	DP_TABLE_COST_TYPE var;
	DP_TABLE_COST_TYPE diagonal;
	DP_TABLE_COST_TYPE hor;
	DP_TABLE_COST_TYPE ver;
	DP_TABLE_Depth_Type dia_disp;
	DP_TABLE_Depth_Type hor_disp;
	DP_TABLE_Depth_Type ver_disp;
};

#define Pxl_Flat_Value 0

class cwz_edge_matcher{
private:
	match_info *info;
	
	dp_element **table ; //如果沒有要平行化, 每個scanline可以重複使用
	int max_interval_num;
	
	DP_TABLE_COST_TYPE slope;
	DP_TABLE_COST_TYPE shift;
	DP_TABLE_COST_TYPE occ_th;
	DP_TABLE_COST_TYPE max_cost;

	c1_pxl_interval *l_buf;
	c1_pxl_interval *r_buf;

	DP_TABLE_Depth_Type *disp_buf;
	int         disp_idx;
	int x_end_id;
	int y_end_id;

	int l_buf_idx;
	int r_buf_idx;
	int minPxlForAnInterval;

	int w, h;
public:
	void init(match_info *_info);
	void buildDPTable(uchar *l_sline, uchar *r_sline, int &cols, int &rows);
	void getPxlIntervalFromScanline(uchar *scanline, c1_pxl_interval *i_buf, int &buff_idx);

	DP_TABLE_COST_TYPE findPrimitivePath(int x, int y);

	void edgeMatch(uchar *left_img, uchar *right_img);

	void reinitForEndOfSline();
};

void cwz_edge_matcher::init(match_info *_info){
	this->w = info->img_width;
	this->h = info->img_height;
	this->info = _info;
	this->minPxlForAnInterval = 5;
	this->max_interval_num        = w / minPxlForAnInterval;

	this->disp_buf = new DP_TABLE_Depth_Type[max_interval_num];
	disp_idx = -1;

	this->occ_th   = 3;
	//計算線性方程的係數
	this->shift    = 2 * this->occ_th;
	this->slope    = (-2 * this->occ_th) / this->occ_th;
	this->max_cost = 99999;
	
	this->table = new dp_element*[max_interval_num];
	for(int i=0 ; i<max_interval_num ; i++) 
		this->table[i] = new dp_element[max_interval_num];
	
	this->l_buf = new c1_pxl_interval[max_interval_num];
	this->r_buf = new c1_pxl_interval[max_interval_num];
	int l_buf_idx = -1;
	int r_buf_idx = -1;
}

void cwz_edge_matcher::reinitForEndOfSline(){
	l_buf_idx = 0;
	r_buf_idx = 0;
	for(int i=0 ; i< max_interval_num; i++)
		memset(table[i], 0, sizeof(dp_element) * max_interval_num);
}

inline void intervalMatchCostFunc(c1_pxl_interval &left, c1_pxl_interval &right, dp_element &element){
	int amt = left.pxl_amt + right.pxl_amt;
	DP_TABLE_COST_TYPE area_mean = (left.area + right.area) /amt;
	element.var = ((left.sq_area + right.sq_area) / amt) - (area_mean * area_mean);
	element.diagonal = element.var * sqrt(left.pxl_amt*left.pxl_amt + right.pxl_amt*right.pxl_amt);
}

inline DP_TABLE_COST_TYPE occFunc(DP_TABLE_COST_TYPE cost, DP_TABLE_COST_TYPE th, DP_TABLE_COST_TYPE slope, DP_TABLE_COST_TYPE shift){
	return slope * cost + shift;
}
/**************************************************************************************************
在intervalHorCostFunc()與intervalVerCostFunc()中
	如果前一點的cost == max_cost 的話不算入的原因是因為

vertical跟horizontal line主要是看下一個點的var如果太高的話, 很值得避開
	所以ver跟hor會藉由occFunc算出一個較低的cost
	總之ver跟hor的計算目的是為了看下一個點是不是值得避開
	上一個點的cost只是拿來輔助計算

由於本演算法的目的 只打算讓edge matcher處理low texture的區域
	因此中間可能會跳過texture區域, 上一個點不肯定會是什麼
	所以若是上一點的cost為無限大, 就不要納入考量
	以免造成錯誤的結果
**************************************************************************************************/
inline void intervalHorCostFunc(dp_element **table, int k,int rows, int y, int x, DP_TABLE_COST_TYPE max_cost, DP_TABLE_COST_TYPE th, DP_TABLE_COST_TYPE slope, DP_TABLE_COST_TYPE shift){
	if(y==0){//第一個
		table[y][x].hor = k * occFunc( table[y+1][x].var, th, slope, shift);
	}else if(y==(rows-1)){//最後一個
		table[y][x].hor = k * occFunc( table[y-1][x].var, th, slope, shift);
	}else{
		if(table[y-1][x].var != max_cost)
			table[y][x].hor = k * occFunc( (table[y-1][x].var + table[y+1][x].var) / 2, th, slope, shift);
		else
			table[y][x].hor = k * occFunc(table[y+1][x].var, th, slope, shift);
	}
}
inline void intervalVerCostFunc(dp_element **table, int k,int cols, int y, int x, DP_TABLE_COST_TYPE max_cost, DP_TABLE_COST_TYPE th, DP_TABLE_COST_TYPE slope, DP_TABLE_COST_TYPE shift){
	if(x == 0){//第一個
		table[y][x].ver = k * occFunc( table[y][x+1].var, th, slope, shift);
	}else if(x == (cols-1)){//最後一個
		table[y][x].ver = k * occFunc( table[y][x-1].var, th, slope, shift);
	}else{
		if(table[y][x-1].var != max_cost)
			table[y][x].ver = k * occFunc( (table[y][x-1].var + table[y][x+1].var) / 2, th, slope, shift);
		else
			table[y][x].ver = k * occFunc( table[y][x+1].var, th, slope, shift);
	}
}

void cwz_edge_matcher::buildDPTable(uchar *l_sline, uchar *r_sline, int &cols, int &rows){
	getPxlIntervalFromScanline(l_sline, this->l_buf, this->l_buf_idx);
	getPxlIntervalFromScanline(r_sline, this->r_buf, this->r_buf_idx);
	cols = l_buf_idx+1;
	rows = r_buf_idx+1;
	int y_b_id = 0;
	int x_b_id = 0;//buffer的index計算會比格子的少1
	for(int y=1 ; y<rows; y++){
		y_b_id = y-1;
		for(int x=1 ; x<cols; x++){
			x_b_id = x-1;
			//確定兩者是否能夠比較(cid的差距低於max_disparity限制)
			if( (r_buf[y_b_id].sID > l_buf[x_b_id].sID) &&
				(r_buf[y_b_id].cID - l_buf[x_b_id].cID < info->max_x_d)
			){
				intervalMatchCostFunc(l_buf[x_b_id], r_buf[y_b_id], table[y][x]);
			}else{//不能match
				//cost設無限大
				table[y][x].var      = max_cost;
				table[y][x].diagonal = max_cost;
			}
			intervalVerCostFunc(table, r_buf[y_b_id].pxl_amt, cols, y  , x-1, max_cost, occ_th, slope, shift);
			intervalHorCostFunc(table, l_buf[x_b_id].pxl_amt, rows, y-1, x  , max_cost, occ_th, slope, shift);
		}
	}
}

void cwz_edge_matcher::getPxlIntervalFromScanline(uchar *scanline, c1_pxl_interval *i_buf, int &buff_idx){
	buff_idx = 0;
	unsigned short sID     = -1;
	unsigned short pxl_amt = 0;
	unsigned int area      = 0;
	unsigned int sq_area   = 0;
	int i;
	for(i=0 ; i<w ; i++){
		if(scanline[i] == Pxl_Flat_Value){//平坦區域
			if(sID == -1)
				sID = i;
			pxl_amt++;
			area    += scanline[i];
			sq_area += scanline[i] * scanline[i]; 
		}else{//是edge
			if(sID == -1)//目前沒有正在處理的interval element
				continue;
			else if(pxl_amt < minPxlForAnInterval){//正在處理的interval數量不足, 當作沒有
				sID     = -1;
				pxl_amt =  0;
				area    =  0;
				sq_area =  0;
			}else{//
				i_buf[buff_idx].sID     = sID;
				i_buf[buff_idx].eID     = i-1;
				i_buf[buff_idx].cID     = (sID+i)/2;
				i_buf[buff_idx].pxl_amt = pxl_amt;
				i_buf[buff_idx].area    = area;
				i_buf[buff_idx].sq_area = sq_area;
				buff_idx++;
			}
		}
	}
	//紀錄最後一筆
	if(sID != -1 && pxl_amt >= minPxlForAnInterval){
		i_buf[buff_idx].sID     = sID;
		i_buf[buff_idx].eID     = i-1;
		i_buf[buff_idx].cID     = (sID+i)/2;
		i_buf[buff_idx].pxl_amt = pxl_amt;
		i_buf[buff_idx].area    = area;
		i_buf[buff_idx].sq_area = sq_area;
		buff_idx++;
	}
}

void cwz_edge_matcher::edgeMatch(uchar *left_img, uchar *right_img){
	for(int i=0 ; i<this->h ; i++){
		int rows, cols;
		int base_idx = i * h;
		buildDPTable(&left_img[base_idx], &right_img[base_idx], cols, rows);
		this->disp_idx = -1;
		this->x_end_id = cols - 2;
		this->y_end_id = rows - 2;
		findPrimitivePath(0, 0);
	}
}

DP_TABLE_COST_TYPE cwz_edge_matcher::findPrimitivePath(int x, int y){
	if(x < this->x_end_id && y < this->y_end_id){
		DP_TABLE_COST_TYPE hor_cost = table[y  ][x+1].hor + findPrimitivePath(x+1, y  );
		DP_TABLE_COST_TYPE ver_cost = table[y+1][x  ].ver + findPrimitivePath(x  , y+1);
		DP_TABLE_COST_TYPE dia_cost = table[y+1][x+1].diagonal;

		if(dia_cost < max_cost)
			dia_cost += findPrimitivePath(x+1, y+1);

		if(hor_cost < ver_cost)
			if(hor_cost < dia_cost)
				
	}
}