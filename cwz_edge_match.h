#include "common_func.h"
#include "cwz_cl_data_type.h"

#define IS_CWZ_EDGE_MATCHER_DEBUG true
#define IS_CWZ_EDGE_MATCHER_edgeMatch_DEBUG true
#define IS_CWZ_EDGE_MATCHER_buildDPTable_DEBUG true
//#define IS_CWZ_EDGE_MATCHER_getPxlIntervalFromScanline_DEBUG true
#define IS_CWZ_EDGE_MATCHER_updateOptCost_DEBUG true
/**/
//only for 1 channel use
struct c1_pxl_interval{
	short sID;
	short eID;
	short cID;
	unsigned short pxl_amt;
	unsigned int area;    //summation of intensities of pixels in an interval 
	unsigned int sq_area; //square of area 因為只是一維 所以int就夠用了
};//align 16 bytes for 32-bytes program

#define DP_TABLE_COST_TYPE float
#define DP_TABLE_Depth_Type uchar
struct dp_element{
	union{
		DP_TABLE_COST_TYPE  var;      //used to save variance in buildDPTable()
		DP_TABLE_COST_TYPE  opt_cost; // used to keep optimal cost in updateOptCost()
	};
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
	
	DP_TABLE_COST_TYPE slope;
	DP_TABLE_COST_TYPE shift;
	DP_TABLE_COST_TYPE occ_th;
	DP_TABLE_COST_TYPE max_cost;

	c1_pxl_interval *l_buf;
	c1_pxl_interval *r_buf;

	int x_end_id;
	int y_end_id;

	int l_buf_len;
	int r_buf_len;
	int minPxlForAnInterval;

	int w, h;
public:
	void init(match_info *_info);
	void getPxlIntervalFromScanline(uchar *scanline, uchar *scanline_mask, c1_pxl_interval *i_buf, int &buff_idx);
	inline void calcCostForNormalDPElement(int x, int y, int cols, int row);
	void buildDPTable(uchar *l_sline, uchar *r_sline, uchar *l_sline_mask, uchar *r_sline_mask, int &x_end_id, int &y_end_id);
	
	DP_TABLE_COST_TYPE updateOptCost(int x, int y);
	void edgeMatch(uchar *left_img, uchar *right_img, uchar *left_mask, uchar *right_mask, DP_TABLE_Depth_Type *dmap);
};

void cwz_edge_matcher::init(match_info *_info){
	this->info = _info;
#if DP_TABLE_Depth_Type == uchar
	if(info->max_x_d < 255)
		this->max_disp_plus_1 = info->max_x_d + 1;
	else{
		printf("cwz_edge_matcher::init: error, info->max_x_d == 255. max_disp_plus_1 will be assign a value out of scope of uchar.\n");
		system("PAUSE");
	}
#else
	printf("cwz_edge_matcher::init: error, DP_TABLE_Depth_Type != uchar, please make sure max_disp_plus_1 will not go over the limit of DP_TABLE_Depth_Type's scope.\n");
	system("PAUSE");
#endif
	this->w = info->img_width;
	this->h = info->img_height;
	this->minPxlForAnInterval = 5;
	this->max_interval_num        = w / minPxlForAnInterval;

	this->occ_th     = 1000;  // occ_th ==> max_var
	this->max_cost   = 3000;
		
	//計算線性方程的係數
	this->shift    = 2 * this->occ_th;
	this->slope    = (-2 * this->occ_th) / this->occ_th;
	
	this->table = new dp_element*[max_interval_num];
	for(int i=0 ; i<max_interval_num ; i++) 
		this->table[i] = new dp_element[max_interval_num];
	
	this->l_buf = new c1_pxl_interval[max_interval_num];
	this->r_buf = new c1_pxl_interval[max_interval_num];
	int l_buf_len = -1;
	int r_buf_len = -1;
}
inline void recordInterval(c1_pxl_interval *i_buf, int &buff_idx, short &sID, int &i, unsigned short &pxl_amt, unsigned int &area, unsigned int &sq_area){
	i_buf[buff_idx].sID     = sID;
	i_buf[buff_idx].eID     = i-1;
	i_buf[buff_idx].cID     = (sID+i)/2;
	i_buf[buff_idx].pxl_amt = pxl_amt;
	i_buf[buff_idx].area    = area;
	i_buf[buff_idx].sq_area = sq_area;
	buff_idx++;
}
void cwz_edge_matcher::getPxlIntervalFromScanline(uchar *scanline, uchar *scanline_mask, c1_pxl_interval *i_buf, int &buff_idx){
	buff_idx = 0;
	short sID     = -1;
	unsigned short pxl_amt = 0;
	unsigned int area      = 0;
	unsigned int sq_area   = 0;
	int i;
#ifdef IS_CWZ_EDGE_MATCHER_getPxlIntervalFromScanline_DEBUG
	printf("		--- getPxlIntervalFromScanline Debug ---");
#endif

	for(i=0 ; i<w ; i++){
		if(scanline_mask[i] == Pxl_Flat_Value){//平坦區域
			if(sID == -1)
				sID = i;
			pxl_amt++;
			area    += scanline[i];
			sq_area += scanline[i] * scanline[i]; 
		}else{//是edge
			if(sID != -1){
				if(pxl_amt >= minPxlForAnInterval){//足夠大, 紀錄此interval
					recordInterval(i_buf, buff_idx, sID, i, pxl_amt, area, sq_area);
#ifdef IS_CWZ_EDGE_MATCHER_getPxlIntervalFromScanline_DEBUG
					printf("			edge區域\n");
					printf("			sID:%3d\n", i_buf[buff_idx-1].sID);
#endif
				}
				//洗掉interval data的暫存區
				sID     = -1;
				pxl_amt =  0;
				area    =  0;
				sq_area =  0;
			}
		}
	}
	//紀錄最後一筆
	if(sID != -1 && pxl_amt >= minPxlForAnInterval)
		recordInterval(i_buf, buff_idx, sID, i, pxl_amt, area, sq_area);
}
/**************************************************************************************************
cost計算是參考Stereo by Intra- and Inter-Scanline Search Using Dynamic Programming
	DP Table參考自Fig.3的2D Intra-Scanline Search
		table[x][y].dia = cost of ↘ ; matched path
				   .ver = cost of ↓ ; path(b)
				   .hor = cost of → ; path(a)
		[x,y  ] → [x+1,y  ] → [x+2,y  ]
		   ↓   ↘     ↓    ↘     ↓  
		[x,y+1] → [x+1,y+1] → [x+2,y+1] 
		   ↓   ↘     ↓    ↘     ↓  
		[x,y+2] → [x+1,y+2] → [x+2,y+2] 
	path(c)
		並不存在在此處, 這邊定義的路線只有水平、垂直、對角
	occFunc()
		參考Fig.10
	在Occlusion Cost計算上有做調整 ( 在函式occCostFunct()中 )
		不算上一個diagonal的cost
		直覺上垂直或水平都是為了閃避diagonal
		所以其cost應該由diagonal決定
**************************************************************************************************/
inline void matchCostFunc(c1_pxl_interval &left, c1_pxl_interval &right, dp_element &element){
	int amt = left.pxl_amt + right.pxl_amt;
	DP_TABLE_COST_TYPE area_mean = (left.area + right.area) /amt;
	element.var = ((left.sq_area + right.sq_area) / amt) - (area_mean * area_mean);
	element.diagonal = element.var * sqrt(left.pxl_amt*left.pxl_amt + right.pxl_amt*right.pxl_amt);
	element.dia_disp = left.sID - right.sID;
}
inline DP_TABLE_COST_TYPE occFunc(DP_TABLE_COST_TYPE cost, DP_TABLE_COST_TYPE th, DP_TABLE_COST_TYPE slope, DP_TABLE_COST_TYPE shift){
	//cost高於th會產生負的cost, 可是定義上希望cost最小就是0
	if(cost < th)
		return slope * cost + shift;
	else
		return 0;
}
inline DP_TABLE_COST_TYPE occCostFunct(int k, DP_TABLE_COST_TYPE var, DP_TABLE_COST_TYPE max_cost, DP_TABLE_COST_TYPE th, DP_TABLE_COST_TYPE slope, DP_TABLE_COST_TYPE shift){
	return k * occFunc(var, th, slope, shift);
}
inline void cwz_edge_matcher::calcCostForNormalDPElement(int x, int y, int cols, int rows){
	//確定兩者是否能夠比較(起始index)
	if( (l_buf[x].sID >= r_buf[y].sID) &&
		((l_buf[x].sID - r_buf[y].sID) < info->max_x_d)
	){
		matchCostFunc(l_buf[x], r_buf[y], table[y][x]);
		table[y][x].ver = occCostFunct(r_buf[y].pxl_amt, table[y][x].var, max_cost, occ_th, slope, shift);
		table[y][x].hor = occCostFunct(l_buf[x].pxl_amt, table[y][x].var, max_cost, occ_th, slope, shift);
	}else{//不能match
		//cost設無限大
		table[y][x].var      = max_cost;
		table[y][x].diagonal = max_cost;
		table[y][x].dia_disp = 0;
		//水平跟垂直的cost也應該設無限大減1
		//
		table[y][x].hor      = max_cost-1;
		table[y][x].ver      = max_cost-1;
	}
}
void cwz_edge_matcher::buildDPTable(uchar *l_sline, uchar *r_sline, uchar *l__sline_mask, uchar *r_sline_mask, int &x_end_id, int &y_end_id){
#ifdef IS_CWZ_EDGE_MATCHER_buildDPTable_DEBUG
	printf("	-- buildDPTable() DEBUG --\n");
#endif
	getPxlIntervalFromScanline(l_sline, l__sline_mask, this->l_buf, this->l_buf_len);
#ifdef IS_CWZ_EDGE_MATCHER_buildDPTable_DEBUG
	printf("	l_buf_len = %5d\n", l_buf_len);
	printf("	list l_buf[].sID and .pxl_amt:\n");
	for(int i=0; i<l_buf_len ; i++)
		printf("		sID: %3i | pxl_amt: %2i\n", l_buf[i].sID, l_buf[i].pxl_amt);
#endif
	getPxlIntervalFromScanline(r_sline, r_sline_mask, this->r_buf, this->r_buf_len);
#ifdef IS_CWZ_EDGE_MATCHER_buildDPTable_DEBUG
	printf("	r_buf_len = %5d\n", r_buf_len);
	printf("	list r_buf[].sID and .pxl_amt:\n");
	for(int i=0; i<r_buf_len ; i++)
		printf("		sID: %3i | pxl_amt: %2i\n", r_buf[i].sID, r_buf[i].pxl_amt);
#endif

	int cols = l_buf_len+1;
	int rows = r_buf_len+1;
	x_end_id = cols - 1;
	y_end_id = rows - 1;
	for(int y=0 ; y<y_end_id; y++){
		for(int x=0 ; x<x_end_id; x++){
			calcCostForNormalDPElement(x, y, cols, rows);
			table[y][x].isUpdated    = false;
		}
		//x的最後一個element單獨處理(只有垂直的cost需要算)
		table[y][x_end_id].ver = occCostFunct(r_buf[y].pxl_amt, table[y][x_end_id-1].var, max_cost, occ_th, slope, shift);
		table[y][x_end_id].isUpdated    = false;
	}
	//y的最後一排單獨處理(只需計算水平的cost)
	int former_y_id = y_end_id-1;
	for(int x=0 ; x<x_end_id; x++){
		table[y_end_id][x].hor = occCostFunct(l_buf[x].pxl_amt, table[former_y_id][x].var, max_cost, occ_th, slope, shift);
		table[y_end_id][x].isUpdated    = false;
	}
	//終點什麼都不用算
	table[y_end_id][x_end_id].isUpdated = false;
}

DP_TABLE_COST_TYPE cwz_edge_matcher::updateOptCost(int x, int y){
	if(table[y][x].isUpdated)
		return table[y][x].opt_cost;

	if(x < this->x_end_id){
		if(y < this->y_end_id){
			table[y][x].hor += updateOptCost(x+1, y  );
			table[y][x].ver += updateOptCost(x  , y+1);
			//若diagonal為max_cost 之後的路就都不用算了
			if(table[y][x].diagonal < max_cost)
				table[y][x].diagonal += updateOptCost(x+1, y+1);

			if(table[y][x].hor < table[y][x].ver)
				if(table[y][x].hor < table[y][x].diagonal){
					table[y][x].opt_cost = table[y][x].hor;
					table[y][x].isUpdated = true;
					table[y][x].next_prim_path = &table[y][x+1];
					table[y][x].prim_path_disp = 0;
				}else{
					table[y][x].opt_cost = table[y][x].ver;
					table[y][x].isUpdated = true;
					table[y][x].next_prim_path = &table[y+1][x];
					table[y][x].prim_path_disp = max_disp_plus_1;
				}
			else
				if(table[y][x].diagonal < table[y][x].ver){
					table[y][x].opt_cost = table[y][x].diagonal;
					table[y][x].isUpdated = true;
					table[y][x].next_prim_path = &table[y+1][x+1];
					table[y][x].prim_path_disp = table[y][x].dia_disp;
				}else{
					table[y][x].opt_cost = table[y][x].ver;
					table[y][x].isUpdated = true;
					table[y][x].next_prim_path = &table[y+1][x];
					table[y][x].prim_path_disp = max_disp_plus_1;
				}			
			return table[y][x].opt_cost;
		}else{
			table[y][x].hor += updateOptCost(x+1, y);
			table[y][x].opt_cost = table[y][x].hor;
			table[y][x].isUpdated = true;
			table[y][x].next_prim_path = &table[y][x+1];
			table[y][x].prim_path_disp = 0;
			return table[y][x].opt_cost;
		}
	}else if(y < this->y_end_id){
		table[y][x].ver += updateOptCost(x, y+1);
		table[y][x].opt_cost = table[y][x].ver;
		table[y][x].isUpdated = true;
		table[y][x].next_prim_path = &table[y+1][x];
		table[y][x].prim_path_disp = max_disp_plus_1;
		return table[y][x].opt_cost;
	}else{
		//the end
		table[y][x].opt_cost  = 0;
		table[y][x].isUpdated = true;
		table[y][x].next_prim_path = NULL;
		table[y][x].prim_path_disp = max_disp_plus_1;
		return 0;
	}
}
inline DP_TABLE_Depth_Type getDisparity(dp_element *&element, DP_TABLE_Depth_Type max_disp_plus_1){
	while(element->prim_path_disp == max_disp_plus_1){
		element = element->next_prim_path;
	}
	return element->prim_path_disp;
}
void cwz_edge_matcher::edgeMatch(uchar *left_img, uchar *right_img, uchar *left_mask, uchar *right_mask, DP_TABLE_Depth_Type *dmap){
#ifdef IS_CWZ_EDGE_MATCHER_edgeMatch_DEBUG
	printf("-- edgeMatch() DEBUG --\n");
	printf("w:%3d | h:%3d\n", w, h);
#endif
	for(int i=0 ; i<this->h ; i++){
		int base_idx = i * h;
#ifdef IS_CWZ_EDGE_MATCHER_edgeMatch_DEBUG
		printf("[i=%3d]: base_idx=%d\n", i, base_idx);
#endif
		buildDPTable(&left_img[base_idx], &right_img[base_idx], &left_mask[base_idx], &right_mask[base_idx], x_end_id, y_end_id);

#ifdef IS_CWZ_EDGE_MATCHER_edgeMatch_DEBUG
		for(int y=0 ; y<r_buf_len ; y++)
			for(int x=0 ; x<l_buf_len ; x++){
				printf("	(%3d,%3d):\n", x, y);
				printf("		dia(%2.4f) hor(%2.4f) ver(%2.4f) var(%2.4f)\n", table[y][x].diagonal, table[y][x].hor, table[y][x].ver, table[y][x].var);
			}
#endif

		updateOptCost(0, 0);

		//copy disparity value
#if DP_TABLE_Depth_Type == uchar
		//如果深度圖是以byte為單位, 可以直接使用memset
		dp_element *node = &table[0][0];
		for(int idx=0; idx<l_buf_len ; idx++){
			memset(&dmap[base_idx+l_buf[idx].sID], getDisparity(node, max_disp_plus_1), l_buf[idx].pxl_amt);
#ifdef IS_CWZ_EDGE_MATCHER_edgeMatch_DEBUG
			printf("Disparity: %d\n", node->prim_path_disp);
#endif
			node = node->next_prim_path;
		}
#else
		printf("cwz_edge_matcher::edgeMatch: DP_TABLE_Depth_Type is not uchar, please use for loop other than memset to setup dmap value.\n");
		system("PAUSE");
#endif

#ifdef IS_CWZ_EDGE_MATCHER_edgeMatch_DEBUG
		system("PAUSE");
#endif
	}
}

