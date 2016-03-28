#include "common_func.h"
#include "cwz_cl_data_type.h"
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
	this->minPxlForAnInterval = 10;
	this->max_interval_num    = w / minPxlForAnInterval + 1;

	this->max_cost   = w * MAX_INTENSITY;
	
	this->table = new dp_element*[max_interval_num];
	for(int i=0 ; i<max_interval_num ; i++) 
		this->table[i] = new dp_element[max_interval_num];
	
	this->l_buf = new c1_pxl_interval[max_interval_num];
	this->r_buf = new c1_pxl_interval[max_interval_num];
	int l_buf_len = -1;
	int r_buf_len = -1;
}
inline void recordInterval(c1_pxl_interval *i_buf, int &buff_idx, short &sID, int &i, unsigned short &pxl_amt, int &area){
	i_buf[buff_idx].sID     = sID;
	i_buf[buff_idx].eID     = i-1;
	i_buf[buff_idx].cID     = (sID+i)/2;
	i_buf[buff_idx].pxl_amt = pxl_amt;
	i_buf[buff_idx].area    = area;
	i_buf[buff_idx].mean    = area / pxl_amt;
	buff_idx++;
}
void cwz_edge_matcher::getPxlIntervalFromScanline(uchar *scanline, uchar *scanline_mask, c1_pxl_interval *i_buf, int &buff_idx){
	buff_idx = 0;
	short sID     = -1;
	unsigned short pxl_amt = 0;
	int area      = 0;
	int i;

	for(i=0 ; i<w ; i++){
		if(scanline_mask[i] == Pxl_Flat_Value){//平坦區域
			if(sID == -1)
				sID = i;
			pxl_amt++;
			area    += scanline[i];
		}else{//是edge
			if(sID != -1){
				if(pxl_amt >= minPxlForAnInterval){//足夠大, 紀錄此interval
					recordInterval(i_buf, buff_idx, sID, i, pxl_amt, area);
				}
				//洗掉interval data的暫存區
				sID     = -1;
				pxl_amt =  0;
				area    =  0;
			}
		}
	}
	//紀錄最後一筆
	if(sID != -1 && pxl_amt >= minPxlForAnInterval)
		recordInterval(i_buf, buff_idx, sID, i, pxl_amt, area);
}
/**************************************************************************************************
		table[x][y].dia = cost of ↘ ; matched path
				   .ver = cost of ↓ ; right interval skipped
				   .hor = cost of → ; left interval skipped
		[x,y  ] → [x+1,y  ] → [x+2,y  ]
		   ↓   ↘     ↓    ↘     ↓  
		[x,y+1] → [x+1,y+1] → [x+2,y+1] 
		   ↓   ↘     ↓    ↘     ↓  
		[x,y+2] → [x+1,y+2] → [x+2,y+2] 
**************************************************************************************************/
inline DP_TABLE_Depth_Type get_eID_disp_if_valid(c1_pxl_interval &left, c1_pxl_interval &right, int max_disparity){
	if(left.eID > right.eID){
		DP_TABLE_Depth_Type tmp = left.eID - right.eID;
		if(tmp < max_disparity)
			return tmp;
	}
	return 0;
}
inline void matchCostFunc(c1_pxl_interval &left, c1_pxl_interval &right, dp_element &element, int max_disparity){
	float mean_diff  = abs(left.mean - right.mean) + 1; 
	element.diagonal = abs((mean_diff*left.pxl_amt) - (mean_diff*right.pxl_amt));

	//get diagonal disparity
	element.dia_disp = left.sID - right.sID;
	if(element.dia_disp == 0)
		element.dia_disp = get_eID_disp_if_valid(left, right, max_disparity);
}
inline DP_TABLE_COST_TYPE occCostFunc(c1_pxl_interval &buf){
	return buf.area;
}
inline bool matchCondiCheck(c1_pxl_interval &l_buf, c1_pxl_interval &r_buf, match_info *info){
	if(l_buf.sID >= r_buf.sID)
		return ((l_buf.sID - r_buf.sID) < info->max_x_d);
		//if((l_buf.sID - r_buf.sID) < info->max_x_d)
			//return (abs(l_buf.pxl_amt - r_buf.pxl_amt) < (2 * std::min(l_buf.pxl_amt, r_buf.pxl_amt)));
	return false;
}
inline void cwz_edge_matcher::calcCostForNormalDPElement(int x, int y, int cols, int rows){
	if(matchCondiCheck(l_buf[x], r_buf[y], info)){
		matchCostFunc(l_buf[x], r_buf[y], table[y][x], info->max_x_d);	
	}else{//不能match
		table[y][x].diagonal = max_cost;
		table[y][x].dia_disp = 0;
	}
	table[y][x].ver = occCostFunc(r_buf[y]);
	table[y][x].hor = occCostFunc(l_buf[x]);
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
		table[y][x_end_id].ver = occCostFunc(r_buf[y]);
		table[y][x_end_id].isUpdated    = false;
	}
	//y的最後一排單獨處理(只需計算水平的cost)
	int former_y_id = y_end_id-1;
	for(int x=0 ; x<x_end_id; x++){
		table[y_end_id][x].hor = occCostFunc(l_buf[x]);
		table[y_end_id][x].isUpdated    = false;
	}
	//終點什麼都不用算
	table[y_end_id][x_end_id].isUpdated = false;
}

void cwz_edge_matcher::writeThreeWayCostToFile(std::string fname){
	cleanFile(fname);
	std::ofstream myfile (fname.c_str(), std::ios::app);

    //myfile << data <<"\n";
	for(int y=0 ; y<=table_y_e_id ; y++){
			for(int x=0 ; x<=table_x_e_id ; x++){
				myfile << "(" << x << ":" << y << ") , ";
				myfile << "["<<table[y][x].hor << "] , ,";
			}
			myfile << "\n";
			for(int x=0 ; x<=table_x_e_id ; x++){
				myfile << "[" << table[y][x].ver << "] , ";
				myfile << "[" << table[y][x].diagonal << "] , ,";
			}
			myfile << "\n";
			myfile << "\n";
	}

    myfile.close();
}
void cwz_edge_matcher::writeUpdatedOptCostToFile(std::string fname){
	cleanFile(fname);
	std::ofstream myfile (fname.c_str(), std::ios::app);

    //myfile << data <<"\n";
	for(int y=0 ; y<=table_y_e_id ; y++){
			for(int x=0 ; x<=table_x_e_id ; x++){
				myfile << "(" << x << ":" << y << "): "<< table[y][x].opt_cost<<",";
				if((table[y][x].hor < table[y][x].diagonal) &&
				   (table[y][x].hor < table[y][x].ver))
					myfile << "--- " << (int)table[y][x].prim_path_disp << ",";
				else
					myfile << " ,";
			}
			myfile << "\n";
			for(int x=0 ; x<=table_x_e_id ; x++){
				if((table[y][x].ver < table[y][x].diagonal) &&
				   (table[y][x].ver < table[y][x].hor))
					myfile << "| " << (int)table[y][x].prim_path_disp << ", ,";
				else if(table[y][x].prim_path_disp != 0)
					myfile << " ,\\" << (int)table[y][x].prim_path_disp <<",";
				else 
					myfile << " , ,";
			}
			myfile << "\n";
	}

    myfile.close();
}

inline void writeUpdateGuard(bool &isUpdated){
/*
	updateOptCost()
		放在一開始的
		if(table[y][x].isUpdated)
			return table[y][x].opt_cost;
		用來防止重複寫入的guard其實很陽春
		有些情況比較複雜會有多個分之同時通過guard
		到達後面
		第一個將isUpdated改成true之後
			由於後面幾個都已經通過了guard
			在寫之前沒有判斷isUpdated是否有被更心為true的話
			就會造成重複寫入
			那些ver dia hor都會重覆加入後面路徑的opt_cost
*/
#ifdef IS_CWZ_EDGE_MATCHER_DEBUG
	if(isUpdated){
		printf("Write Guard is passed, data has been overwritten.\n");
		system("PAUSE");
	}else
#endif
		isUpdated = true;
}
DP_TABLE_COST_TYPE cwz_edge_matcher::updateOptCost(int x, int y){
	if(table[y][x].isUpdated)
		return table[y][x].opt_cost;

	if(x < this->table_x_e_id){
		if(y < this->table_y_e_id){
			table[y][x].hor += updateOptCost(x+1, y  );
			table[y][x].ver += updateOptCost(x  , y+1);
			//若diagonal為max_cost 之後的路就都不用算了
			if(table[y][x].diagonal < max_cost)
				table[y][x].diagonal += updateOptCost(x+1, y+1);

			if(table[y][x].hor < table[y][x].ver){
				if(table[y][x].hor < table[y][x].diagonal){
					table[y][x].opt_cost = table[y][x].hor;
					writeUpdateGuard(table[y][x].isUpdated);
					table[y][x].next_prim_path = &table[y][x+1];
					table[y][x].prim_path_disp = 0;
				}else{
					table[y][x].opt_cost = table[y][x].diagonal;
					writeUpdateGuard(table[y][x].isUpdated);
					table[y][x].next_prim_path = &table[y+1][x+1];
					table[y][x].prim_path_disp = table[y][x].dia_disp;
				}
			}else{
				if(table[y][x].diagonal < table[y][x].ver){
					table[y][x].opt_cost = table[y][x].diagonal;
					writeUpdateGuard(table[y][x].isUpdated);
					table[y][x].next_prim_path = &table[y+1][x+1];
					table[y][x].prim_path_disp = table[y][x].dia_disp;
				}else{
					table[y][x].opt_cost = table[y][x].ver;
					writeUpdateGuard(table[y][x].isUpdated);
					table[y][x].next_prim_path = &table[y+1][x];
					table[y][x].prim_path_disp = max_disp_plus_1;
				}	
			}
#ifdef IS_CWZ_EDGE_MATCHER_updateOptCost_DEBUG
			if(GLOBAL_DEBUG_INDEX == GLOBAL_DEBUG_STOP_INDEX && 
			   (x==0 && y==0)){
				printf("dia:%f ver:%f hor:%f\n", table[y][x].diagonal, table[y][x].ver, table[y][x].hor);
				printf("l_buf[%d].sID:%d  | r_buf[%d].sID:%d\n", x, l_buf[x].sID, y, r_buf[y].sID);
				printf("l_buf[%d].eID:%d  | r_buf[%d].eID:%d\n", x, l_buf[x].eID, y, r_buf[y].eID);
				printf("dia_disp:%d\n", table[y][x].dia_disp);
				printf("table[%d][%d].prim_path_disp: %d\n", y, x, table[y][x].prim_path_disp);
				printf("table[%d][%d].opt_cost: %f\n", y, x, table[y][x].opt_cost);
				system("PAUSE");
			}
#endif
			return table[y][x].opt_cost;
		}else{
			table[y][x].hor += updateOptCost(x+1, y);
			table[y][x].opt_cost = table[y][x].hor;
			writeUpdateGuard(table[y][x].isUpdated);
			table[y][x].next_prim_path = &table[y][x+1];
			table[y][x].prim_path_disp = 0;
			return table[y][x].opt_cost;
		}
	}else if(y < this->table_y_e_id){
		table[y][x].ver += updateOptCost(x, y+1);
		table[y][x].opt_cost = table[y][x].ver;
		writeUpdateGuard(table[y][x].isUpdated);
		table[y][x].next_prim_path = &table[y+1][x];
		table[y][x].prim_path_disp = max_disp_plus_1;
		return table[y][x].opt_cost;
	}else{
		//the end
		table[y][x].opt_cost  = 0;
		writeUpdateGuard(table[y][x].isUpdated);
		table[y][x].next_prim_path = NULL;
		table[y][x].prim_path_disp = max_disp_plus_1;
		return 0;
	}
}
inline DP_TABLE_Depth_Type getDisparity(dp_element *&element, DP_TABLE_Depth_Type max_disp_plus_1){
	while(element->prim_path_disp == max_disp_plus_1)//如果前面的diagonal disparity在算的時候
													 //條件沒設好, 造成disparity變成負值
													 //因為是disparity type是unsigned char
													 //負的值會變成正的, 如果剛好等於max_disp_plus_1的值, 這邊就會誤判該disparity為vertical因此跳過
													 //造成不可預期的問題
													 //	(之前的剛好碰到左右inverval的sID相減為0因此使用eID: r_buf[0].eID = 235, l_buf[0].eID = 92)
												     //	(結果eID減下去disparity剛好變成113跟設定的max_disp_plus_1正好相同, 結果一個disparity被跳過)
												     //	造成while跑過頭, 存取了NULL的next_prim_path, 之後跑element->prim_path_disp, 程式直接crash)
													 //l.eID跟r.eID差距太大也不行, 否則會超過max_disparity, 也很可能剛好等於max_disp_plus_1, 造成crash
	{
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
#ifdef IS_CWZ_EDGE_MATCHER_edgeMatch_DEBUG
		GLOBAL_DEBUG_INDEX = i;
		printf("scanline index i:%d\n", i);
#endif
		int base_idx = i * w;
		buildDPTable(&left_img[base_idx], &right_img[base_idx], &left_mask[base_idx], &right_mask[base_idx], table_x_e_id, table_y_e_id);
#ifdef IS_CWZ_EDGE_MATCHER_edgeMatch_DEBUG
		if(i==GLOBAL_DEBUG_STOP_INDEX){
			printf("i:%d | table_x_e_id:%d | table_y_e_id:%d\n", i, table_x_e_id, table_y_e_id);
			writeThreeWayCostToFile("OriginCost.txt");
		}
#endif
		updateOptCost(0, 0);
#ifdef IS_CWZ_EDGE_MATCHER_edgeMatch_DEBUG
		if(i==GLOBAL_DEBUG_STOP_INDEX){
			writeThreeWayCostToFile("AggregatedCost.txt");
			writeUpdatedOptCostToFile("OptCost.txt");
		}
#endif
		//copy disparity value
#if DP_TABLE_Depth_Type == uchar
		//如果深度圖是以byte為單位, 可以直接使用memset
		dp_element *node = &table[0][0];

		for(int idx=0; idx<l_buf_len ; idx++){
			memset(&dmap[base_idx+l_buf[idx].sID], getDisparity(node, max_disp_plus_1), l_buf[idx].pxl_amt);
#ifdef IS_CWZ_EDGE_MATCHER_edgeMatch_DEBUG
			if(i == GLOBAL_DEBUG_STOP_INDEX)
				printf("l_buf[idx].sID:%d amt:%d | Disparity:%d\n", l_buf[idx].sID, l_buf[idx].pxl_amt, node->prim_path_disp);
#endif
			node = node->next_prim_path;
		}
#ifdef IS_CWZ_EDGE_MATCHER_DEBUG
		if(node->prim_path_disp != this->max_disp_plus_1){
			printf("edgeMatch error: end disp wrong: node->prim_path_disp != this->max_disp_plus_1\n");
			system("PAUSE");
		}
#endif
#else
		printf("cwz_edge_matcher::edgeMatch: DP_TABLE_Depth_Type is not uchar, please use for loop other than memset to setup dmap value.\n");
		system("PAUSE");
#endif

#ifdef IS_CWZ_EDGE_MATCHER_edgeMatch_DEBUG
		if(i==GLOBAL_DEBUG_STOP_INDEX)
			system("PAUSE");
#endif
	}
}

