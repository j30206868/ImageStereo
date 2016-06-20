#include "cwz_mst.h"

float cwz_mst::sigma = default_sigma;
bool cwz_mst::setWtoOne = setWto1;
float cwz_mst::upbound = 1;
float *cwz_mst::whistogram = new float[IntensityLimit];

inline int get_1d_idx_from_2d(int x, int y, int w){
	return y * w + x;
}
inline void addEdge(TEleUnit *img, int **edge_node_list, short *distance, int edge_idx, int x0, int y0, int x1, int y1, int w, int channel){
	int idx0 = edge_node_list[edge_idx][0] = get_1d_idx_from_2d(x0, y0, w);
	int idx1 = edge_node_list[edge_idx][1] = get_1d_idx_from_2d(x1, y1, w);

	if( channel == 1 ){
		distance[edge_idx] = std::abs(img[idx0] - img[idx1]);
	}else if( channel == 3 ){
		idx0 *= 3;
		idx1 *= 3;

		uchar b = std::abs(img[idx0  ] - img[idx1  ]);
		uchar g = std::abs(img[idx0+1] - img[idx1+1]);
		uchar r = std::abs(img[idx0+2] - img[idx1+2]);

		uchar max_distance = std::max( std::max(b, g), r);

		distance[edge_idx] =  max_distance;
	}else{
		printf("cwz_mst addEdge(...): channel is not equal to 1 nor 3.\n");
		system("PAUSE");
	}
}
inline void push(int id_in, int *stack, int &end){
	end++;
	stack[end] = id_in;
}
inline int pop(int *stack, int &end){
	int result = -1;
	if(end > -1){
		result = stack[end];
		end--;
	}
	return result;
}

float *cwz_mst::get_agt_result(){
	return this->agt_result;
}
void cwz_mst::init(int _h, int _w, int _ch, int max_x_dis, int max_y_dis){
	this->h = _h;
	this->w = _w;

	this->max_x_disparity = max_x_dis;
	this->max_y_disparity = max_y_dis;

	this->node_amt = _h * _w;
	this->edge_amt = (_h-1) * _w + h * (_w-1);
	this->channel = _ch;

	this->edge_node_list = new_2d_arr<int>(this->edge_amt, 2);
	this->distance = new short[edge_amt];
	this->cost_sorted_edge_idx = new int[edge_amt];

	this->node_conn_node_list = new_2d_arr<int>(this->node_amt, 4);
	this->node_conn_weights   = new_2d_arr<int>(this->node_amt, 4);
	this->node_conn_node_num  = new_1d_arr(this->node_amt, 0);

	this->node_group = new int[this->node_amt];
	for(int i=0 ; i<this->node_amt ; i++){ this->node_group[i] = i; }

	this->id_stack = new int[this->node_amt];
	this->id_stack_e = -1;
	this->node_parent_id = new_1d_arr<int>(this->edge_amt, -1);
	//最終要使用的 tree 結構
	this->child_node_num = new_1d_arr<int>(this->node_amt, 0);
	this->child_node_list = new_2d_arr<int>( this->node_amt, 4, -1);
	this->node_idx_from_p_to_c= new_1d_arr(this->node_amt, -1);
	this->node_weight = new int[this->node_amt];

	for(int i=0 ; i<IntensityLimit ; i++){ histogram[i] = 0; }

	this->agt_result = new float[ node_amt * max_x_disparity ];
	this->best_disparity = new CWZDISPTYPE[node_amt];

	for(int i=0 ; i<IntensityLimit ; i++){
		if(cwz_mst::setWtoOne)
			cwz_mst::whistogram[i] = exp(-double(1) / (cwz_mst::sigma * (IntensityLimit - 1)));
		else{
			cwz_mst::whistogram[i] = exp(-double(i) / (cwz_mst::sigma * (IntensityLimit - 1)));
		
			if(cwz_mst::whistogram[i] > cwz_mst::upbound){
				cwz_mst::whistogram[i] = cwz_mst::upbound;
			}
		}
	}

	this->isInit = true;
}
void cwz_mst::set_img(TEleUnit *_img){
	this->img = _img;
	this->hasImg = true;
}
void cwz_mst::build_edges(){
	int y0,y1,x0,x1;
	int edge_idx = 0;
//先加入橫邊(左右的)
	for(y0=0;y0<h;y0++)
	{
		y1=y0;
		for(int x0=0;x0<w-1;x0++)
		{
			x1=x0+1;
			addEdge(img, edge_node_list, distance,  edge_idx++, x0, y0, x1, y1, w, channel);
		}
	}
//再加入直邊(上下的)
	for(int x0=0;x0<w;x0++)
	{
		x1=x0;
		for(y0=0;y0<h-1;y0++)
		{
			y1=y0+1;
			addEdge(img, edge_node_list, distance,  edge_idx++, x0, y0, x1, y1, w, channel);
		}
	}
}
void cwz_mst::counting_sort(){
	for(int i=0 ; i<edge_amt ; i++){ histogram[ distance[i] ]++; }
	//calculate start index for each intensity level
	int his_before = histogram[0];
	int his_this;
	histogram[0] = 0;
	for(int i=1 ; i<IntensityLimit ; i++){
		his_this = histogram[i]; 
		histogram[i] = histogram[i-1] + his_before;
		his_before = his_this;
	}
	for(int i=0 ; i<edge_amt ; i++){
		int deserved_order = histogram[ distance[i] ]++;
		cost_sorted_edge_idx[ deserved_order ] = i;
	}
}
int  cwz_mst::findset(int i){
	if(node_group[i] != i){
		node_group[i] = findset(node_group[i]);
	}
	return node_group[i];
}
void cwz_mst::kruskal_mst(){
	for(int i=0 ; i<edge_amt ; i++){
		int edge_idx = cost_sorted_edge_idx[i];

		int n0 = edge_node_list[ edge_idx ][0];
		int n1 = edge_node_list[ edge_idx ][1];
		
		int p0 = findset(n0);
		int p1 = findset(n1);

		if(p0 != p1){//此兩點不在同一個集合中
			//點與點互相連結
			node_conn_node_list[n0][ node_conn_node_num[n0] ] = n1;
			node_conn_node_list[n1][ node_conn_node_num[n1] ] = n0;

			node_conn_weights[n0][ node_conn_node_num[n0] ] = distance[edge_idx];
			node_conn_weights[n1][ node_conn_node_num[n1] ] = distance[edge_idx];

			node_conn_node_num[n0]++;
			node_conn_node_num[n1]++;

			node_group[p0] = p1;
		}
	}
}
void cwz_mst::build_tree(){
	int parent_id = 0;
	int possible_child_id = -1;
	int t_c = 0;//tree node counter

	node_idx_from_p_to_c[ t_c++ ] = parent_id;
	node_weight[parent_id]    = 0;
	node_parent_id[parent_id] = 0;

	push(0, id_stack, id_stack_e);
	while( id_stack_e > -1 ){
		parent_id = pop(id_stack, id_stack_e);

		for(int edge_c=0 ; edge_c < node_conn_node_num[parent_id] ; edge_c++){
			possible_child_id = node_conn_node_list[parent_id][edge_c];
			if( node_parent_id[possible_child_id] != -1 ){
				continue;//its the parent, pass
			}
			node_idx_from_p_to_c[ t_c++ ] = possible_child_id;
			node_weight[possible_child_id] = node_conn_weights[parent_id][edge_c];
			node_parent_id[possible_child_id] = parent_id;

			child_node_list[parent_id][ child_node_num[parent_id]++ ] = possible_child_id;

			if( node_conn_node_num[possible_child_id] > 1)//definitely has one edge connect to parent(num eq 1 is a leaf node)
				push(possible_child_id, id_stack, id_stack_e);
		}
	}
}
//
void cwz_mst::cost_agt(){
	//up agt
	for(int i = node_amt-1 ; i >= 0 ; i--){
		int node_i = node_idx_from_p_to_c[i];
		int node_disparity_i = node_i * max_x_disparity;

		for(int child_c=0 ; child_c < child_node_num[node_i] ; child_c++){
			int child_i = child_node_list[node_i][child_c];
			int child_disparity_i = child_i * max_x_disparity;

			for(int d=0 ; d<max_x_disparity ; d++){
				agt_result[ node_disparity_i+d ] += agt_result[ child_disparity_i+d ] * cwz_mst::whistogram[ node_weight[child_i] ];
			}
		}
	}
	//down agt
	for(int i=1 ; i<node_amt ; i++){
		int node_i = node_idx_from_p_to_c[i];
		int parent_i = node_parent_id[node_i];
		int node_disparity_i = node_i * max_x_disparity;
		int parent_disparity_i = parent_i * max_x_disparity;

		float w = cwz_mst::whistogram[ node_weight[ node_i ] ];
		float one_m_sqw = (1.0 - w * w);

		for(int d=0 ; d<max_x_disparity ; d++){
			agt_result[node_disparity_i+d] = w         * agt_result[ parent_disparity_i+d ] +
											 one_m_sqw * agt_result[node_disparity_i+d];
		}
	}
}
void cwz_mst::cost_agt(float *match_cost_result){
	//up agt
	for(int i = node_amt-1 ; i >= 0 ; i--){
		int node_i = node_idx_from_p_to_c[i];
		int node_disparity_i = node_i * max_x_disparity;

		for(int d=0 ; d<max_x_disparity ; d++)
			agt_result[ node_disparity_i+d ] = match_cost_result[ node_disparity_i+d ];

		for(int child_c=0 ; child_c < child_node_num[node_i] ; child_c++){
			int child_i = child_node_list[node_i][child_c];
			int child_disparity_i = child_i * max_x_disparity;

			for(int d=0 ; d<max_x_disparity ; d++){
				agt_result[ node_disparity_i+d ] += agt_result[ child_disparity_i+d ] * cwz_mst::whistogram[ node_weight[child_i] ];
			}
		}
	}
	//down agt
	for(int i=1 ; i<node_amt ; i++){
		int node_i = node_idx_from_p_to_c[i];
		int parent_i = node_parent_id[node_i];
		int node_disparity_i = node_i * max_x_disparity;
		int parent_disparity_i = parent_i * max_x_disparity;

		float w = cwz_mst::whistogram[ node_weight[ node_i ] ];
		float one_m_sqw = (1.0 - w * w);

		for(int d=0 ; d<max_x_disparity ; d++){
			agt_result[node_disparity_i+d] = w         * agt_result[ parent_disparity_i+d ] +
											 one_m_sqw * agt_result[node_disparity_i+d];
		}
	}
}
CWZDISPTYPE *cwz_mst::pick_best_dispairty(){
	for(int i=0 ; i<this->node_amt ; i++){
		int i_ = i * max_x_disparity;

		float min_cost = agt_result[i_+0];
		best_disparity[i] = 0;
		for(int d=1 ; d<max_x_disparity ; d++){
			if( agt_result[i_+d] < min_cost ){
				best_disparity[i] = d;
				min_cost = agt_result[i_+d];
			}
		}
	}
	return best_disparity;
}
//
void cwz_mst::mst(){
	if( this->isInit && this->hasImg ){
		this->build_edges();
		this->counting_sort();
		this->kruskal_mst();
		this->build_tree();
	}
}
void cwz_mst::profile_mst(){
	printf("--	start mst profiling	--\n");
	double total = 0;
	double build_edge_s;
	double counting_sort_s;
	double kruskal_s;
	double build_tree_s;
	time_t start;
	if( this->isInit && this->hasImg ){
		start = clock();
		this->build_edges();
		build_edge_s = double(clock() - start) / CLOCKS_PER_SEC;

		start = clock();
		this->counting_sort();
		counting_sort_s = double(clock() - start) / CLOCKS_PER_SEC;

		start = clock();
		this->kruskal_mst();
		kruskal_s = double(clock() - start) / CLOCKS_PER_SEC;

		start = clock();
		this->build_tree();
		build_tree_s = double(clock() - start) / CLOCKS_PER_SEC;

		printf("build_edges  : %2.4fs\n", build_edge_s);
		printf("counting_sort: %2.4fs\n", counting_sort_s);
		printf("kruskal_mst  : %2.4fs\n", kruskal_s);
		printf("build_tree   : %2.4fs\n", build_tree_s);
		printf("---------------------\n");
		printf("   total time: %2.4fs\n", build_edge_s+counting_sort_s+kruskal_s+build_tree_s);

		int total_w = 0;
		for(int i=0 ; i<this->node_amt ; i++){
			total_w += node_weight[i];
			//printf("[%3d]weight:%d\n", i, node_weight[i]);
			//system("PAUSE");
		}
		printf("     node_amt: %d nodes\n", node_amt);
		printf(" total weight: %d\n", total_w);
	}
	printf("--	endof mst profiling	--\n");
}
//for reuse
void cwz_mst::reinit(){
	memset(this->histogram, 0, sizeof(int) * IntensityLimit);
	for(int i=0 ; i<this->node_amt ; i++){ this->node_group[i] = i; }
	memset(this->node_conn_node_num, 0, sizeof(int) * this->node_amt);
	this->id_stack_e = -1;
	memset(this->child_node_num, 0, sizeof(int) * this->node_amt); 
	memset(this->node_parent_id, -1, sizeof(int) * this->node_amt); 
}

void cwz_mst::updateHistogram(){
	for(int i=0 ; i<IntensityLimit ; i++){
		if(cwz_mst::setWtoOne)
			cwz_mst::whistogram[i] = exp(-double(1) / (cwz_mst::sigma * (IntensityLimit - 1)));
		else{
			cwz_mst::whistogram[i] = exp(-double(i) / (cwz_mst::sigma * (IntensityLimit - 1)));
			//set up bound
			if(cwz_mst::whistogram[i] > cwz_mst::upbound){
				cwz_mst::whistogram[i] = cwz_mst::upbound;
			}
		}
	}
}

void cwz_mst::updateSigma(float _sigma){
	cwz_mst::sigma = _sigma;
	cwz_mst::updateHistogram();
}

void cwz_mst::updateWtoOne(bool _setWtoOne){
	cwz_mst::setWtoOne = _setWtoOne;
	cwz_mst::updateHistogram();
}

void compute_gradient(float*gradient, uchar **gray_image, int h, int w)
{
	float gray,gray_minus,gray_plus;
	int node_idx = 0;
	for(int y=0;y<h;y++)
	{
		gray_minus=gray_image[y][0];
		gray=gray_plus=gray_image[y][1];
		gradient[node_idx]=gray_plus-gray_minus+127.5;

		node_idx++;

		for(int x=1;x<w-1;x++)
		{
			gray_plus=gray_image[y][x+1];
			gradient[node_idx]=0.5*(gray_plus-gray_minus)+127.5;

			gray_minus=gray;
			gray=gray_plus;
			node_idx++;
		}
		
		gradient[node_idx]=gray_plus-gray_minus+127.5;
		node_idx++;
	}
}