#include "cwz_mst.h"

float cwz_mst::sigma = default_sigma;
bool cwz_mst::setWtoOne = setWto1;
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

	default_root_id = 0;

	this->node_amt = _h * _w;
	this->edge_amt = (_h-1) * _w + h * (_w-1);
	this->channel = _ch;

	this->edge_node_list = new_2d_arr<int>(this->edge_amt, 2);
	this->distance = new short[edge_amt];
	this->cost_sorted_edge_idx = new int[edge_amt];

	this->node_conn_node_list = new_2d_arr<int>(this->node_amt, 4);
	this->node_conn_weights   = new_2d_arr<int>(this->node_amt, 4);
	this->node_conn_node_num  = new_1d_arr(this->node_amt, 0);

	//for segmentation
	this->seg_threshold = defaultSegThreshold;
	this->region_amt_limit = this->node_amt / 10.0;
	this->root_list = new int*[this->region_amt_limit];
	for(int i=0 ; i<this->region_amt_limit; i++) this->root_list[i] = new int[2];
	this->root_list[0][0] = default_root_id;	this->root_list[0][1] = -1;
	this->root_list_count = 1;//�q1�}�l, �]���ܤַ|���@����
	this->can_be_root_node = new bool[this->node_amt];
	memset(this->can_be_root_node, true, sizeof(bool) * this->node_amt);
	//

	this->node_group = new int[this->node_amt];
	for(int i=0 ; i<this->node_amt ; i++){ this->node_group[i] = i; }

	this->id_stack = new int[this->node_amt];
	this->id_stack_e = -1;
	this->node_parent_id = new_1d_arr<int>(this->edge_amt, -1);
	//�̲׭n�ϥΪ� tree ���c
	this->child_node_num = new_1d_arr<int>(this->node_amt, 0);
	this->child_node_list = new_2d_arr<int>( this->node_amt, 4, -1);
	this->node_idx_from_p_to_c= new_1d_arr(this->node_amt, -1);
	this->node_weight = new int[this->node_amt];

	for(int i=0 ; i<IntensityLimit ; i++){ histogram[i] = 0; }

	this->agt_result = new float[ node_amt * max_x_disparity ];
	this->best_disparity = new TEleUnit[node_amt];

	for(int i=0 ; i<IntensityLimit ; i++){
		if(cwz_mst::setWtoOne)
			cwz_mst::whistogram[i] = exp(-double(1) / (cwz_mst::sigma * (IntensityLimit - 1)));
		else
			cwz_mst::whistogram[i] = exp(-double(i) / (cwz_mst::sigma * (IntensityLimit - 1)));
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
//���[�J����(���k��)
	for(y0=0;y0<h;y0++)
	{
		y1=y0;
		for(int x0=0;x0<w-1;x0++)
		{
			x1=x0+1;
			addEdge(img, edge_node_list, distance,  edge_idx++, x0, y0, x1, y1, w, channel);
		}
	}
//�A�[�J����(�W�U��)
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

		if(p0 != p1){//�����I���b�P�@�Ӷ��X��
			//�I�P�I���۳s��
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
	int parent_id = default_root_id;
	int possible_child_id = -1;
	int t_c = 0;//tree node counter

	node_idx_from_p_to_c[ t_c++ ] = parent_id;
	node_weight[parent_id]    = 0;
	node_parent_id[parent_id] = 0;

	do{
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
		parent_id = pop(id_stack, id_stack_e);
	}while( parent_id != -1 );
}
void cwz_mst::seg_kruskal_mst(){
	for(int i=0 ; i<edge_amt ; i++){
		int edge_idx = cost_sorted_edge_idx[i];

		int n0 = edge_node_list[ edge_idx ][0];
		int n1 = edge_node_list[ edge_idx ][1];
		
		int p0 = findset(n0);
		int p1 = findset(n1);

		if(p0 != p1){//�����I���b�P�@�Ӷ��X��
			//�I�P�I���۳s��
			if(distance[edge_idx] < seg_threshold){
				node_conn_node_list[n0][ node_conn_node_num[n0] ] = n1;
				node_conn_node_list[n1][ node_conn_node_num[n1] ] = n0;

				node_conn_weights[n0][ node_conn_node_num[n0] ] = distance[edge_idx];
				node_conn_weights[n1][ node_conn_node_num[n1] ] = distance[edge_idx];

				node_conn_node_num[n0]++;
				node_conn_node_num[n1]++;
			}else{
				this->root_list[ this->root_list_count ][0] = n0;
				this->root_list[ this->root_list_count ][1] = n1;
				this->root_list_count++;
			}

			node_group[p0] = p1;
		}
	}
}
void cwz_mst::seg_build_tree(){
	int parent_id = default_root_id;
	int possible_child_id = -1;
	int t_c = 0;//tree node counter

	for(int r_i=0 ; r_i < this->root_list_count ; r_i++){
		int candidate_r_id = this->root_list[r_i][0];
		if( this->can_be_root_node[candidate_r_id] ){
			parent_id      = candidate_r_id;
			candidate_r_id = this->root_list[r_i][1];
		}else{ 
			parent_id = this->root_list[r_i][1];//�p�G��1���I�����root, ��2���I�@�w�i�H, �_�h��ܺt��k�����D
		
			//debug use
			if( !this->can_be_root_node[parent_id] ){
				printf("Both node is not able to be root, algorithm may not work, please terminate the program and debug it.\n");
				system("PAUSE");
			}//
		}

		this->root_list[r_i][0] = t_c;         //�אּ����root�`�I�bnode_idx_from_p_to_c[]����index
		node_idx_from_p_to_c[ t_c ] = parent_id;
		node_weight[parent_id]    =  0;
		node_parent_id[parent_id] = -1;
		t_c++;

		do{
			for(int edge_c=0 ; edge_c < node_conn_node_num[parent_id] ; edge_c++){
				possible_child_id = node_conn_node_list[parent_id][edge_c];
				if( node_parent_id[possible_child_id] != -1 ){
					continue;//its the parent, pass
				}
				node_idx_from_p_to_c[ t_c++ ] = possible_child_id;
				node_weight[possible_child_id] = node_conn_weights[parent_id][edge_c];
				node_parent_id[possible_child_id] = parent_id;

				child_node_list[parent_id][ child_node_num[parent_id]++ ] = possible_child_id;
				this->can_be_root_node[possible_child_id] = false;//once become ones child, it can't be root candidate anymore

				if( node_conn_node_num[possible_child_id] > 1)//definitely has one edge connect to parent(num eq 1 is a leaf node)
					push(possible_child_id, id_stack, id_stack_e);
			}
			parent_id = pop(id_stack, id_stack_e);
		}while( parent_id != -1 );
	}

	if(t_c == this->node_amt){
		printf("t_c == this->node_amt\n");
	}else{
		printf("t_c != this->node_amt, algorithm error.\n");
		system("PAUSE");
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
TEleUnit *cwz_mst::pick_best_dispairty(){
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
	if( this->isInit && this->hasImg ){
		double total_b_edge = 0;
		double total_c_sort = 0;
		double total_mst = 0;
		double total_b_tree = 0;

		int test_amt = 10;
		for(int i=0; i<test_amt; i++){
			if(i>0)
				reinit();
			cwz_timer::start();
			this->build_edges();
			total_b_edge += cwz_timer::return_time();
			cwz_timer::start();
			this->counting_sort();
			total_c_sort += cwz_timer::return_time();
			cwz_timer::start();
			this->kruskal_mst();
			total_mst += cwz_timer::return_time();
			cwz_timer::start();
			this->build_tree();
			total_b_tree += cwz_timer::return_time();
		}
		printf("redo for %d times and has averaged.\n", test_amt);
		printf("build_edges  : %5.5fs\n"  , total_b_edge/test_amt);
		printf("counting_sort: %5.5fs\n", total_c_sort/test_amt);
		printf("kruskal_mst  : %5.5fs\n"  , total_mst   /test_amt);
		printf("build_tree   : %5.5fs\n"   , total_b_tree/test_amt);
	}
	printf("--	endof mst profiling	--\n");
	system("PAUSE");
}
//for reuse
void cwz_mst::reinit(){
	memset(this->histogram, 0, sizeof(int) * IntensityLimit);
	for(int i=0 ; i<this->node_amt ; i++){ this->node_group[i] = i; }
	memset(this->node_conn_node_num, 0, sizeof(int) * this->node_amt);
	//for segmentation
	this->root_list[0][0] = default_root_id;	this->root_list[0][1] = -1;
	this->root_list_count = 1;
	memset(this->can_be_root_node, true, sizeof(bool) * this->node_amt);
	//
	this->id_stack_e = -1;
	memset(this->child_node_num, 0, sizeof(int) * this->node_amt); 
	memset(this->node_parent_id, -1, sizeof(int) * this->node_amt); 
}

void cwz_mst::updateSigma(float _sigma){
	cwz_mst::sigma = _sigma;
	for(int i=0 ; i<IntensityLimit ; i++){
		if(cwz_mst::setWtoOne)
			cwz_mst::whistogram[i] = exp(-double(1) / (cwz_mst::sigma * (IntensityLimit - 1)));
		else
			cwz_mst::whistogram[i] = exp(-double(i) / (cwz_mst::sigma * (IntensityLimit - 1)));
	}
}

void cwz_mst::updateWtoOne(bool _setWtoOne){
	cwz_mst::setWtoOne = _setWtoOne;
	for(int i=0 ; i<IntensityLimit ; i++){
		if(cwz_mst::setWtoOne)
			cwz_mst::whistogram[i] = exp(-double(1) / (cwz_mst::sigma * (IntensityLimit - 1)));
		else
			cwz_mst::whistogram[i] = exp(-double(i) / (cwz_mst::sigma * (IntensityLimit - 1)));
	}
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