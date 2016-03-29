//For inputkey loop control usage
namespace cwz_loop_ctrl{
	//控制matching cost threshold
	static float Match_Cost_Th = 0;
	static const float Match_Cost_Step = 0.5;
	static float Match_Cost_Least_W = 0.01;
	static float Match_Cost_Least_W_Step = 0.01;

	//控制目前使用的方法
	static int M_Key_counter = 0;
	static int       Mode = 0;
	static const int METHOD_TREE = 0;
	static const int METHOD_TREE_NO_REFINE = 1;
	static const int MEDTHO_CV_SGNM = 2;
	static const int M_Key_total = 3;

	//控制濾波
	static bool Do_Guided_Filer = true;
};