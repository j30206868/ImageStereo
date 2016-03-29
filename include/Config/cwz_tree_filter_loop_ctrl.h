#ifndef CWZ_TREE_FILTER_LOOP_CTRL_H
#define CWZ_TREE_FILTER_LOOP_CTRL_H

#include "common_func.h"
#include "TreeFilter/cwz_disparity_generation.h"

class cwz_cmd_processor{
private:
	std::string commandstr;
	int *frame_count;
	
public:
	cwz_cmd_processor(int *_frame_count);
	void showRule();
	bool readTreeLoopCommandStr();
};

cwz_cmd_processor::cwz_cmd_processor(int *_frame_count){
	this->frame_count = _frame_count;
}
void cwz_cmd_processor::showRule(){
	printf("Modify parameter command rule:\n");
	std::cout << "s=0~1" << "; sigma value by now:" << cwz_mst::sigma << std::endl;
	const char *wtoonestate = cwz_mst::setWtoOne?"True":"False";
	std::cout << "wto1=0 or 1" << "; setWtoOne value by now:" << wtoonestate << std::endl;
	std::cout << "img=number" << "; jump to specified image number" << std::endl;
}
bool cwz_cmd_processor::readTreeLoopCommandStr(){
	std::getline(std::cin, this->commandstr);

	const int del_num = 2;
	std::string *deliminators = new std::string[del_num];
	deliminators[0] = ";";//分隔各項指令
	deliminators[1] = "=";//分隔參數名稱與值

	int cmd_arr_length = 0;
	std::string *blocks = splitInstructions(this->commandstr, deliminators, del_num, cmd_arr_length);

	if(cmd_arr_length == 0){ return true; }
	if(cmd_arr_length % 2 != 0){ printf("cwz_cmd_processor::readTreeLoopCommandStr() Error: cmd_arr_length % 2 != 0;\n"); system("PAUSE"); return true; }

	// do everything that need to be done by following the command
	for(int i=0 ; i<cmd_arr_length ; i+=2){
		//blocks[i] = 參數 ; blocks[i+1] = 值
		if(blocks[i] == "s"){
			float new_sigma = atof(blocks[i+1].c_str());
			cwz_mst::updateSigma( new_sigma );
			printf("update cwz_mst::sigma to %f\n", new_sigma);
		}else if(blocks[i] == "wto1"){
			bool new_setwtoone = atoi(blocks[i+1].c_str());
			cwz_mst::updateWtoOne( new_setwtoone );
			printf("update cwz_mst::WtoOne to %s\n", new_setwtoone?"True":"False" );
		}else if(blocks[i] == "img"){
			*this->frame_count = std::atoi(blocks[i+1].c_str());
			printf("Jump to image number %d\n", *this->frame_count);
		}else{
			printf("cwz_cmd_processor::readTreeLoopCommandStr() Error: variable name '%s' can't be recognized.\n", blocks[i]); 
			system("PAUSE");
		}
	}
	return false;
}

int processInputKey(int inputkey, int &status, int &frame_count){
	//will return control code for caller outside which maybe inside a while loop
	//As the return code state, outside loop should follow the instruction 'continue' or 'break'
	enum{ result_nothing = 0, result_continue = 1, result_break = 2};
	do{
		if(inputkey == 'e'){
			status = cwz_loop_ctrl::CV_IMG_STATUS_EXIT;
		}else if(inputkey == 's' || inputkey == 'f'){
			status = cwz_loop_ctrl::CV_IMG_STATUS_FRAME_BY_FRAME;
		}else if(inputkey == 'p'){
			status = cwz_loop_ctrl::CV_IMG_STATUS_MODIFY_PARAM;
			frame_count--;
		}else if(inputkey == ','){//懶的+shift所以直接用跟<同格的,
			if(frame_count != 1)
				frame_count-=2;
			else//避免讀到index為00或負的檔案結果爆掉
				frame_count = 0;
			status = cwz_loop_ctrl::CV_IMG_STATUS_FRAME_BY_FRAME;
			break;
		}else if(inputkey == '.'){//懶的+shift所以直接用跟>同格的.
			status = cwz_loop_ctrl::CV_IMG_STATUS_FRAME_BY_FRAME;
			break;
		}else if(inputkey == 'k'){
			status = cwz_loop_ctrl::CV_IMG_STATUS_KEEPGOING;
		}else if(inputkey == 'm'){
			cwz_loop_ctrl::M_Key_counter++;
			int mode_v = cwz_loop_ctrl::M_Key_counter % cwz_loop_ctrl::M_Key_total;
			if(mode_v == cwz_loop_ctrl::MEDTHO_CV_SGNM){
				cwz_loop_ctrl::Mode = cwz_loop_ctrl::MEDTHO_CV_SGNM;
			}else if(mode_v == cwz_loop_ctrl::METHOD_TREE){
				cwz_loop_ctrl::Mode = cwz_loop_ctrl::METHOD_TREE;
			}else if(mode_v == cwz_loop_ctrl::METHOD_TREE_NO_REFINE){
				cwz_loop_ctrl::Mode = cwz_loop_ctrl::METHOD_TREE_NO_REFINE;
			}
			frame_count--;
			break;
		}else if(inputkey == 't'){
			cwz_loop_ctrl::Match_Cost_Th = abs(cwz_loop_ctrl::Match_Cost_Th - cwz_loop_ctrl::Match_Cost_Step);
			frame_count--;
			break;
		}else if(inputkey == 'T'){
			cwz_loop_ctrl::Match_Cost_Th += cwz_loop_ctrl::Match_Cost_Step;
			frame_count--;
			break;
		}else if(inputkey == 'l'){
			cwz_loop_ctrl::Match_Cost_Least_W = abs(cwz_loop_ctrl::Match_Cost_Least_W - cwz_loop_ctrl::Match_Cost_Least_W_Step);
			frame_count--;
			break;
		}else if(inputkey == 'L'){
			cwz_loop_ctrl::Match_Cost_Least_W += cwz_loop_ctrl::Match_Cost_Least_W_Step;
			frame_count--;
			break;
		}else if(inputkey == 'w'){
			cwz_mst::upbound = abs(cwz_mst::upbound - 0.01);
			frame_count--;
			break;
		}else if(inputkey == 'W'){
			cwz_mst::upbound = abs(cwz_mst::upbound + 0.01);
			frame_count--;
			break;
		}else if(inputkey == 'g'){
			if(cwz_loop_ctrl::Do_Guided_Filer)
				cwz_loop_ctrl::Do_Guided_Filer = false;
			else
				cwz_loop_ctrl::Do_Guided_Filer = true;
			frame_count--;
			break;
		}

		inputkey = -1;
		if(status == cwz_loop_ctrl::CV_IMG_STATUS_MODIFY_PARAM){
			cwz_cmd_processor cmd_proc(&frame_count);
			cmd_proc.showRule();
			bool isEnd = cmd_proc.readTreeLoopCommandStr();
			isEnd = true;
			if(isEnd){ status = cwz_loop_ctrl::CV_IMG_STATUS_FRAME_BY_FRAME; }
		}
		else if(status == cwz_loop_ctrl::CV_IMG_STATUS_FRAME_BY_FRAME){	inputkey = cv::waitKey(0);  	}
	}while(inputkey != -1);
	if(status == cwz_loop_ctrl::CV_IMG_STATUS_EXIT){ return result_break; }
	//else if(status == cwz_loop_ctrl::CV_IMG_STATUS_MODIFY_PARAM){
	//	return result_continue;
	//}
	return result_nothing;
}
#endif