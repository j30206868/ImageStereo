#ifndef CWZ_TREE_FILTER_LOOP_CTRL_H
#define CWZ_TREE_FILTER_LOOP_CTRL_H

#include "common_func.h"
#include "cwz_disparity_generation.h"



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
	deliminators[0] = ";";//���j�U�����O
	deliminators[1] = "=";//���j�ѼƦW�ٻP��

	int cmd_arr_length = 0;
	std::string *blocks = splitInstructions(this->commandstr, deliminators, del_num, cmd_arr_length);

	if(cmd_arr_length == 0){ return true; }
	if(cmd_arr_length % 2 != 0){ printf("cwz_cmd_processor::readTreeLoopCommandStr() Error: cmd_arr_length % 2 != 0;\n"); system("PAUSE"); return true; }

	// do everything that need to be done by following the command
	for(int i=0 ; i<cmd_arr_length ; i+=2){
		//blocks[i] = �Ѽ� ; blocks[i+1] = ��
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

#endif