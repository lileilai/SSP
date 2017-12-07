#include "Import.hpp"
#include "DetailedConfig.hpp"
#include "LatentModel.hpp"
#include "OrbitModel.hpp"
#include "Task.hpp"
#include <omp.h>
// 400s for each experiment.
int main(int argc, char* argv[])
{
	srand(time(nullptr));
	//omp_set_num_threads(6);

	Model* model = nullptr;

	/*model = new MFactorSemantics
		(FB15K, General, report_path, semantic_tfile_FB15K, 10, 0.01, 0.1, 0.04, 100);
	model->load("D:\\Temp\\MFactorE.10-0.01-0.1-0.04-100.model");
	((MFactorSemantics*)model)->analyze();
	
	MFactorSemantics& m = *((MFactorSemantics*)model);
	while (true)
	{
		string str_in;
		cout << "Ask:";
		getline(cin, str_in);
		vector<int> re = m.infer_entity(str_in, 10);
		for (auto & elem : re)
		{
			cout << m.tells[elem].substr(0, 120) <<endl;
		}
	}*/
	Dataset FB15k("FB15k", "./data/FB15k/", "train.txt", "valid.txt", "test.txt", false);
	
	/*model=new SemanticModel_Joint(FB15k,LinkPredictionTail,"./report/","./data/FB15k_description/FB15k_mid2description.txt",50,0.001,1.8,0.2,0.1);
	model->run(5000);
	model->test();*/

	model=new SemanticModel(FB15k,LinkPredictionTail,"./report/","./semantic_v1.txt",50,0.001,1.8,0.1);
	model->run(8000);
	model->test();
	delete model;

	/*
	Dataset FB15k("FB15k", "./data/FB15k/", "train.txt", "valid.txt", "test.txt", false);
	model=new TransG(FB15k,LinkPredictionTail,"./report/",400,0.0015,3.0,3,0.1);
	model->run(1000);
	model->test();
	delete model;*/

	/*Dataset FB15k("FB15k", "./data/FB15k/", "train.txt", "valid.txt", "test.txt", false);
	model=new MFactorE(FB15k,LinkPredictionTail,"./report/,",3,0.0004,2.5,0.04,10);
	model->run(2000);
	model->test();
	delete model;*/

	/*Dataset FB15k("FB15k", "./data/FB15k/", "train.txt", "valid.txt", "test.txt", false);
	model=new TransE(FB15k,LinkPredictionTail,"./report/",100,0.001,1);
	model->run(1000);
	model->test();
	delete model;*/

	return 0;
}
