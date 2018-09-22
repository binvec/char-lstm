#include "Trainer.h"
#include <windows.h>

int IsKeyDown(const int id)
{
	return GetAsyncKeyState(id) & 0x8000 ? 1 : 0;
}

int main()
{
	google::InitGoogleLogging("");
	google::SetCommandLineOption("GLOG_minloglevel", "2");

	std::wcout.imbue(std::locale("chs"));
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
	Trainer trainer("solver.prototxt", "poetry.txt", "char-lstm_iter_102277.solverstate", 32, 32);
	//while (true)
	//{
	//	trainer.Update();
	//	if (IsKeyDown('Q'))
	//	{
	//		trainer.Save();
	//		break;
	//	}
	//}
	trainer.Predict(std::vector<word>{L'社', L'会', L'主', L'义', L'好'}, 13, 0.9);
	trainer.Predict(std::vector<word>{L'飞', L'龙', L'骑', L'脸', L'怎', L'么', L'输'}, 13, 0.9);
	system("PAUSE");
}