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
	trainer.Predict(std::vector<word>{L'��', L'��', L'��', L'��', L'��'}, 13, 0.9);
	trainer.Predict(std::vector<word>{L'��', L'��', L'��', L'��', L'��', L'ô', L'��'}, 13, 0.9);
	system("PAUSE");
}