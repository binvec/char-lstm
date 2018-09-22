#include "Trainer.h"

Trainer::Trainer(const std::string &solverFile, const std::string &textFile, 
	const std::string &resumeFile, const int sequenceLength, const int batchSize)
{
	this->sequenceLength = sequenceLength;
	this->batchSize = batchSize;

	std::ifstream file(textFile, std::ios::in);
	std::string input_str(std::istreambuf_iterator<char>(file.rdbuf()),
		std::istreambuf_iterator<char>());
	file.close();
	std::wstring_convert< std::codecvt_utf8<word> > conv;
	
	std::wstring input_wstr = conv.from_bytes(input_str);

	int currentIdx = 0;
	for (auto &wd : input_wstr)
	{
		int idx;
		if (wordToInt.count(wd) == 0)
		{
			idx = currentIdx++;
			intToWord.push_back(wd);
			wordToInt[wd] = idx;
		}
		else
		{ 
			idx = wordToInt[wd];
		}
		trainIdx.push_back(idx);
	}
	trainClip = std::vector<int>(trainIdx.size());
	for (int i = 0; i < trainIdx.size(); ++i)
	{
		if (i == 0)
		{
			trainClip[i] = 0;
		}
		else
		{
			if (trainIdx[i - 1] == wordToInt[L'\n'])
			{
				trainClip[i] = 0;
			}
			else
			{
				trainClip[i] = 1;
			}
		}
	}

	caffe::SolverParameter solver_param;
	caffe::ReadProtoFromTextFileOrDie(solverFile, &solver_param);
	solver.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param));
	net = solver->net();

	if (!resumeFile.empty())
	{
		solver->Restore(resumeFile.c_str());
	}

	blobData = net->blob_by_name("data");
	blobClip = net->blob_by_name("clip");
	blobLabel = net->blob_by_name("label");
	blobLoss = net->blob_by_name("loss");
	blobOutput = net->blob_by_name("ip1");

	clip = std::vector<float>(sequenceLength * batchSize);
	labels = std::vector<float>(sequenceLength * batchSize);
	data = std::vector<float>(sequenceLength * batchSize);
	randomEngine = std::mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

void Trainer::Update()
{
	blobData->Reshape(std::vector<int>{sequenceLength, batchSize});
	blobClip->Reshape(std::vector<int>{sequenceLength, batchSize});
	blobLabel->Reshape(std::vector<int>{sequenceLength * batchSize, 1});
	net->Reshape();
	for (int i = 0; i < batchSize; ++i)
	{
		int currentIdx = std::uniform_int_distribution<int>(0, trainIdx.size() - sequenceLength - 2)(randomEngine);
		for (int j = 0; j < sequenceLength; ++j)
		{
			data[batchSize * j + i] = trainIdx[currentIdx + j];
			labels[batchSize * j + i] = trainIdx[currentIdx + j + 1];
			clip[batchSize * j + i] = trainClip[currentIdx + j];
		}
	}
	blobData->set_cpu_data(data.data());
	blobLabel->set_cpu_data(labels.data());
	blobClip->set_cpu_data(clip.data());
	solver->Step(1);
}

void Trainer::Predict(std::vector<word> head, int length, float temperature)
{
	blobData->Reshape(std::vector<int>{sequenceLength, 1});
	blobClip->Reshape(std::vector<int>{sequenceLength, 1});
	blobLabel->Reshape(std::vector<int>{sequenceLength, 1});

	for (auto &h : head)
	{
		std::vector<float> test_clip(sequenceLength, 0.0f);
		std::deque<int> inputIdx;
		for (int i = 0; i < sequenceLength - 1; ++i)
		{
			inputIdx.push_back(0);
		}
		inputIdx.push_back(wordToInt[h]);
		std::wcout << h;
		for (int i = 0; i < length - 1; ++i)
		{
			std::vector<float> inputVector(sequenceLength);
			for (int j = 0; j < sequenceLength; ++j)
			{
				inputVector[j] = (float)inputIdx[j];
			}
			blobData->set_cpu_data(inputVector.data());
			blobClip->set_cpu_data(test_clip.data());
			net->Forward();
			const float* data = net->blob_by_name("ip1")->cpu_data();
			int offset = (sequenceLength - 1) * intToWord.size();
			std::vector<float> vectorData(data + offset, data + offset + intToWord.size());
			int pred = GetNextPredict(vectorData, temperature);
			std::wcout << intToWord[pred];
			inputIdx.push_back(pred);
			inputIdx.pop_front();
			for (int j = 0; j < sequenceLength - 1; j++)
			{
				test_clip[j] = test_clip[j + 1];
			}
			test_clip.back() = 1.0f;
			for (int j = 0; j < sequenceLength - 1; j++)
			{
				if (pred == wordToInt[L'\n'])
				{
					test_clip[j + 1] = 0.0f;
				}
			}
		}
		std::wcout << std::endl;
	}
}

void Trainer::Save()
{
	solver->Snapshot();
}

int Trainer::GetNextPredict(std::vector<float>& data, float temperature)
{
	std::vector<float> proba(data.size()), accumulatedProba(data.size());
	auto maxValue = std::max_element(data.begin(), data.end());
	for (int i = 0; i < data.size(); ++i)
	{
		proba[i] = exp((data[i] - *maxValue) / temperature);
	}
	float expoSum = std::accumulate(proba.begin(), proba.end(), 0.0f);
	proba[0] /= expoSum;
	accumulatedProba[0] = proba[0];
	float randomNumber = std::uniform_real_distribution<float>(0.0f, 1.0f)(randomEngine);
	for (int i = 1; i < proba.size(); ++i)
	{
		if (accumulatedProba[i - 1] > randomNumber)
		{
			return i - 1;
		}
		proba[i] /= expoSum;
		accumulatedProba[i] = accumulatedProba[i - 1] + proba[i];
	}
	return proba.size() - 1;
}
