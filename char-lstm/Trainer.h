#pragma once
#include <chrono>
#include <codecvt>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <numeric>
#include <vector>
#include "caffe_header.h"

using word = wchar_t;

class Trainer
{
public:
	Trainer(const std::string &solverFile, const std::string &textFile, const std::string &resumeFile, const int sequenceLength, const int batchSize);
	void Update();
	void Predict(std::vector<word> head, int length, float temperature);
	void Save();
private:

	int GetNextPredict(std::vector<float> &data, float temperature);

	int sequenceLength;
	int batchSize;

	boost::shared_ptr<caffe::Solver<float>> solver;
	boost::shared_ptr<caffe::Net<float>> net;

	boost::shared_ptr<caffe::Blob<float>> blobData;
	boost::shared_ptr<caffe::Blob<float>> blobLabel;
	boost::shared_ptr<caffe::Blob<float>> blobClip;
	boost::shared_ptr<caffe::Blob<float>> blobLoss;
	boost::shared_ptr<caffe::Blob<float>> blobOutput;

	std::map<word, int> wordToInt;
	std::vector<word> intToWord;
	std::vector<int> trainIdx;
	std::vector<int> trainClip;

	std::vector<float> clip;
	std::vector<float> labels;
	std::vector<float> data;

	std::mt19937 randomEngine;

};