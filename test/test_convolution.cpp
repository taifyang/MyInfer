#include <gtest/gtest.h>
#include <glog/logging.h>
#include "convolution.hpp"

 //单卷积单通道
TEST(test_layer, conv1)
{
	LOG(INFO) << "start test conv1\n";
	using namespace my_infer;

	uint32_t output_channel = 1;
	uint32_t in_channel = 1;
	uint32_t kernel_h = 3;
	uint32_t kernel_w = 3;
	uint32_t padding_h = 0;
	uint32_t padding_w = 0;
	uint32_t stride_h = 1;
	uint32_t stride_w = 1;
	uint32_t groups = 1;
	bool use_bias = false;

	// 单个卷积核的情况
	std::vector<float> weights;
	for (int i = 0; i < 3; ++i) 
	{
		weights.push_back(float(i + 1));
		weights.push_back(float(i + 1));
		weights.push_back(float(i + 1));
	}

	std::vector<std::shared_ptr<ftensor>> inputs;
	Eigen::Tensor<float, 3> input_data(4, 4, 1);
	input_data.setValues(
		{
			{{1}, {2}, {3}, {4}},
			{{5}, {6}, {7}, {8}},
			{{9}, {10}, {11}, {12}},
			{{13}, {14}, {15}, {16}},
		});
	std::shared_ptr<ftensor> input = std::make_shared<ftensor>(1, 4, 4);
	input->set_data(input_data);
	LOG(INFO) << "input:";
	input->Show();

	ConvolutionLayer convolution_layer(output_channel, in_channel, kernel_h, kernel_w, padding_h, padding_w, stride_h, stride_w, groups, use_bias);
	convolution_layer.set_weights(weights);
	inputs.push_back(input);
	std::vector<std::shared_ptr<ftensor>> outputs(1);

	convolution_layer.Forward(inputs, outputs);
	LOG(INFO) << "result: ";
	for (int i = 0; i < outputs.size(); ++i) 
	{
		outputs.at(i)->Show();
	}
}

TEST(test_layer, conv2)
{
	LOG(INFO) << "start test conv2\n";
	using namespace my_infer;

	uint32_t output_channel = 1;
	uint32_t in_channel = 1;
	uint32_t kernel_h = 3;
	uint32_t kernel_w = 3;
	uint32_t padding_h = 0;
	uint32_t padding_w = 0;
	uint32_t stride_h = 1;
	uint32_t stride_w = 1;
	uint32_t groups = 1;
	bool use_bias = false;

	// 单个卷积核的情况
	std::vector<float> weights;
	for (int i = 0; i < 9; i++)
	{
		weights.push_back(float(i + 1));
	}

	std::vector<std::shared_ptr<ftensor>> inputs;
	Eigen::Tensor<float, 3> input_data(4, 4, 1);
	input_data.setValues(
		{
			{{1}, {2}, {3}, {4}},
			{{5}, {6}, {7}, {8}},
			{{9}, {10}, {11}, {12}},
			{{13}, {14}, {15}, {16}},
		});
	std::shared_ptr<ftensor> input = std::make_shared<ftensor>(1, 4, 4);
	input->set_data(input_data);
	LOG(INFO) << "input:";
	input->Show();

	ConvolutionLayer convolution_layer(output_channel, in_channel, kernel_h, kernel_w, padding_h, padding_w, stride_h, stride_w, groups, use_bias);
	convolution_layer.set_weights(weights);
	inputs.push_back(input);
	std::vector<std::shared_ptr<ftensor>> outputs(1);

	convolution_layer.Forward(inputs, outputs);
	LOG(INFO) << "result: ";
	for (int i = 0; i < outputs.size(); ++i)
	{
		outputs.at(i)->Show();
	}
}

// 多卷积多通道
TEST(test_layer, conv3) 
{
	LOG(INFO) << "start test conv3\n";
	using namespace my_infer;

	uint32_t output_channel = 3;
	uint32_t in_channel = 3;
	uint32_t kernel_h = 3;
	uint32_t kernel_w = 3;
	uint32_t padding_h = 0;
	uint32_t padding_w = 0;
	uint32_t stride_h = 1;
	uint32_t stride_w = 1;
	uint32_t groups = 1;
	bool use_bias = false;

	// 多个卷积核的情况
	std::vector<float> weights;
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < 3; ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				weights.push_back(float(j + 1));
				weights.push_back(float(j + 1));
				weights.push_back(float(j + 1));
			}
		}
	}

	std::vector<std::shared_ptr<ftensor>> inputs;
	Eigen::Tensor<float, 3> input_data(4, 4, 3);
	input_data.setValues(
		{
			{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}},
			{{5, 5, 5}, {6, 6, 6}, {7, 7, 7}, {8, 8, 8}},
			{{9, 9, 9}, {10, 10, 10}, {11, 11, 11}, {12, 12, 12} },
			{{13, 13, 13}, {14, 14, 14}, {15, 15, 15}, {16, 16, 16}},
		});
	std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 4, 4);
	input->set_data(input_data);

	LOG(INFO) << "input:";
	input->Show();

	ConvolutionLayer convolution_layer(output_channel, in_channel, kernel_h, kernel_w, padding_h, padding_w, stride_h, stride_w, groups, use_bias);
	convolution_layer.set_weights(weights);
	inputs.push_back(input);
	std::vector<std::shared_ptr<ftensor>> outputs(1);

	convolution_layer.Forward(inputs, outputs);
	LOG(INFO) << "result: ";
	for (int i = 0; i < outputs.size(); ++i) 
	{
		outputs.at(i)->Show();
	}
}

// 多卷积多通道
TEST(test_layer, conv4)
{
	LOG(INFO) << "start test conv4\n";
	using namespace my_infer;

	uint32_t output_channel = 3;
	uint32_t in_channel = 3;
	uint32_t kernel_h = 3;
	uint32_t kernel_w = 3;
	uint32_t padding_h = 0;
	uint32_t padding_w = 0;
	uint32_t stride_h = 1;
	uint32_t stride_w = 1;
	uint32_t groups = 1;
	bool use_bias = false;

	// 多个卷积核的情况
	std::vector<float> weights;
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < 27; ++i)
		{
			weights.push_back(float(c + i + 1));
		}
	}

	std::vector<std::shared_ptr<ftensor>> inputs;
	Eigen::Tensor<float, 3> input_data(4, 4, 3);
	input_data.setValues(
		{
			{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}},
			{{5, 5, 5}, {6, 6, 6}, {7, 7, 7}, {8, 8, 8}},
			{{9, 9, 9}, {10, 10, 10}, {11, 11, 11}, {12, 12, 12} },
			{{13, 13, 13}, {14, 14, 14}, {15, 15, 15}, {16, 16, 16}},
		});
	std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 4, 4);
	input->set_data(input_data);

	LOG(INFO) << "input:";
	input->Show();

	ConvolutionLayer convolution_layer(output_channel, in_channel, kernel_h, kernel_w, padding_h, padding_w, stride_h, stride_w, groups, use_bias);
	convolution_layer.set_weights(weights);
	inputs.push_back(input);
	std::vector<std::shared_ptr<ftensor>> outputs(1);

	convolution_layer.Forward(inputs, outputs);
	LOG(INFO) << "result: ";
	for (int i = 0; i < outputs.size(); ++i)
	{
		outputs.at(i)->Show();
	}
}