#include <glog/logging.h>
#include <gtest/gtest.h>
#include "linear.hpp"

TEST(test_layer, forward_linear1)
{
	using namespace my_infer;
	const uint32_t in_features = 2;
	const uint32_t out_features = 3;
	const uint32_t in_dims = 4;

	LinearLayer linear_layer(in_features, out_features, false);
	std::vector<float> weights(in_features * out_features, 1.f);										//2*3
	linear_layer.set_weights(weights);

	std::vector<std::shared_ptr<Tensor<float>>> inputs;
	std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, in_features, in_dims);	//2*4
	input->Fill(1.f);
	inputs.push_back(input);

	std::vector<std::shared_ptr<Tensor<float>>> outputs;
	std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, out_features, in_dims);	//3*4
	outputs.push_back(output);

	const auto status = linear_layer.Forward(inputs, outputs);
	ASSERT_EQ(outputs.size(), 1);
	const auto& output_tensor = outputs.at(0);
	for (int i = 0; i < output_tensor->size(); ++i)
	{
		ASSERT_EQ(output_tensor->index(i), in_features);
	}
}

TEST(test_layer, forward_linear2) 
{
	using namespace my_infer;
	const uint32_t in_features = 2;
	const uint32_t out_features = 3;
	const uint32_t in_dims = 4;
	LinearLayer linear_layer(in_features, out_features, false);

	std::vector<float> weights;
	for (int i = 0; i < out_features; ++i)
	{
		for (int j = 0; j < in_features; ++j)
		{
			weights.push_back(j + 1);
		}
	}
	linear_layer.set_weights(weights);

	std::vector<std::shared_ptr<Tensor<float>>> inputs;
	std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, in_features, in_dims);
	input->Fill(1.f);
	inputs.push_back(input);

	std::vector<std::shared_ptr<Tensor<float>>> outputs;
	std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, out_features, in_dims);
	outputs.push_back(output);

	linear_layer.Forward(inputs, outputs);
	ASSERT_EQ(outputs.size(), 1);
	const auto& output_tensor = outputs.at(0);
	for (int i = 0; i < output_tensor->size(); ++i)
	{
		ASSERT_EQ(output_tensor->index(i), 3);
	}
}

TEST(test_layer, forward_linear3)
{
	using namespace my_infer;
	const uint32_t in_features = 2;
	const uint32_t out_features = 3;
	const uint32_t in_dims = 4;
	LinearLayer linear_layer(in_features, out_features, true);

	std::vector<float> weights;
	for (int i = 0; i < out_features; ++i)
	{
		for (int j = 0; j < in_features; ++j)
		{
			weights.push_back(j + 1);
		}
	}
	linear_layer.set_weights(weights);

	std::vector<float> bias;
	for (int i = 0; i < out_features; ++i)
	{
		bias.push_back(i);
	}
	linear_layer.set_bias(bias);

	std::vector<std::shared_ptr<Tensor<float>>> inputs;
	std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, in_features, in_dims);
	input->Fill(1.f);
	inputs.push_back(input);

	std::vector<std::shared_ptr<Tensor<float>>> outputs;
	std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, out_features, in_dims);
	outputs.push_back(output);

	linear_layer.Forward(inputs, outputs);
	ASSERT_EQ(outputs.size(), 1);
	const auto& output_tensor = outputs.at(0);
	for (int i = 0; i < in_dims; ++i)
	{
		for (int j = 0; j < out_features; ++j)
		{
			ASSERT_EQ(output_tensor->index(j), 3 + j);
		}
	}
}