#include <gtest/gtest.h>
#include <glog/logging.h>
#include "tensor.hpp"
#include "tensor_utils.hpp"
#include "cat.hpp"

TEST(test_layer, cat1) 
{
	using namespace my_infer;
	int input_size = 4;
	int output_size = 2;
	int input_channels = 6;
	int output_channels = 12;

	std::vector<std::shared_ptr<Tensor<float>>> inputs;
	for (int i = 0; i < input_size; ++i) 
	{
		std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(input_channels, 3, 3); 
		input->Rand();
		inputs.push_back(input); //4*6*32*32
	}
	std::vector<std::shared_ptr<Tensor<float>>> outputs(output_size); //2*12*
	CatLayer cat_layer(1);
	cat_layer.Forward(inputs, outputs);

	for (uint32_t i = 0; i < outputs.size(); ++i) 
	{
		ASSERT_EQ(outputs.at(i)->channels(), output_channels);
	}

	for (int i = 0; i < input_size / 2; ++i) 
	{
		for (int input_channel = 0; input_channel < input_channels; ++input_channel) 
		{
			//std::cout << inputs.at(i)->slice(input_channel) << std::endl;
			//std::cout << outputs.at(i)->slice(input_channel) << std::endl;
			Eigen::Tensor<float, 0> diff =(inputs.at(i)->slice(input_channel)- outputs.at(i)->slice(input_channel)).abs().maximum();
			EXPECT_LE((float)diff(0), 0.01f);
		}
	}

	for (int i = input_size / 2; i < input_size; ++i) 
	{
		for (int input_channel = input_channels; input_channel < input_channels * 2; ++input_channel)
		{
			Eigen::Tensor<float, 0> diff = (inputs.at(i)->slice(input_channel - input_channels) - outputs.at(i - output_size)->slice(input_channel)).abs().maximum();
			EXPECT_LE((float)diff(0), 0.01f);
		}
	}
}