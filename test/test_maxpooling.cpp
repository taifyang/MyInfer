#include <glog/logging.h>
#include <gtest/gtest.h>
#include "tensor.hpp"
#include "maxpooling.hpp"

TEST(test_layer, maxpooling)
{
	LOG(INFO) << "start test maxpooling\n";
	using namespace my_infer;
	uint32_t pooling_h = 2;
	uint32_t pooling_w = 2;
	uint32_t padding_h = 0;
	uint32_t padding_w = 0;
	uint32_t stride_h = 1;
	uint32_t stride_w = 1;

	Eigen::Tensor<float, 3> input_data(3, 3, 2); //d0 d1 d2
	input_data.setValues(
		{
			{{1, 1},{2, 2},{3, 3}},
			{{4, 4},{5, 5},{6, 6}},
			{{7, 7},{8, 8},{9, 9}},
		});

	std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(2, 3, 3); //d2 d0 d1
	input->data() = input_data;
	input->Show();

	std::vector<std::shared_ptr<Tensor<float>>> inputs;
	std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
	inputs.push_back(input);

	MaxPoolingLayer maxpooling_layer(padding_h, padding_w, pooling_h, pooling_w, stride_h, stride_w);
	maxpooling_layer.Forward(inputs, outputs);
	ASSERT_EQ(outputs.size(), 1);
	auto output = outputs.at(0);
	output->Show();

	ASSERT_EQ(output->rows(), 2);
	ASSERT_EQ(output->cols(), 2);

	ASSERT_EQ(output->at(0, 0, 0), 5);
	ASSERT_EQ(output->at(0, 0, 1), 6);
	ASSERT_EQ(output->at(0, 1, 0), 8);
	ASSERT_EQ(output->at(0, 1, 1), 9);

	ASSERT_EQ(output->at(1, 0, 0), 5);
	ASSERT_EQ(output->at(1, 0, 1), 6);
	ASSERT_EQ(output->at(1, 1, 0), 8);
	ASSERT_EQ(output->at(1, 1, 1), 9);
}