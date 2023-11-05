#include <glog/logging.h>
#include <gtest/gtest.h>
#include "tensor.hpp"
#include "avgpooling.hpp"

TEST(test_layer, avgpooling)
{
	LOG(INFO) << "start test avgpooling\n";
	using namespace my_infer;

	uint32_t output_h = 2;
	uint32_t output_w = 2;

	Eigen::Tensor<float, 3> input_data(4, 4, 2);
	input_data.setValues(
		{
			{{1, 1}, {2, 2}, {3, 3}, {4, 4} },
			{{5, 5}, {6, 6}, {7, 7}, {8, 8}},
			{{9, 9}, {10, 10}, {11, 11}, {12, 12}},
			{{13, 13}, {14, 14}, {15, 15}, {16, 16}},
		});

	std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(2, 4, 4);
	input->data() = input_data;
	input->Show();

	std::vector<std::shared_ptr<Tensor<float>>> inputs;
	std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
	inputs.push_back(input);

	AvgPoolingLayer avgpooling_layer(output_h, output_w);
	avgpooling_layer.Forward(inputs, outputs);
	ASSERT_EQ(outputs.size(), 1);
	auto output = outputs.at(0);
	output->Show();

	ASSERT_EQ(output->rows(), 2);
	ASSERT_EQ(output->cols(), 2);
	ASSERT_EQ(output->at(0, 0, 0), 3.5);
	ASSERT_EQ(output->at(0, 0, 1), 5.5);
	ASSERT_EQ(output->at(0, 1, 0), 11.5);
	ASSERT_EQ(output->at(0, 1, 1), 13.5);
	ASSERT_EQ(output->at(1, 0, 0), 3.5);
	ASSERT_EQ(output->at(1, 0, 1), 5.5);
	ASSERT_EQ(output->at(1, 1, 0), 11.5);
	ASSERT_EQ(output->at(1, 1, 1), 13.5);
}