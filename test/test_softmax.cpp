#include <glog/logging.h>
#include <gtest/gtest.h>
#include "tensor.hpp"
#include "softmax.hpp"

TEST(test_layer, softmax)
{
	LOG(INFO) << "start test softmax\n";
	using namespace my_infer;
	std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
	input->index(0) = -1.f;
	input->index(1) = -2.f;
	input->index(2) = 3.f;

	std::vector<std::shared_ptr<Tensor<float>>> inputs;
	inputs.push_back(input);
	std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
	SoftmaxLayer softmax_layer;
	softmax_layer.Forward(inputs, outputs);

	ASSERT_EQ(outputs.size(), 1);
	float sum = 0;
	for (int i = 0; i < input->size(); ++i)
	{
		sum += std::exp(input->index(i));
	}
	for (int i = 0; i < outputs.size(); ++i)
	{
		ASSERT_EQ(outputs.at(i)->index(0), std::exp(input->index(0)) / sum);
		ASSERT_EQ(outputs.at(i)->index(1), std::exp(input->index(1)) / sum);
		ASSERT_EQ(outputs.at(i)->index(2), std::exp(input->index(2)) / sum);
	}
}