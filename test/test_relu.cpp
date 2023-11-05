#include <glog/logging.h>
#include <gtest/gtest.h>
#include "tensor.hpp"
#include "relu.hpp"

TEST(test_layer, relu1) 
{
	LOG(INFO) << "start test relu1\n";
	using namespace my_infer;
	std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
	input->index(0) = -1.f;
	input->index(1) = -2.f;
	input->index(2) = 3.f;

	std::vector<std::shared_ptr<Tensor<float>>> inputs;
	inputs.push_back(input);
	std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
	ReluLayer relu_layer;
	relu_layer.Forward(inputs, outputs);

	ASSERT_EQ(outputs.size(), 1);
	for (int i = 0; i < outputs.size(); ++i) 
	{
		ASSERT_EQ(outputs.at(i)->index(0), 0.f);
		ASSERT_EQ(outputs.at(i)->index(1), 0.f);
		ASSERT_EQ(outputs.at(i)->index(2), 3.f);
	}
}

TEST(test_layer, relu2) 
{
	LOG(INFO) << "start test relu2\n";
	using namespace my_infer;
	std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 224, 224);
	input->Rand();
	//input->Fill(-1);
	std::vector<std::shared_ptr<Tensor<float>>> inputs;
	inputs.push_back(input);
	std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

	ReluLayer relu_layer;
	relu_layer.Forward(inputs, outputs);
	for (int i = 0; i < inputs.size(); ++i) 
	{
		std::shared_ptr<Tensor<float>> input_ = inputs.at(i);

		for (uint32_t i = 0; i < input_->channels(); i++)
		{
			for (uint32_t j = 0; j < input_->rows(); j++)
			{
				for (uint32_t k = 0; k < input_->cols(); k++)
				{
					if (input_->at(i, j, k) < 0)
						input_->at(i, j, k) = 0;
				}
			}
		}

		std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
		CHECK(input_->size() == output_->size());
		uint32_t size = input_->size();
		for (uint32_t j = 0; j < size; ++j)
		{
			ASSERT_EQ(output_->index(j), input_->index(j));
		}
	}
}