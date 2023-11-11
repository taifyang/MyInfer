#include <gtest/gtest.h>
#include <glog/logging.h>
#include "tensor.hpp"
#include "silu.hpp"

TEST(test_layer, forward_silu1) 
{
	using namespace my_infer;
	std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(32, 224, 512);
	input->Rand();
	std::vector<std::shared_ptr<Tensor<float>>> inputs;
	inputs.push_back(input);
	std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

	SiLULayer silu_layer;
	silu_layer.Forward(inputs, outputs);
	for (int i = 0; i < inputs.size(); ++i)
	{
		std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
		std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
		CHECK(input_->size() == output_->size());
		uint32_t size = input_->size();
		for (uint32_t j = 0; j < size; ++j) 
			ASSERT_LE(std::abs(output_->index(j) - input_->index(j) / (1 + std::exp(-input_->index(j)))), 1e-5);
	}
}