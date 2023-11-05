#include <glog/logging.h>
#include <gtest/gtest.h>
#include "tensor.hpp"
#include "flatten.hpp"

TEST(test_layer, forward_flatten_layer) 
{
	LOG(INFO) << "start test forward_flatten_layer\n";
	using namespace my_infer;

	std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(2, 3, 4);
	input->Rand();

	std::vector<std::shared_ptr<Tensor<float>>> inputs;
	std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

	inputs.push_back(input);

	FlattenLayer flatten_layer(1, 3);
	flatten_layer.Forward(inputs, outputs);
	ASSERT_EQ(outputs.size(), 1);

	const auto& shapes = outputs.front()->shapes();
	ASSERT_EQ(shapes.size(), 3);

	ASSERT_EQ(shapes.at(0), 1);
	ASSERT_EQ(shapes.at(1), 2 * 3 * 4);
	ASSERT_EQ(shapes.at(2), 1);

	const auto& raw_shapes = outputs.front()->raw_shapes();
	ASSERT_EQ(raw_shapes.size(), 1);
	ASSERT_EQ(raw_shapes.front(), 2 * 3 * 4);

	uint32_t size1 = inputs.front()->size();
	uint32_t size2 = outputs.front()->size();
	ASSERT_EQ(size1, size2);
}