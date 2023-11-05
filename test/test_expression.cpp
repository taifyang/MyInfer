#include <gtest/gtest.h>
#include <glog/logging.h>
#include "expression.hpp"

TEST(test_expression, expression1)
{
	LOG(INFO) << "start test expression1\n";
	using namespace my_infer;

	ExpressionLayer expression_layer("add(@0,@1)");
	std::vector<std::shared_ptr<ftensor>> inputs;
	std::vector<std::shared_ptr<ftensor>> outputs;

	int batch_size = 4;
	for (int i = 0; i < batch_size; ++i) 
	{
		std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
		input->Fill(1.f);
		inputs.push_back(input);
	}
	for (int i = 0; i < batch_size; ++i)
	{
		std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
		input->Fill(2.f);
		inputs.push_back(input);
	}
	for (int i = 0; i < batch_size; ++i)
	{
		std::shared_ptr<ftensor> output = std::make_shared<ftensor>(3, 224, 224);
		outputs.push_back(output);
	}

	expression_layer.Forward(inputs, outputs);
	for (int i = 0; i < batch_size; ++i) 
	{
		const auto& result = outputs.at(i);
		for (int j = 0; j < result->size(); ++j) 
		{
			ASSERT_EQ(result->index(j), 3.f);
		}
	}
}

TEST(test_expression, expression2)
{
	LOG(INFO) << "start test expression2\n";
	using namespace my_infer;

	ExpressionLayer expression_layer("add(mul(@0,@1),@2)");
	std::vector<std::shared_ptr<ftensor>> inputs;
	std::vector<std::shared_ptr<ftensor>> outputs;

	int batch_size = 4;
	for (int i = 0; i < batch_size; ++i)
	{
		std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
		input->Fill(1.f);
		inputs.push_back(input);
	}
	for (int i = 0; i < batch_size; ++i) 
	{
		std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
		input->Fill(2.f);
		inputs.push_back(input);
	}
	for (int i = 0; i < batch_size; ++i) 
	{
		std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
		input->Fill(3.f);
		inputs.push_back(input);
	}
	for (int i = 0; i < batch_size; ++i) 
	{
		std::shared_ptr<ftensor> output = std::make_shared<ftensor>(3, 224, 224);
		outputs.push_back(output);
	}

	expression_layer.Forward(inputs, outputs);
	for (int i = 0; i < batch_size; ++i) 
	{
		const auto& result = outputs.at(i);
		for (int j = 0; j < result->size(); ++j)
		{
			ASSERT_EQ(result->index(j), 5.f);
		}
	}
}
