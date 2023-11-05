#include <glog/logging.h>
#include <gtest/gtest.h>
#include "runtime_ir.hpp"

TEST(test_forward, forward1)
{
	LOG(INFO) << "start test forward1\n";
	using namespace my_infer;

	const std::string& param_path = "./tmp/test.pnnx.param";
	const std::string& bin_path = "./tmp/test.pnnx.bin";
	RuntimeGraph graph(param_path, bin_path);
	graph.Build("pnnx_input_0", "pnnx_output_0");
	uint32_t batch_size = 1;
	std::vector<sftensor> inputs(batch_size);
	for (uint32_t i = 0; i < batch_size; ++i)
	{
		inputs.at(i) = std::make_shared<ftensor>(1, 16, 16);
		inputs.at(i)->Fill(1.f);
	}
	const std::vector<sftensor>& outputs = graph.Forward(inputs, true);
}

TEST(test_forward, forward2) 
{
	LOG(INFO) << "start test forward2\n";
	using namespace my_infer;

	const std::string& param_path = "./tmp/resnet18_hub.pnnx.param";
	const std::string& bin_path = "./tmp/resnet18_hub.pnnx.bin";
	RuntimeGraph graph(param_path, bin_path);
	graph.Build("pnnx_input_0", "pnnx_output_0");
	uint32_t batch_size = 2;
	std::vector<sftensor> inputs(batch_size);
	for (uint32_t i = 0; i < batch_size; ++i) 
	{
		inputs.at(i) = std::make_shared<ftensor>(3, 256, 256);
		inputs.at(i)->Fill(1.f);
	}
	const std::vector<sftensor>& outputs = graph.Forward(inputs, true);
}
