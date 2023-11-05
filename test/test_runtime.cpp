#include <gtest/gtest.h>
#include <glog/logging.h>
#include "runtime_ir.hpp"

TEST(test_runtime, runtime1) 
{
	LOG(INFO) << "start test runtime1\n";
	using namespace my_infer;

	const std::string& param_path = "./tmp/test.pnnx.param";
	const std::string& bin_path = "./tmp/test.pnnx.bin";
	RuntimeGraph graph(param_path, bin_path);
	graph.Build("pnnx_input_0", "pnnx_output_0");
}

TEST(test_runtime, runtime2)
{
	LOG(INFO) << "start test runtime2\n";
	using namespace my_infer;

	const std::string& param_path = "./tmp/resnet18_hub.pnnx.param";
	const std::string& bin_path = "./tmp/resnet18_hub.pnnx.bin";
	RuntimeGraph graph(param_path, bin_path);
	graph.Build("pnnx_input_0", "pnnx_output_0");
}