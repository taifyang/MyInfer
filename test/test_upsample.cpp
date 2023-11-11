#include <glog/logging.h>
#include <gtest/gtest.h>
#include "upsample.hpp"
#include "load_data.hpp"
#include "runtime_ir.hpp"

TEST(test_layer, forward_upsample1)
{
	using namespace my_infer;
	UpSampleLayer layer(2.f, 2.f);

	const uint32_t channels = 1;
	const uint32_t rows = 3;
	const uint32_t cols = 4;

	std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(channels, rows, cols);
	input->Rand();

	std::vector<std::shared_ptr<Tensor<float>>> inputs;
	inputs.push_back(input);

	std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
	layer.Forward(inputs, outputs);

	inputs[0]->Show();
	outputs[0]->Show();

	for (int i = 0; i < outputs.size(); ++i)
	{
		const auto& output = outputs.at(i);
		for (int c = 0; c < channels; ++c)
		{
			const auto& output_channel = output->slice(i);
			const auto& input_channel = input->slice(i);
			ASSERT_EQ(output_channel.dimension(0) / input_channel.dimension(0), 2);
			ASSERT_EQ(output_channel.dimension(1) / input_channel.dimension(1), 2);

			for (int r = 0; r < output_channel.dimension(0); ++r)
			{
				for (int c_ = 0; c_ < output_channel.dimension(1); ++c_)
				{
					ASSERT_EQ(input_channel(r / 2, c_ / 2, 0), output_channel(r, c_, 0)) << r << " " << c_;
				}
			}
		}
	}
}