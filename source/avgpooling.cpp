#include <glog/logging.h>
#include <numeric>
#include "avgpooling.hpp"
#include "layer_factory.hpp"

namespace my_infer 
{
	AvgPoolingLayer::AvgPoolingLayer(uint32_t output_h, uint32_t output_w)
		: Layer("AdaptiveAveragePooling"), output_h_(output_h), output_w_(output_w) {}

	InferStatus AvgPoolingLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, std::vector<std::shared_ptr<Tensor<float>>>& outputs) 
	{
		if (inputs.empty())
		{
			LOG(ERROR) << "The input feature map of adaptive pooling layer is empty";
			return InferStatus::kInferFailedInputEmpty;
		}

		if (inputs.size() != outputs.size()) 
		{
			LOG(ERROR) << "The input and output size of adaptive pooling layer is not adapting";
			return InferStatus::kInferFailedInputOutSizeAdaptingError;
		}

		if (output_w_ <= 0 || output_h_ <= 0) 
		{
			LOG(ERROR) << "The output size of adaptive pooling is less than zero";
			return InferStatus::kInferFailedOutputSizeError;
		}

		const uint32_t batch = inputs.size();
		for (uint32_t i = 0; i < batch; ++i) 
		{
			const std::shared_ptr<ftensor>& input_data = inputs.at(i);
			const std::shared_ptr<ftensor>& output_data = outputs.at(i);
			if (input_data == nullptr) 
			{
				LOG(ERROR) << "The input feature map of adaptive pooling layer is empty";
				return InferStatus::kInferFailedInputEmpty;
			}
			if (output_data != nullptr)
			{
				if (output_data->rows() != output_h_ || output_data->cols() != output_w_) 
				{
					LOG(ERROR) << "The output size of adaptive pooling is not adapting";
					return InferStatus::kInferFailedOutputSizeError;
				}
			}
		}

#pragma omp parallel for num_threads(batch)
		for (uint32_t i = 0; i < batch; ++i) 
		{
			const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
			const uint32_t input_h = input->rows();
			const uint32_t input_w = input->cols();
			const uint32_t input_c = input->channels();
			const uint32_t stride_h = uint32_t(std::floor(input_h / output_h_));
			const uint32_t stride_w = uint32_t(std::floor(input_w / output_w_));
			CHECK(stride_w > 0 && stride_h > 0) << "The stride parameter is set incorrectly. It must always be greater than 0";

			const uint32_t pooling_h = input_h - (output_h_ - 1) * stride_h;
			const uint32_t pooling_w = input_w - (output_w_ - 1) * stride_w;
			CHECK(pooling_w > 0 && pooling_h > 0) << "The pooling parameter is set incorrectly. It must always be greater than 0";

			std::shared_ptr<Tensor<float>> output = outputs.at(i);
			output = std::make_shared<Tensor<float>>(input_c, output_h_, output_w_);
			CHECK(output->rows() == output_h_ && output->cols() == output_w_ && output->channels() == input_c) << "The output size of adaptive pooling is error";
			
			for (uint32_t ic = 0; ic < input_c; ++ic)
			{
				for (uint32_t r = 0; r < input_h - pooling_h + 1; r += stride_h)
				{
					for (uint32_t c = 0; c < input_w - pooling_w + 1; c += stride_w)
					{
						Eigen::array<Eigen::DenseIndex, 3> offsets = { r, c, ic };
						Eigen::array<Eigen::DenseIndex, 3> extents = { pooling_h, pooling_w, 1 };
						Eigen::Tensor<float, 3> region = input->data().slice(offsets, extents);
						Eigen::Tensor<float, 0> mean = region.mean();
						output->at(ic, int(r / stride_h), int(c / stride_w)) = (float)mean(0);
					}
				}
			}
			outputs.at(i) = output;
			//output->write_tensor("output.txt");
		}
		return InferStatus::kInferSuccess;
	}

	ParseParameterAttrStatus AvgPoolingLayer::GetInstance(const std::shared_ptr<RuntimeOperator> & op, std::shared_ptr<Layer> & avg_layer)
	{
		CHECK(op != nullptr) << "Adaptive pooling operator is nullptr";
		const auto & params = op->params;
		CHECK(!params.empty()) << "Operator parameter is empty";

		const auto & output_hw = dynamic_cast<RuntimeParameterIntArray*>(params.at("output_size"));
		if (!output_hw) 
		{
			LOG(ERROR) << "Can not find the output size parameter";
			return ParseParameterAttrStatus::kParameterMissingOutHW;
		}

		const auto& output_hw_arr = output_hw->value;
		if (output_hw_arr.size() != 2) 
		{
			LOG(ERROR) << "Can not find the output size parameter";
			return ParseParameterAttrStatus::kParameterMissingOutHW;
		}
		avg_layer = std::make_shared<AvgPoolingLayer>(output_hw_arr.at(0), output_hw_arr.at(1));

		return ParseParameterAttrStatus::kParameterAttrParseSuccess;
	}

	LayerRegistererWrapper kAdaptiveAvgpoolingGetInstance("nn.AdaptiveAvgPool2d", AvgPoolingLayer::GetInstance);
}