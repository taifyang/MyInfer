#include "silu.hpp"
#include "layer_factory.hpp"

namespace my_infer 
{
	SiLULayer::SiLULayer() : Layer("SiLU") {}

	InferStatus SiLULayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, std::vector<std::shared_ptr<Tensor<float>>>& outputs) 
	{
		if (inputs.empty()) 
		{
			LOG(ERROR) << "The input feature map of silu layer is empty";
			return InferStatus::kInferFailedInputEmpty;
		}

		if (inputs.size() != outputs.size()) 
		{
			LOG(ERROR) << "The input and output size of silu layer is not adapting";
			return InferStatus::kInferFailedInputOutSizeAdaptingError;
		}

		const uint32_t batch_size = inputs.size();
#pragma omp parallel for num_threads(batch_size)
		for (uint32_t i = 0; i < batch_size; ++i)
		{
			const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
			CHECK(input != nullptr) << "The input feature map of silu layer is empty!";

			std::shared_ptr<Tensor<float>> output = outputs.at(i);
			if (output == nullptr)
			{
				output = std::make_shared<Tensor<float>>(input->shapes());
				outputs.at(i) = output;
			}
			for (uint32_t j = 0; j < output->size(); j++)
			{
				output->data()(j) = input->data()(j) / (1.f + std::exp(-input->data()(j)));
			}
			//output->write_tensor("output.txt");
		}
		return InferStatus::kInferSuccess;
	}

	ParseParameterAttrStatus SiLULayer::GetInstance(const std::shared_ptr<RuntimeOperator> & op, std::shared_ptr<Layer> & silu_layer)
	{
		CHECK(op != nullptr) << "SiLU operator is nullptr";
		silu_layer = std::make_shared<SiLULayer>();
		return ParseParameterAttrStatus::kParameterAttrParseSuccess;
	}

	LayerRegistererWrapper kSiluGetInstance("nn.SiLU", SiLULayer::GetInstance);
}
