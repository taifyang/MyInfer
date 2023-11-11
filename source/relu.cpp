#include "relu.hpp"
#include "layer_factory.hpp"

namespace my_infer
{
	InferStatus ReluLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, std::vector<std::shared_ptr<Tensor<float>>>& outputs) 
	{
		if (inputs.empty()) 
		{
			LOG(ERROR) << "The input feature map of relu layer is empty";
			return InferStatus::kInferFailedInputEmpty;
		}
		if (inputs.size() != outputs.size()) 
		{
			LOG(ERROR) << "The input and output size is not adapting";
			return InferStatus::kInferFailedInputOutSizeAdaptingError;
		}

		const uint32_t batch_size = inputs.size();
		for (uint32_t i = 0; i < batch_size; ++i) 
		{
			const sftensor& input_data = inputs.at(i);
			const sftensor& output_data = outputs.at(i);
			if (input_data == nullptr) 
			{
				LOG(ERROR) << "The input feature map of relu layer is empty";
				return InferStatus::kInferFailedInputEmpty;
			}
			if (output_data != nullptr)
			{
				if (input_data->shapes() != output_data->shapes()) 
				{
					LOG(ERROR) << "The input and output size is not adapting";
					return InferStatus::kInferFailedInputOutSizeAdaptingError;
				}
			}
		}

#pragma omp parallel for num_threads(batch_size)
		for (uint32_t i = 0; i < batch_size; ++i) 
		{
			const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
			std::shared_ptr<Tensor<float>> output = outputs.at(i);
			if (output == nullptr)
			{
				output = std::make_shared<Tensor<float>>(input->shapes());
				outputs.at(i) = output;
			}
			for (uint32_t j = 0; j < output->size(); j++)
			{
				output->data()(j) = (input->data()(j) < 0 ? 0 : input->data()(j));
			}
			//output->write_tensor("output.txt");
		}
		return InferStatus::kInferSuccess;
	}

	ParseParameterAttrStatus ReluLayer::GetInstance(const std::shared_ptr<RuntimeOperator>& op, std::shared_ptr<Layer>& relu_layer) 
	{
		CHECK(op != nullptr) << "Relu operator is nullptr";
		relu_layer = std::make_shared<ReluLayer>();
		return ParseParameterAttrStatus::kParameterAttrParseSuccess;
	}

	LayerRegistererWrapper kReluGetInstance("nn.ReLU", ReluLayer::GetInstance);
}  
