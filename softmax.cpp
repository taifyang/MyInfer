#include <glog/logging.h>
#include <numeric>
#include "softmax.hpp"

namespace my_infer 
{
	SoftmaxLayer::SoftmaxLayer() : Layer("Softmax") { }

	InferStatus SoftmaxLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, std::vector<std::shared_ptr<Tensor<float>>>& outputs) 
	{
		if (inputs.empty()) 
		{
			LOG(ERROR) << "The input feature map of softmax layer is empty";
			return InferStatus::kInferFailedInputEmpty;
		}

		if (inputs.size() != outputs.size()) 
		{
			LOG(ERROR) << "The input and output size is not adapting";
			return InferStatus::kInferFailedInputOutSizeAdaptingError;
		}

		const uint32_t batch_size = inputs.size();
#pragma omp parallel for num_threads(batch_size)
		for (uint32_t i = 0; i < batch_size; ++i) 
		{
			const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
			CHECK(input != nullptr) << "The input feature map for softmax layer is empty";

			std::shared_ptr<Tensor<float>> output = input->Clone();
			//const float max = *std::max_element(output_data->data().data(), output_data->data().data() + output_data->size());
			const float sum = std::accumulate(output->data().data(), output->data().data() + output->size(), 0.0f, [](float a, float b) {return a + exp(b); });
			for (uint32_t i = 0; i < output->size(); i++)
			{
				output->index(i) = std::exp(output->index(i)) / sum;
			}
			output->Show();
			outputs.at(i) = output;
		}
		return InferStatus::kInferSuccess;
	}
}