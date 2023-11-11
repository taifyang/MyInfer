#include "upsample.hpp"
#include "layer_factory.hpp"

namespace my_infer 
{
	UpSampleLayer::UpSampleLayer(float scale_h, float scale_w, UpSampleMode mode): Layer("upsample"), scale_h_(scale_h), scale_w_(scale_w), mode_(mode) {}

	InferStatus UpSampleLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, std::vector<std::shared_ptr<Tensor<float>>>& outputs)
	{
		if (inputs.empty())
		{
			LOG(ERROR) << "The input feature map of upsample layer is empty";
			return InferStatus::kInferFailedInputEmpty;
		}

		if (inputs.size() != outputs.size()) 
		{
			LOG(ERROR) << "The input and output size is not adapting";
			return InferStatus::kInferFailedInputOutSizeAdaptingError;
		}

		for (uint32_t i = 0; i < inputs.size(); ++i)
		{
			const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
			CHECK(input != nullptr) << "The input feature map of cat layer is empty";
		}

		LOG_IF(FATAL, this->mode_ != UpSampleMode::kModeNearest) << "Unsupported upsample mode: " << int(mode_);

		const uint32_t batch_size = inputs.size();
#pragma omp parallel for num_threads(batch_size)
		for (int i = 0; i < batch_size; ++i) 
		{
			const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
			std::shared_ptr<Tensor<float>> output = outputs.at(i);
			if (output == nullptr) 
			{
				output = std::make_shared<Tensor<float>>(input->channels(), uint32_t(input->rows() * scale_h_), uint32_t(input->cols() * scale_w_));
			}

			CHECK(output->rows() == input->rows() * scale_h_) << "The height of the feature map is not adapting!";
			CHECK(output->cols() == input->cols() * scale_w_) << "The width of the feature map is not adapting!";
			CHECK(output->channels() == input->channels()) << "The channel of the feature map is not adapting!";

			const uint32_t channels = input->channels();
			for (uint32_t c = 0; c < channels; ++c) 
			{
				const uint32_t input_w = input->cols();
				const uint32_t input_h = input->rows();
				const uint32_t output_w = output->cols();
				const uint32_t output_h = output->rows();

				for (uint32_t w = 0; w < output_w; ++w)
				{
					const uint32_t src_w = uint32_t((float)w / this->scale_w_);
					CHECK(src_w < input_w);

					for (uint32_t h = 0; h < output_h; ++h) 
					{
						const uint32_t src_h = uint32_t((float)h / this->scale_h_);
						CHECK(src_h < input_h);

						*(output->raw_ptr() + c* output_w * output_h + w * output_h + h) = *(input->raw_ptr() + c * input_w * input_h + src_w * input_h + src_h);
					}
				}
			}
			outputs.at(i) = output;
		}
		return InferStatus::kInferSuccess;
	}

	ParseParameterAttrStatus UpSampleLayer::GetInstance(const std::shared_ptr<RuntimeOperator> & op, std::shared_ptr<Layer> & upsample_layer) 
	{
		CHECK(op != nullptr) << "Upsample operator is null";
		const auto & params = op->params;
		CHECK(!params.empty()) << "Operator parameter is empty";
		if (params.find("scale_factor") == params.end()) 
		{
			LOG(ERROR) << "Can not find the scale factor parameter";
			return ParseParameterAttrStatus::kParameterMissingScale;
		}

		const auto& scale_param = params.at("scale_factor");
		const auto& scales = dynamic_cast<RuntimeParameterFloatArray*>(params.at("scale_factor"));
		if (scales == nullptr) 
		{
			LOG(ERROR) << "Can not find the scale factor parameter";
			return ParseParameterAttrStatus::kParameterMissingScale;
		}
		CHECK(scales->value.size() == 2) << "Scale factor need two dimension";

		if (params.find("mode") == params.end()) 
		{
			LOG(ERROR) << "Can not find the mode parameter";
			return ParseParameterAttrStatus::kParameterMissingResizeMode;
		}

		const auto& mode = dynamic_cast<RuntimeParameterString*>(params.at("mode"));
		CHECK(mode->value == "nearest") << "The mode " << mode->value << " is not supported!";
		upsample_layer = std::make_shared<UpSampleLayer>(scales->value.at(0), scales->value.at(1));
		return ParseParameterAttrStatus::kParameterAttrParseSuccess;
	}

	LayerRegistererWrapper kUpSamplerGetInstance("nn.Upsample", UpSampleLayer::GetInstance);
}