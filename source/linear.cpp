#include <glog/logging.h>
#include "layer_factory.hpp"
#include "linear.hpp"

namespace my_infer 
{
	LinearLayer::LinearLayer(int32_t in_features, int32_t out_features, bool use_bias)
		: ParamLayer("Linear"), use_bias_(use_bias), in_features_(in_features), out_features_(out_features) 
	{
		this->InitWeightParam(1, 1, out_features, in_features);
		if (use_bias)
		{
			this->InitBiasParam(1, 1, out_features, 1);
		}
	}

	InferStatus LinearLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, std::vector<std::shared_ptr<Tensor<float>>>& outputs) 
	{
		if (inputs.empty()) 
		{
			LOG(ERROR) << "The input feature map of linear layer is empty";
			return InferStatus::kInferFailedInputEmpty;
		}

		if (inputs.size() != outputs.size()) 
		{
			LOG(ERROR) << "The input and output size is not adapting";
			return InferStatus::kInferFailedInputOutSizeAdaptingError;
		}

		if (this->weights_.empty())
		{
			LOG(ERROR) << "The weight parameters is empty";
			return InferStatus::kInferFailedWeightParameterError;
		}
		else
		{
			if (this->use_bias_ && this->weights_.size() != this->bias_.size())
			{
				return InferStatus::kInferFailedBiasParameterError;
				LOG(ERROR) << "The size of the weight and bias parameters is not equal";
			}
		}

		if (weights_.size() != 1) 
		{
			LOG(ERROR) << "The size of weight parameters is not one";
			return InferStatus::kInferFailedWeightParameterError;
		}

		if (use_bias_ && this->bias_.size() != 1) 
		{
			LOG(ERROR) << "The size of bias parameters is not one";
			return InferStatus::kInferFailedBiasParameterError;
		}

		uint32_t batch = inputs.size();
		const std::shared_ptr<Tensor<float>>& weight = weights_.front();
		Eigen::MatrixXf weight_mat = Eigen::Map<Eigen::MatrixXf>(weight->data().data(), out_features_, in_features_); //1000*512

#pragma omp parallel for num_threads(batch)
		for (uint32_t i = 0; i < batch; ++i) 
		{
			// input matmul weight
			const std::shared_ptr<Tensor<float>>& input = inputs.at(i); 
			const std::vector<uint32_t>& input_shapes = input->shapes();//512 1 1
			//CHECK(input_shapes.size() == 3 && input_shapes.front() == 1);

			const uint32_t feature_dims = input_shapes.at(1);
			CHECK(weight_mat.rows() == out_features_);
			//CHECK(weight_mat.cols() == feature_dims && feature_dims == in_features_); //512 512  512
			const uint32_t input_dim = input_shapes.at(2);

			Eigen::MatrixXf col_mat = Eigen::Map<Eigen::MatrixXf>(input->data().data(), in_features_, input_dim);

			std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, out_features_, input_dim);
			CHECK(output->channels() == 1 && output->rows() == out_features_ && output->cols() == input_dim);
			const auto & output_raw_shapes = output->raw_shapes();
			CHECK(output_raw_shapes.size() == 2);
			CHECK(output_raw_shapes.at(0) == out_features_ && output_raw_shapes.at(1) == input_dim);

			Eigen::MatrixXf result = weight_mat * col_mat; 

			if (use_bias_) 
			{
				CHECK(!this->bias_.empty() && this->bias_.size() == 1);
				std::shared_ptr<Tensor<float>> bias = this->bias_.front();
				Eigen::MatrixXf bias_mat= Eigen::Map<Eigen::MatrixXf>(bias->data().data(), out_features_, 1); 
				CHECK(bias_mat.rows() == out_features_);
				CHECK(bias_mat.cols() == 1);

				bias_mat = bias_mat * Eigen::MatrixXf::Ones(1, input_dim);
				result += bias_mat;
			}

			std::copy(result.data(), result.data() + result.size(), output->data().data());
			outputs.at(i) = output;
		}

		return InferStatus::kInferSuccess;
	}

	ParseParameterAttrStatus LinearLayer::GetInstance(const std::shared_ptr<RuntimeOperator> & op, std::shared_ptr<Layer> & linear_layer) 
	{
		CHECK(op != nullptr) << "Linear operator is nullptr";
		const auto & params = op->params;
		if (params.find("bias") == params.end()) 
		{
			LOG(ERROR) << "Can not find the use bias parameter";
			return ParseParameterAttrStatus::kParameterMissingUseBias;
		}
		const auto& use_bias_param = dynamic_cast<RuntimeParameterBool*>(params.at("bias"));
		if (use_bias_param == nullptr) 
		{
			LOG(ERROR) << "Can not find the use bias parameter";
			return ParseParameterAttrStatus::kParameterMissingUseBias;
		}

		const auto& attr = op->attribute;
		CHECK(!attr.empty()) << "Operator attributes is empty";

		if (attr.find("weight") == attr.end()) 
		{
			LOG(ERROR) << "Can not find the weight parameter";
			return ParseParameterAttrStatus::kAttrMissingWeight;
		}

		if (use_bias_param->value) 
		{
			if (attr.find("bias") == attr.end()) 
			{
				LOG(ERROR) << "Can not find the bias parameter";
				return ParseParameterAttrStatus::kAttrMissingBias;
			}
		}

		const auto& weight = attr.at("weight");
		const auto& bias = attr.at("bias");
		const auto& shapes = weight->shape;
		CHECK(shapes.size() == 2) << "The graph only support two dimension matrix multiply";

		int32_t out_features = shapes.at(0);
		int32_t in_features = shapes.at(1);
		const bool use_bias = use_bias_param->value;

		linear_layer = std::make_shared<LinearLayer>(in_features, out_features, use_bias);
		if (use_bias) 
		{
			linear_layer->set_bias(bias->get<float>());
		}

		// load weights
		linear_layer->set_weights(weight->get<float>());
		return ParseParameterAttrStatus::kParameterAttrParseSuccess;
	}

	LayerRegistererWrapper kLinearGetInstance("nn.Linear", LinearLayer::GetInstance);
}