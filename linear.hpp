#pragma once

#include "layer.hpp"
#include "param_layer.hpp"

namespace my_infer 
{
	class LinearLayer : public ParamLayer 
	{
	public:
		explicit LinearLayer(int32_t in_features, int32_t out_features, bool use_bias);

		InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

		static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator>& op, std::shared_ptr<Layer>& linear_layer);

	private:
		int32_t in_features_ = 0;
		int32_t out_features_ = 0;
		bool use_bias_ = false;
	};
}