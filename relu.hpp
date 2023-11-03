#pragma once

#include "layer.hpp"

namespace my_infer
{
	class ReluLayer : public Layer 
	{
	public:
		ReluLayer() : Layer("Relu") { }

		InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

		static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator>& op, std::shared_ptr<Layer>& relu_layer);
	};
}