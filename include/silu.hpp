#pragma once

#include "layer.hpp"

namespace my_infer 
{
	class SiLULayer : public Layer 
	{
	public:
		explicit SiLULayer();

		InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

		static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator>& op, std::shared_ptr<Layer>& silu_layer);
	};
}