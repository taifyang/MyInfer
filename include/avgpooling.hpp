#pragma once

#include "layer.hpp"

namespace my_infer 
{
	class AvgPoolingLayer : public Layer 
	{
	public:
		explicit AvgPoolingLayer(uint32_t output_h, uint32_t output_w);

		InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

		static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator>& op, std::shared_ptr<Layer>& avg_layer);

	private:
		uint32_t output_h_ = 0;
		uint32_t output_w_ = 0;
	};
}