#pragma once

#include "layer.hpp"

namespace my_infer 
{
	class CatLayer : public Layer 
	{
	public:
		explicit CatLayer(int dim);

		InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

		static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator>& op, std::shared_ptr<Layer>& cat_layer);

	private:
		int32_t dim_ = 0;
	};
}