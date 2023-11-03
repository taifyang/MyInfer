#pragma once

#include "layer.hpp"

namespace my_infer
{
	class SoftmaxLayer : public Layer 
	{
	public:
		explicit SoftmaxLayer();

		InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;
	};
}


