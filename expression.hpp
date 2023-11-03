#pragma once

#include "layer.hpp"
#include "parse_expression.hpp"

namespace my_infer 
{
	class ExpressionLayer : public Layer 
	{
	public:
		explicit ExpressionLayer(const std::string& statement);

		InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

		static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator>& op, std::shared_ptr<Layer>& expression_layer);

	private:
		std::unique_ptr<ExpressionParser> parser_;
	};
}