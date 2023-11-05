#pragma once

#include <string>
#include "tensor.hpp"
#include "status_code.hpp"
#include "runtime_op.hpp"

namespace my_infer 
{
	class Layer 
	{
	public:
		explicit Layer(std::string layer_name) : layer_name_(std::move(layer_name)) { }

		virtual ~Layer() = default;

		/**
		 * Layer��ִ�к���
		 * @param inputs �������
		 * @param outputs ������
		 * @return ִ�е�״̬
		 */
		virtual InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, std::vector<std::shared_ptr<Tensor<float>>>& outputs);

		/**
		 * ���ز��Ȩ��
		 * @return ���ص�Ȩ��
		 */
		virtual const std::vector<std::shared_ptr<Tensor<float>>>& weights() const;

		/**
		 * ���ز��ƫ����
		 * @return ���ص�ƫ����
		 */
		virtual const std::vector<std::shared_ptr<Tensor<float>>>& bias() const;

		/**
		 * ����Layer��Ȩ��
		 * @param weights Ȩ��
		 */
		virtual void set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights);

		/**
		 * ����Layer��ƫ����
		 * @param bias ƫ����
		 */
		virtual void set_bias(const std::vector<std::shared_ptr<Tensor<float>>>& bias);

		/**
		 * ����Layer��Ȩ��
		 * @param weights Ȩ��
		 */
		virtual void set_weights(const std::vector<float>& weights);

		/**
		 * ����Layer��ƫ����
		 * @param bias ƫ����
		 */
		virtual void set_bias(const std::vector<float>& bias);

		/**
		 * ���ز������
		 * @return �������
		 */
		virtual const std::string& layer_name() const { return this->layer_name_; }

	protected:
		std::string layer_name_; /// Layer������
	};
}