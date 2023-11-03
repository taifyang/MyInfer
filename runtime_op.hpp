#pragma once

#include <vector>
#include <unordered_map>
#include <map>
#include <memory>
#include <string>
#include "ir.h"
#include "layer.hpp"
#include "runtime_operand.hpp"
#include "runtime_attr.hpp"
#include "runtime_parameter.hpp"

namespace my_infer
{
	class Layer;

	// ����ͼ�еļ���ڵ�
	struct RuntimeOperator 
	{
		int32_t meet_num = 0;			// ����ڵ㱻�����ӽڵ���ʵ��Ĵ���
		virtual ~RuntimeOperator();
		std::string name;				// ����ڵ������
		std::string type;				// ����ڵ������
		std::shared_ptr<Layer> layer;	// �ڵ��Ӧ�ļ���Layer

		std::vector<std::string> output_names;				// �ڵ������ڵ�����
		std::shared_ptr<RuntimeOperand> output_operands;	// �ڵ�����������

		std::map<std::string, std::shared_ptr<RuntimeOperand>> input_operands;		// �ڵ�����������
		std::vector<std::shared_ptr<RuntimeOperand>> input_operands_seq;			// �ڵ�������������˳������
		std::map<std::string, std::shared_ptr<RuntimeOperator>>	output_operators;	// ����ڵ�����ֺͽڵ��Ӧ

		std::map<std::string, RuntimeParameter*> params;						// ���ӵĲ�����Ϣ
		std::map<std::string, std::shared_ptr<RuntimeAttribute>> attribute;		// ���ӵ�������Ϣ���ں�Ȩ����Ϣ
	};

	class RuntimeOperatorUtils 
	{
	public:
		/**
		 * ���ͼ�ǵ�һ�����У�����ݽڵ�����operand����״׼���ú���Layer����������Ҫ��Tensor
		 * ���ͼ�ǵڶ����������У���������operand����״��operand����������״�Ƿ�ƥ��
		 * @param operators ����ͼ�еļ���ڵ�
		 */
		static void InitOperatorInput(const std::vector<std::shared_ptr<RuntimeOperator>>& operators);

		/**
		 * ���ͼ�ǵ�һ�����У�����ݽڵ����operand����״׼���ú���Layer����������Ҫ��Tensor
		 * ���ͼ�ǵڶ����������У��������operand����״��operand����������״�Ƿ�ƥ��
		 * @param pnnx_operators pnnxͼ�ڵ�
		 * @param operators KuiperInfer����ͼ�еļ���ڵ�
		 */
		static void InitOperatorOutput(const std::vector<pnnx::Operator*>& pnnx_operators, const std::vector<std::shared_ptr<RuntimeOperator>>& operators);
	};
}