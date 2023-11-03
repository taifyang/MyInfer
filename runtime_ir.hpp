#pragma once

#include <vector>
#include <string>
#include <glog/logging.h>
#include <memory>
#include <map>
#include <queue>

#include "ir.h"
#include "layer_factory.hpp"
#include "runtime_operand.hpp"
#include "runtime_op.hpp"

namespace my_infer 
{
	// ����ͼ�ṹ���ɶ������ڵ�ͽڵ�֮���������ͼ���
	class RuntimeGraph 
	{
	public:
		/**
		 * ��ʼ������ͼ
		 * @param param_path ����ͼ�Ľṹ�ļ�
		 * @param bin_path ����ͼ�е�Ȩ���ļ�
		 */
		RuntimeGraph(std::string param_path, std::string bin_path);

		/**
		 * ��������ͼ
		 * @param input_name ����ͼ����ڵ������
		 * @param output_name  ����ͼ����ڵ������
		 */
		void Build(const std::string& input_name, const std::string& output_name);

		/**
		 * ����Ȩ���ļ�
		 * @param bin_path Ȩ���ļ�·��
		 */
		void set_bin_path(const std::string& bin_path);

		/**
		 * ���ýṹ�ļ�
		 * @param param_path  �ṹ�ļ�·��
		 */
		void set_param_path(const std::string& param_path);

		/**
		 * ����Ȩ���ļ�
		 * @return ����Ȩ���ļ�
		 */
		const std::string& bin_path() const;

		/**
		 * ���ؽṹ�ļ�
		 * @return ���ؽṹ�ļ�
		 */
		const std::string& param_path() const;

		/**
		 * ����ͼ��ִ��,���ݹ������������˳��ִ��
		 * @param inputs ����ͼ����������
		 * @param debug �Ƿ���ԣ�������������һЩ�м���Ϣ
		 * @return ����ͼ���������
		 */
		std::vector<std::shared_ptr<Tensor<float>>> Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, bool debug = false);

	private:
		/**
		 * ����ͼ�ĳ�ʼ��
		 * @return �Ƿ��ʼ���ɹ�
		 */
		bool Init();

		/**
		 * ��ʼ��kuiper infer����ͼ�ڵ��е����������
		 * @param inputs pnnx�е����������
		 * @param runtime_operator ����ͼ�ڵ�
		 */
		static void InitGraphOperatorsInput(const std::vector<pnnx::Operand*>& inputs, const std::shared_ptr<RuntimeOperator>& runtime_operator);

		/**
		 * ��ʼ��kuiper infer����ͼ�ڵ��е����������
		 * @param outputs pnnx�е����������
		 * @param runtime_operator ����ͼ�ڵ�
		 */
		static void InitGraphOperatorsOutput(const std::vector<pnnx::Operand*>& outputs, const std::shared_ptr<RuntimeOperator>& runtime_operator);

		/**
		 * ��ʼ��kuiper infer����ͼ�еĽڵ�����
		 * @param attrs pnnx�еĽڵ�����
		 * @param runtime_operator ����ͼ�ڵ�
		 */
		static void InitGraphAttrs(const std::map<std::string, pnnx::Attribute>& attrs, const std::shared_ptr<RuntimeOperator>& runtime_operator);

		/**
		 * ��ʼ��kuiper infer����ͼ�еĽڵ����
		 * @param params pnnx�еĲ�������
		 * @param runtime_operator ����ͼ�ڵ�
		 */
		static void InitGraphParams(const std::map<std::string, pnnx::Parameter>& params, const std::shared_ptr<RuntimeOperator>& runtime_operator);

		/**
		 * ���ݼ���ͼ�еļ���ڵ�������Layer
		 * @param op ����ͼ�еļ���ڵ�
		 * @return �����ɹ���Layer
		 */
		static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<RuntimeOperator>& op);

		/**
		 * ��鵱ǰ�ڵ��Ƿ����
		 * @param op �����Ľڵ�
		 * @return �Ƿ����
		 */
		static bool CheckOperatorReady(const std::shared_ptr<RuntimeOperator>& op);

		/**
		 * ̽����һ��ļ���ڵ�
		 * @param current_op ��ǰ����ڵ�
		 * @param operator_queue ����ڵ�ļ�������
		 * @param layer_output_data ��ǰ�ڵ����������赽��һ�����ڵ������������
		 */
		static void ProbeNextLayer(const std::shared_ptr<RuntimeOperator>& current_op,
			std::deque<std::shared_ptr<RuntimeOperator>>& operator_queue, const std::vector<std::shared_ptr<Tensor<float>>>& layer_output_data);

	private:
		enum class GraphState 
		{
			NeedInit = -2,
			NeedBuild = -1,
			Complete = 0,
		};

		GraphState graph_state_ = GraphState::NeedInit;
		std::string input_name_;														// ����ͼ����ڵ������
		std::string output_name_;														// ����ͼ����ڵ������
		std::string param_path_;														// ����ͼ�Ľṹ�ļ�
		std::string bin_path_;															// ����ͼ��Ȩ���ļ�
		std::map<std::string, std::shared_ptr<RuntimeOperator>> input_operators_maps_;	// ��������ڵ�
		std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators_maps_; // ��������ڵ�
		std::vector<std::shared_ptr<RuntimeOperator>> operators_;						// ����ͼ�ļ���ڵ�
		std::unique_ptr<pnnx::Graph> graph_;											// pnnx��graph
	};
}