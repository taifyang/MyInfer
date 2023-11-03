#include "runtime_op.hpp"

namespace my_infer
{
	RuntimeOperator::~RuntimeOperator()
	{
		for (auto& param : this->params)
		{
			if (param.second != nullptr) 
			{
				delete param.second;
				param.second = nullptr;
			}
		}
	}

	void RuntimeOperatorUtils::InitOperatorInput(const std::vector<std::shared_ptr<RuntimeOperator>>& operators) 
	{
		if (operators.empty()) 
		{
			LOG(ERROR) << "Operators for init input shapes is empty!";
			return;
		}

		for (const auto& op : operators) 
		{
			if (op->input_operands.empty()) 
			{
				continue;
			}
			else 
			{
				const std::map<std::string, std::shared_ptr<RuntimeOperand>>& input_operands_map = op->input_operands;
				// ��ʼ��operator������ռ�
				for (const auto& input_operand_iter : input_operands_map) 
				{
					const auto& input_operand = input_operand_iter.second;
					const auto& type = input_operand->type;
					CHECK(type == RuntimeDataType::kTypeFloat32) << "The graph only support float32 yet!";
					const auto & input_operand_shape = input_operand->shapes;
					// �õ���Ҫ��ʼ���Ŀռ�
					auto & input_datas = input_operand->datas;

					CHECK(!input_operand_shape.empty());
					const int32_t batch = input_operand_shape.at(0);
					CHECK(batch >= 0) << "Dynamic batch size is not supported!";
					CHECK(input_operand_shape.size() == 2 || input_operand_shape.size() == 4 || input_operand_shape.size() == 3)
						<< "Unsupported tensor shape sizes: " << input_operand_shape.size();

					if (!input_datas.empty()) 
						CHECK_EQ(input_datas.size(), batch);
					else 
						input_datas.resize(batch);
				}
			}
		}
	}

	void RuntimeOperatorUtils::InitOperatorOutput(const std::vector<pnnx::Operator*>& pnnx_operators, const std::vector<std::shared_ptr<RuntimeOperator>>& operators)
	{
		CHECK(!pnnx_operators.empty() && !operators.empty());
		CHECK(pnnx_operators.size() == operators.size());
		for (uint32_t i = 0; i < pnnx_operators.size(); ++i)
		{
			// �õ�pnnxԭ�е�����ռ�
			const std::vector<pnnx::Operand*> operands = pnnx_operators.at(i)->outputs;
			CHECK(operands.size() <= 1) << "Only support one node one output yet!";
			if (operands.empty())
			{
				continue;
			}
			CHECK(operands.size() == 1) << "Only support one output";
			// һ���ڵ��֧��һ�������ʵ������pnnx��һ���ڵ�ӵ��������ͬ��������Ҳ�ǲ����ڵ�
			pnnx::Operand * operand = operands.front();
			const auto & runtime_op = operators.at(i);
			CHECK(operand != nullptr) << "Operand output is null";
			const std::vector<int32_t> & operand_shapes = operand->shape;
			// �õ���Ҫ��ʼ��������ռ�
			const auto & output_tensors = runtime_op->output_operands;
			// ��ȡ�ڵ���������Ӧ����״
			const int32_t batch = operand_shapes.at(0);
			CHECK(batch >= 0) << "Dynamic batch size is not supported!";
			CHECK(operand_shapes.size() == 2 || operand_shapes.size() == 4 || operand_shapes.size() == 3) << "Unsupported shape sizes: " << operand_shapes.size();

			// �������ռ�û�б���ʼ����
			if (!output_tensors) 
			{
				// ��Ҫ����ʼ�����������
				std::shared_ptr<RuntimeOperand> output_operand = std::make_shared<RuntimeOperand>();
				// �����������������
				output_operand->shapes = operand_shapes;
				output_operand->type = RuntimeDataType::kTypeFloat32;
				output_operand->name = operand->name + "_output";
				// ����ռ��ʼ��
				for (int j = 0; j < batch; ++j) 
				{
					if (operand_shapes.size() == 4) 
					{
						output_operand->datas.push_back(std::make_shared<Tensor<float>>(operand_shapes.at(1), operand_shapes.at(2), operand_shapes.at(3)));
					}
					else if (operand_shapes.size() == 2) 
					{
						output_operand->datas.push_back(std::make_shared<Tensor<float>>(1, operand_shapes.at(1), 1));
					}
					else 
					{
						// current shape is 3
						output_operand->datas.push_back(std::make_shared<Tensor<float>>(1, operand_shapes.at(1), operand_shapes.at(2)));
					}
				}
				runtime_op->output_operands = std::move(output_operand);
			}
			else 
			{
				// �������ռ䲻Ϊ��
				CHECK(batch == output_tensors->datas.size());
				CHECK(output_tensors->type == RuntimeDataType::kTypeFloat32);
				CHECK(output_tensors->shapes == operand_shapes);
				// �����μ������ռ����״�Ƿ������������������reshape
				for (uint32_t j = 0; j < batch; ++j) 
				{
					const std::vector<uint32_t>& tensor_shapes = output_tensors->datas.at(j)->shapes();
					if (operand_shapes.size() == 4) 
					{
						if (tensor_shapes.at(0) != operand_shapes.at(1) || tensor_shapes.at(1) != operand_shapes.at(2) || tensor_shapes.at(2) != operand_shapes.at(3)) 
						{
							DLOG(WARNING) << "The shape of tensor do not adapting with output operand";
							const auto& target_shapes = std::vector<uint32_t>{(uint32_t)operand_shapes.at(1), (uint32_t)operand_shapes.at(2), (uint32_t)operand_shapes.at(3) };
							output_tensors->datas.at(j)->Reshape(target_shapes);
						}
					}
					else if (operand_shapes.size() == 2)
					{
						if (tensor_shapes.at(0) != 1 || tensor_shapes.at(1) != operand_shapes.at(1) || tensor_shapes.at(2) != 1)
						{
							DLOG(WARNING) << "The shape of tensor do not adapting with output operand";
							const auto& target_shapes = std::vector<uint32_t>{ 1, (uint32_t)operand_shapes.at(1), 1 };
							output_tensors->datas.at(j)->Reshape(target_shapes);
						}
					}
					else 
					{
						// current shape is 3
						if (tensor_shapes.at(0) != 1 || tensor_shapes.at(1) != operand_shapes.at(1) || tensor_shapes.at(2) != operand_shapes.at(2)) 
						{
							DLOG(WARNING) << "The shape of tensor do not adapting with output operand";
							const auto& target_shapes = std::vector<uint32_t>{ 1, (uint32_t)operand_shapes.at(1), (uint32_t)operand_shapes.at(2) };
							output_tensors->datas.at(j)->Reshape(target_shapes);
						}
					}
				}
			}
		}
	}

}  // namespace kuiper_infer
