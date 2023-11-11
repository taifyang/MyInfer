#include <iostream>
#include <iomanip>
#include <memory>
#include <queue>
#include <deque>
#include <utility>
#include "layer_factory.hpp"
#include "runtime_ir.hpp"


namespace my_infer 
{
	RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path): param_path_(std::move(param_path)), bin_path_(std::move(bin_path)) {}

	void RuntimeGraph::set_bin_path(const std::string& bin_path) 
	{
		this->bin_path_ = bin_path;
	}

	void RuntimeGraph::set_param_path(const std::string& param_path) 
	{
		this->param_path_ = param_path;
	}

	const std::string& RuntimeGraph::param_path() const 
	{
		return this->param_path_;
	}

	const std::string& RuntimeGraph::bin_path() const 
	{ 
		return this->bin_path_; 
	}

	bool RuntimeGraph::Init() 
	{
		if (this->bin_path_.empty() || this->param_path_.empty())
		{
			LOG(ERROR) << "The bin path or param path is empty";
			return false;
		}

		this->graph_ = std::make_unique<pnnx::Graph>();
		int load_result = this->graph_->load(param_path_, bin_path_);
		if (load_result != 0) 
		{
			LOG(ERROR) << "Load param path and bin path error: " << param_path_ << " " << bin_path_;
			return false;
		}

		std::vector<pnnx::Operator*> operators = this->graph_->ops;
		if (operators.empty()) 
		{
			LOG(ERROR) << "Can not read the layers' define";
			return false;
		}

		this->operators_.clear();
		for (const pnnx::Operator* op : operators) 
		{
			if (!op) 
			{
				LOG(ERROR) << "Meet the empty node";
				continue;
			}
			else 
			{
				std::shared_ptr<RuntimeOperator> runtime_operator = std::make_shared<RuntimeOperator>();
				// ��ʼ�����ӵ�����
				runtime_operator->name = op->name;
				runtime_operator->type = op->type;

				// ��ʼ�������е�input
				const std::vector<pnnx::Operand*>& inputs = op->inputs;
				if (!inputs.empty()) 
				{
					InitGraphOperatorsInput(inputs, runtime_operator);
				}

				// ��¼���operand�е�����
				const std::vector<pnnx::Operand*>& outputs = op->outputs;
				if (!outputs.empty()) 
				{
					InitGraphOperatorsOutput(outputs, runtime_operator);
				}

				// ��ʼ�������е�attribute(Ȩ��)
				const std::map<std::string, pnnx::Attribute>& attrs = op->attrs;
				if (!attrs.empty()) 
				{
					InitGraphAttrs(attrs, runtime_operator);
				}

				// ��ʼ�������е�parameter
				const std::map<std::string, pnnx::Parameter>& params = op->params;
				if (!params.empty()) 
				{
					InitGraphParams(params, runtime_operator);
				}
				this->operators_.push_back(runtime_operator);
			}
		}

		graph_state_ = GraphState::NeedBuild;
		return true;
	}

	void RuntimeGraph::Build(const std::string& input_name, const std::string& output_name) 
	{
		if (graph_state_ == GraphState::NeedInit) 
		{
			bool init_graph = Init();
			LOG_IF(FATAL, !init_graph) << "Init graph failed!";
		}

		CHECK(graph_state_ >= GraphState::NeedBuild) << "Graph status error, current state is " << int(graph_state_);
		LOG_IF(FATAL, this->operators_.empty()) << "Graph operators is empty, may be no init";

		if (graph_state_ == GraphState::Complete) 
		{
			return;
		}

		// ����ͼ��ϵ
		for (const auto& current_op : this->operators_) 
		{
			// ��ȡ��ǰ�ڵ�����к�̽ڵ�names
			const std::vector<std::string>& output_names = current_op->output_names;
			for (const auto& next_op : this->operators_) 
			{
				if (next_op == current_op) 
				{
					continue;
				}
				// �������ڵ��name���ϵ�ǰ�ڵ�ĺ�̽ڵ�names�����������ڵ���Ϊ��ǰ�ڵ�ĺ��
				if (std::find(output_names.begin(), output_names.end(), next_op->name) != output_names.end()) 
				{
					current_op->output_operators.insert({ next_op->name, next_op });
				}
			}
		}

		this->input_operators_maps_.clear();
		this->output_operators_maps_.clear();
		for (const auto& kOperator : this->operators_) 
		{
			if (kOperator->type == "pnnx.Input") 
			{
				this->input_operators_maps_.insert({ kOperator->name, kOperator });
			}
			else if (kOperator->type == "pnnx.Output") 
			{
				this->output_operators_maps_.insert({ kOperator->name, kOperator });
			}
			else 
			{
				std::shared_ptr<Layer> layer = RuntimeGraph::CreateLayer(kOperator);
				CHECK(layer != nullptr) << "Layer create failed!";
				if (layer)
				{
					kOperator->layer = layer;
				}
			}
		}
		RuntimeOperatorUtils::InitOperatorInput(operators_);
		RuntimeOperatorUtils::InitOperatorOutput(graph_->ops, operators_);
		graph_state_ = GraphState::Complete;
		input_name_ = input_name;
		output_name_ = output_name;
		if (graph_ != nullptr)
		{
			graph_.reset();
			graph_ = nullptr;
		}
	}

	std::vector<std::shared_ptr<Tensor<float>>> RuntimeGraph::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, bool debug)
	{
		// ��鵱ǰ��ִ��ͼ�Ƿ��Ѿ���ʼ�����
		if (graph_state_ < GraphState::Complete)
		{
			LOG(FATAL) << "Graph need be build!";
		}
		CHECK(graph_state_ == GraphState::Complete) << "Graph status error, current state is " << int(graph_state_);

		// �ҵ�ͼ�е���������
		std::shared_ptr<RuntimeOperator> input_op;
		if (input_operators_maps_.find(input_name_) == input_operators_maps_.end()) 
		{
			LOG(FATAL) << "Can not find the input node: " << input_name_;
		}
		else 
		{
			input_op = input_operators_maps_.at(input_name_);
		}

		// �ҵ�ͼ�е��������
		std::shared_ptr<RuntimeOperator> output_op;
		if (output_operators_maps_.find(output_name_) == output_operators_maps_.end()) 
		{
			LOG(FATAL) << "Can not find the output node: " << input_name_;
		}
		else 
		{
			output_op = output_operators_maps_.at(output_name_);
		}

		// ������������һ��Ψһ
		// ִ�ж����������������
		std::deque<std::shared_ptr<RuntimeOperator>> operator_queue;
		operator_queue.push_back(input_op);
		std::map<std::string, double> run_duration_infos;  /// ����ʱ��ͳ��

		if (debug) 
		{
			LOG(INFO) << "Batch Size:" << inputs.size();
			for (int i = 0; i < inputs.size(); ++i) 
			{
				LOG(INFO) << "Input Rows: " << inputs.at(i)->rows() << " Cols: " << inputs.at(i)->cols() << " Channels: " << inputs.at(i)->channels();
			}
			LOG(INFO) << "Inference starting...";
			LOG(INFO) << "--------------------------------------------------" << "\n";
		}

		while (!operator_queue.empty()) 
		{
			// �õ�ִ�ж����еĵ�ǰ�ڵ�
			std::shared_ptr<RuntimeOperator> current_op = operator_queue.front();
			operator_queue.pop_front();

			if (!current_op || current_op == output_op)
			{
				if (debug) 
				{
					LOG(INFO) << "Model Inference End";
				}
				break;
			}

			// �����ǰ�ڵ�Ϊ����ڵ㣬������inputsֱ�ӿ�������̽ڵ���
			if (current_op == input_op) 
			{
				ProbeNextLayer(current_op, operator_queue, inputs);
			}
			else 
			{
				// �����ǰ�ڵ���������ִ�нڵ㣬����ʹ��checkready������Ƿ����
				std::string current_op_name = current_op->name;
				LOG(INFO) << current_op_name;

				if (!CheckOperatorReady(current_op)) 
				{
					if (operator_queue.empty()) 
					{
						// ��current op�����һ���ڵ��ʱ��˵�����Ѿ����ܱ�ready
						LOG(FATAL) << "Current operator is not ready!";
						break;
					}
					else 
					{
						// ���current op�������һ���ڵ㣬�����б�ready�Ŀ�����
						operator_queue.push_back(current_op);
					}
				}
				// ׼���ڵ�layer��������Ҫ������
				const std::vector<std::shared_ptr<RuntimeOperand>>& input_operand_datas = current_op->input_operands_seq;
				// layer������
				std::vector<std::shared_ptr<Tensor<float>>> layer_input_datas;
				for (const auto& input_operand_data : input_operand_datas)
				{
					for (const auto& input_data : input_operand_data->datas) 
					{
						layer_input_datas.push_back(input_data);
					}
				}

				CHECK(!layer_input_datas.empty()) << current_op->name << " Layer input data is empty";
				CHECK(current_op->output_operands != nullptr && !current_op->output_operands->datas.empty()) << "Layer output data is empty";

				const auto & start = std::chrono::steady_clock::now();
				// ִ��operator���е�layer�������
				// layer�ļ����������current_op->output_operands->datas��
				InferStatus status = current_op->layer->Forward(layer_input_datas, current_op->output_operands->datas);
				if (debug) 
				{
					std::replace_if(current_op_name.begin(), current_op_name.end(), [](char c) { return c == '.'; }, '_');
					const double duration = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - start).count();
					if (run_duration_infos.find(current_op->type) == run_duration_infos.end())
					{
						run_duration_infos.insert({ current_op->type, duration });
					}
					else 
					{
						run_duration_infos.at(current_op->type) += duration;
					}
				}

				CHECK(status == InferStatus::kInferSuccess) << current_op->layer->layer_name() << " layer forward failed, error code: " << int(status);
				const auto copy_start = std::chrono::steady_clock::now();
				// ����ǰlayer�ļ������current_op->output_operands->datas��ֵ����̽ڵ��������
				ProbeNextLayer(current_op, operator_queue, current_op->output_operands->datas);
				const double duration = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - copy_start).count();
				if (debug) 
				{
					if (run_duration_infos.find("Copy") == run_duration_infos.end()) 
					{
						run_duration_infos.insert({ "Copy", duration });
					}
					else 
					{
						run_duration_infos.at("Copy") += duration;
					}
				}
			}
		}

		for (const auto& op : this->operators_) 
		{
			op->meet_num = 0;
		}

		CHECK(output_op->input_operands.size() == 1) << "The graph only support one path to the output node yet!";
		// ����ͼ�����һ���ڵ�����룬��������ͼ�����
		const auto & output_op_input_operand = output_op->input_operands.begin();
		const auto & output_operand = output_op_input_operand->second;
		if (debug) 
		{
			LOG(INFO) << "Model Running Information, Time Cost:";
			double duration_all = 0.;
			for (const auto& run_info : run_duration_infos)
			{
				LOG(INFO) << "OP type: " << run_info.first << " duration: " << run_info.second << " s";
				duration_all += run_info.second;
			}
			LOG(INFO) << "All time cost: " << duration_all << " s";
		}
		return output_operand->datas;
	}

	std::shared_ptr<Layer> RuntimeGraph::CreateLayer(const std::shared_ptr<RuntimeOperator>& op) 
	{
		LOG_IF(FATAL, !op) << "Operator is empty!";
		const auto& layer = LayerRegisterer::CreateLayer(op);
		LOG_IF(FATAL, !layer) << "Layer init failed " << op->type;
		return layer;
	}

	void RuntimeGraph::InitGraphOperatorsInput(const std::vector<pnnx::Operand*>& inputs, const std::shared_ptr<RuntimeOperator>& runtime_operator) 
	{
		for (const pnnx::Operand* input : inputs) 
		{
			if (!input) 
			{
				continue;
			}
			const pnnx::Operator* producer = input->producer;
			std::shared_ptr<RuntimeOperand> runtime_operand = std::make_shared<RuntimeOperand>();
			runtime_operand->name = producer->name;
			runtime_operand->shapes = input->shape;

			switch (input->type) 
			{
			case 1: 
			{
				runtime_operand->type = RuntimeDataType::kTypeFloat32;
				break;
			}
			case 0: 
			{
				runtime_operand->type = RuntimeDataType::kTypeUnknown;
				break;
			}
			default: 
			{
				LOG(FATAL) << "Unknown input operand type: " << input->type;
			}
			}
			runtime_operator->input_operands.insert({ producer->name, runtime_operand });
			runtime_operator->input_operands_seq.push_back(runtime_operand);
		}
	}

	void RuntimeGraph::InitGraphOperatorsOutput(const std::vector<pnnx::Operand*>& outputs, const std::shared_ptr<RuntimeOperator>& runtime_operator) 
	{
		for (const pnnx::Operand* output : outputs) 
		{
			if (!output) 
			{
				continue;
			}
			const auto& consumers = output->consumers;
			for (const auto& c : consumers) 
			{
				runtime_operator->output_names.push_back(c->name);
			}
		}
	}

	void RuntimeGraph::InitGraphParams(const std::map<std::string, pnnx::Parameter>& params, const std::shared_ptr<RuntimeOperator>& runtime_operator) 
	{
		for (const auto& pair : params)
		{
			const std::string& name = pair.first;
			const pnnx::Parameter& parameter = pair.second;
			const int type = parameter.type;
			// ���ݴ����pnnx:params ���б�����ÿ�α����õ�һЩ���ԣ�����������ȥ��ʼ�������RuntimeParameter
			// RuntimeParameter�ٴ�ŵ�RutimeOperator��
			switch (type) 
			{
				case int(RuntimeParameterType::kParameterUnknown) :
				{
					RuntimeParameter* runtime_parameter = new RuntimeParameter;
					// ������������ֺ;������ʵ��
					runtime_operator->params.insert({ name, runtime_parameter });
					break;
				}
				case int(RuntimeParameterType::kParameterBool) : 
				{
					RuntimeParameterBool* runtime_parameter = new RuntimeParameterBool;
					runtime_parameter->value = parameter.b; 
					runtime_operator->params.insert({ name, runtime_parameter });
					break;
				}
				case int(RuntimeParameterType::kParameterInt) : 
				{
					RuntimeParameterInt* runtime_parameter = new RuntimeParameterInt;
					runtime_parameter->value = parameter.i;
					runtime_operator->params.insert({ name, runtime_parameter });
					break;
				}
				case int(RuntimeParameterType::kParameterFloat) : 
				{
					RuntimeParameterFloat* runtime_parameter = new RuntimeParameterFloat;
					runtime_parameter->value = parameter.f;
					runtime_operator->params.insert({ name, runtime_parameter });
					break;
				}
				case int(RuntimeParameterType::kParameterString) : 
				{
					RuntimeParameterString* runtime_parameter = new RuntimeParameterString;
					runtime_parameter->value = parameter.s;
					runtime_operator->params.insert({ name, runtime_parameter });
					break;
				}
				case int(RuntimeParameterType::kParameterIntArray) : 
				{
					RuntimeParameterIntArray* runtime_parameter = new RuntimeParameterIntArray;
					runtime_parameter->value = parameter.ai;
					runtime_operator->params.insert({ name, runtime_parameter });
					break;
				}
				case int(RuntimeParameterType::kParameterFloatArray) :
				{
					RuntimeParameterFloatArray* runtime_parameter = new RuntimeParameterFloatArray;
					runtime_parameter->value = parameter.af;
					runtime_operator->params.insert({ name, runtime_parameter });
					break;
				}
				case int(RuntimeParameterType::kParameterStringArray) : 
				{
					RuntimeParameterStringArray* runtime_parameter = new RuntimeParameterStringArray;
					runtime_parameter->value = parameter.as;
					runtime_operator->params.insert({ name, runtime_parameter });
					break;
				}
				default: 
				{
					LOG(FATAL) << "Unknown parameter type";
				}
			}
		}
	}

	void RuntimeGraph::InitGraphAttrs(const std::map<std::string, pnnx::Attribute>& attrs, const std::shared_ptr<RuntimeOperator>& runtime_operator) {

		for (const auto& pair : attrs) 
		{
			const std::string& name = pair.first;
			const pnnx::Attribute& attr = pair.second;
			switch (attr.type) 
			{
			case 1: 
			{
				std::shared_ptr<RuntimeAttribute> runtime_attribute = std::make_shared<RuntimeAttribute>();
				runtime_attribute->type = RuntimeDataType::kTypeFloat32;
				runtime_attribute->weight_data = attr.data;
				runtime_attribute->shape = attr.shape;
				runtime_operator->attribute.insert({ name, runtime_attribute });
				break;
			}
			default: 
			{
				LOG(FATAL) << "Unknown attribute type";
			}
			}
		}
	}

	bool RuntimeGraph::CheckOperatorReady(const std::shared_ptr<RuntimeOperator>& op) 
	{
		CHECK(op != nullptr);
		CHECK(op->meet_num <= op->input_operands.size());
		if (op->meet_num == op->input_operands.size()) 
			return true;
		else 
			return false;
	}

	void RuntimeGraph::ProbeNextLayer(const std::shared_ptr<RuntimeOperator>& current_op,
		std::deque<std::shared_ptr<RuntimeOperator>>& operator_queue, const std::vector<std::shared_ptr<Tensor<float>>>& layer_output_datas) 
	{
		// ��ǰ�ڵ�ĺ�̽ڵ�next_ops
		const auto& next_ops = current_op->output_operators;
		// �����к�̽ڵ���б���
		for (const auto& next_op : next_ops) 
		{
			// �õ���̽ڵ������next_input_operands
			const auto& next_rt_operator = next_op.second;
			const auto& next_input_operands = next_rt_operator->input_operands;
			// ȷ����̽ڵ������������current_op
			if (next_input_operands.find(current_op->name) != next_input_operands.end()) 
			{
				// �õ���̽ڵ�Ĺ���current_op���������ռ� next_input_datas
				/**
				 * next_input_operands:
				 * {
				 *    ����1 -- current_op.name: current_op��Ӧ������ռ�
				 *    ����2 -- other_op.name: other_op��Ӧ������ռ�
				 * }
				 */
				std::vector<std::shared_ptr<ftensor>>& next_input_datas = next_input_operands.at(current_op->name)->datas;
				// ����ǰcurrent_op�������ֵ��next_input_datas��
				for (int i = 0; i < next_input_datas.size(); ++i) 
				{
					next_input_datas.at(i) = layer_output_datas.at(i);
				}
				// ��̽ڵ�ķ��ʴ�����1
				next_rt_operator->meet_num += 1;
				if (std::find(operator_queue.begin(), operator_queue.end(), next_rt_operator) == operator_queue.end()) 
				{
					// ����̽ڵ��Ƿ��Ѿ�ready��������ִ�ж���
					if (CheckOperatorReady(next_rt_operator)) 
					{
						operator_queue.push_back(next_rt_operator);
					}
				}
			}
		}
	}
}
