#pragma once

#include <vector>
#include <string>
#include <memory>
#include "status_code.hpp"
#include "runtime_datatype.hpp"
#include "tensor.hpp"

namespace my_infer 
{
	// ����ڵ���������Ĳ�����
	struct RuntimeOperand 
	{
		std::string name;										// ������������
		std::vector<int32_t> shapes;							// ����������״
		std::vector<std::shared_ptr<Tensor<float>>> datas;		// �洢������
		RuntimeDataType type = RuntimeDataType::kTypeUnknown;	// �����������ͣ�һ����float
	};
}