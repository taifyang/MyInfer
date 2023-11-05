#pragma once

#include <vector>
#include <glog/logging.h>
#include "status_code.hpp"
#include "runtime_datatype.hpp"

namespace my_infer
{
	// ����ͼ�ڵ��������Ϣ
	struct RuntimeAttribute
	{
		std::vector<char> weight_data;							// �ڵ��е�Ȩ�ز���
		std::vector<int> shape;									// �ڵ��е���״��Ϣ
		RuntimeDataType type = RuntimeDataType::kTypeUnknown;	// �ڵ��е���������

		/**
		 * �ӽڵ��м���Ȩ�ز���
		 * @tparam T Ȩ������
		 * @return Ȩ�ز�������
		 */
		template<class T>
		std::vector<T> get()
		{
			// ���ڵ������е�Ȩ������
			CHECK(!weight_data.empty());
			CHECK(type != RuntimeDataType::kTypeUnknown);

			std::vector<T> weights;
			switch (type)
			{
			case RuntimeDataType::kTypeFloat32: // ���ص�����������float
			{
				const bool is_float = std::is_same<T, float>::value;
				CHECK_EQ(is_float, true);
				const uint32_t float_size = sizeof(float);
				CHECK_EQ(weight_data.size() % float_size, 0);
				for (uint32_t i = 0; i < weight_data.size() / float_size; ++i)
				{
					float weight = *((float*)weight_data.data() + i);
					weights.push_back(weight);
				}
				break;
			}
			default:
			{
				LOG(FATAL) << "Unknown weight data type";
			}
			}
			return weights;
		}
	};
}