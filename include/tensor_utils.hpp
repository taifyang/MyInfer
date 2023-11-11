#pragma once

#include "tensor.hpp"

namespace my_infer 
{
	/**
	 * ������������״�ϵ���չ
	 * @param tenor1 ����1
	 * @param tensor2 ����2
	 * @return ��״һ�µ�����
	 */
	std::tuple<sftensor, sftensor> TensorBroadcast(const sftensor& tensor1, const sftensor& tensor2);

	/**
	 * �����������
	 * @param tensor ����������
	 * @param pads ���Ĵ�С
	 * @param padding_value ����ֵ
	 * @return ���֮�������
	 */
	sftensor TensorPadding(const sftensor& tensor,const std::vector<uint32_t>& pads, float padding_value);

	/**
	 * �Ƚ�tensor��ֵ�Ƿ���ͬ
	 * @param a ��������1
	 * @param b ��������2
	 * @param threshold ����֮�������ֵ
	 * @return �ȽϽ��
	 */
	bool TensorIsSame(const sftensor& a, const sftensor& b, float threshold = 1e-6f);

	/**
	 * �������
	 * @param tensor1 ��������1
	 * @param tensor2 ��������2
	 * @return ������ӵĽ��
	 */
	sftensor TensorElementAdd(const sftensor& tensor1, const sftensor& tensor2);

	/**
	 * �������
	 * @param tensor1 ��������1
	 * @param tensor2 ��������2
	 * @param output_tensor �������
	 */
	void TensorElementAdd(const sftensor& tensor1, const sftensor& tensor2, const sftensor& output_tensor);

	/**
	 * ������
	 * @param tensor1 ��������1
	 * @param tensor2 ��������2
	 * @param output_tensor �������
	 */
	void TensorElementMultiply(const sftensor& tensor1, const sftensor& tensor2, const sftensor& output_tensor);

	/**
	 * �������
	 * @param tensor1 ��������1
	 * @param tensor2 ��������2
	 * @return ������˵Ľ��
	 */
	sftensor TensorElementMultiply(const sftensor& tensor1, const sftensor& tensor2);

	/**
	 * ����һ������
	 * @param channels ͨ������
	 * @param rows ����
	 * @param cols ����
	 * @return �����������
	 */
	sftensor TensorCreate(uint32_t channels, uint32_t rows, uint32_t cols);

	/**
	 * ����һ������
	 * @param shapes ��������״
	 * @return �����������
	 */
	sftensor TensorCreate(const std::vector<uint32_t>& shapes);

	/**
	 * ����һ������������
	 * @param ��Clone������
	 * @return �µ�����
	 */
	sftensor TensorClone(sftensor tensor);
}  // namespace my_infer

