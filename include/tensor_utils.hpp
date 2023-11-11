#pragma once

#include "tensor.hpp"

namespace my_infer 
{
	/**
	 * 对张量进行形状上的扩展
	 * @param tenor1 张量1
	 * @param tensor2 张量2
	 * @return 形状一致的张量
	 */
	std::tuple<sftensor, sftensor> TensorBroadcast(const sftensor& tensor1, const sftensor& tensor2);

	/**
	 * 对张量的填充
	 * @param tensor 待填充的张量
	 * @param pads 填充的大小
	 * @param padding_value 填充的值
	 * @return 填充之后的张量
	 */
	sftensor TensorPadding(const sftensor& tensor,const std::vector<uint32_t>& pads, float padding_value);

	/**
	 * 比较tensor的值是否相同
	 * @param a 输入张量1
	 * @param b 输入张量2
	 * @param threshold 张量之间差距的阈值
	 * @return 比较结果
	 */
	bool TensorIsSame(const sftensor& a, const sftensor& b, float threshold = 1e-6f);

	/**
	 * 张量相加
	 * @param tensor1 输入张量1
	 * @param tensor2 输入张量2
	 * @return 张量相加的结果
	 */
	sftensor TensorElementAdd(const sftensor& tensor1, const sftensor& tensor2);

	/**
	 * 张量相加
	 * @param tensor1 输入张量1
	 * @param tensor2 输入张量2
	 * @param output_tensor 输出张量
	 */
	void TensorElementAdd(const sftensor& tensor1, const sftensor& tensor2, const sftensor& output_tensor);

	/**
	 * 矩阵点乘
	 * @param tensor1 输入张量1
	 * @param tensor2 输入张量2
	 * @param output_tensor 输出张量
	 */
	void TensorElementMultiply(const sftensor& tensor1, const sftensor& tensor2, const sftensor& output_tensor);

	/**
	 * 张量相乘
	 * @param tensor1 输入张量1
	 * @param tensor2 输入张量2
	 * @return 张量相乘的结果
	 */
	sftensor TensorElementMultiply(const sftensor& tensor1, const sftensor& tensor2);

	/**
	 * 创建一个张量
	 * @param channels 通道数量
	 * @param rows 行数
	 * @param cols 列数
	 * @return 创建后的张量
	 */
	sftensor TensorCreate(uint32_t channels, uint32_t rows, uint32_t cols);

	/**
	 * 创建一个张量
	 * @param shapes 张量的形状
	 * @return 创建后的张量
	 */
	sftensor TensorCreate(const std::vector<uint32_t>& shapes);

	/**
	 * 返回一个深拷贝后的张量
	 * @param 待Clone的张量
	 * @return 新的张量
	 */
	sftensor TensorClone(sftensor tensor);
}  // namespace my_infer

