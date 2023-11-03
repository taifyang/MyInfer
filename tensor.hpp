#pragma once

#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>


namespace my_infer 
{
	template <typename T = float>
	class Tensor {};

	template <>
	class Tensor<float>
	{
	public:
		explicit Tensor() = default;

		/**
		 * ��������
		 * @param channels ������ͨ����
		 * @param rows ����������
		 * @param cols ����������
		 */
		explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

		/**
		 * ����һ��һά����
		 * @param size һά������Ԫ�صĸ���
		 */
		explicit Tensor(uint32_t size);

		/**
		 * ����һ����ά����
		 * @param rows ��ά�����ĸ߶�
		 * @param cols ��ά�����Ŀ��
		 */
		explicit Tensor(uint32_t rows, uint32_t cols);

		/**
		 * ��������
		 * @param shapes ������ά��
		 */
		explicit Tensor(const std::vector<uint32_t>& shapes);

		Tensor(const Tensor& tensor);

		Tensor(Tensor&& tensor) noexcept;

		Tensor<float>& operator=(Tensor&& tensor) noexcept;

		Tensor<float>& operator=(const Tensor& tensor);

		/**
		 * ��������������
		 * @return ����������
		 */
		uint32_t rows() const;

		/**
		 * ��������������
		 * @return ����������
		 */
		uint32_t cols() const;

		/**
		 * ����������ͨ����
		 * @return ������ͨ����
		 */
		uint32_t channels() const;

		/**
		 * ����������Ԫ�ص�����
		 * @return ������Ԫ������
		 */
		uint32_t size() const;

		/**
		 * ���������еľ�������
		 * @param data ����
		 */
		void set_data(const Eigen::Tensor<float, 3>& data);

		/**
		 * ���������Ƿ�Ϊ��
		 * @return �����Ƿ�Ϊ��
		 */
		bool empty() const;

		/**
		 * ����������offsetλ�õ�Ԫ��
		 * @param offset ��Ҫ���ʵ�λ��
		 * @return offsetλ�õ�Ԫ��
		 */
		float index(uint32_t offset) const;

		/**
		 * ����������offsetλ�õ�Ԫ��
		 * @param offset ��Ҫ���ʵ�λ��
		 * @return offsetλ�õ�Ԫ��
		 */
		float& index(uint32_t offset);

		/**
		 * �����ĳߴ��С
		 * @return �����ĳߴ��С
		 */
		std::vector<uint32_t> shapes() const;

		/**
		 * ������ʵ�ʳߴ��С
		 * @return ������ʵ�ʳߴ��С
		 */
		const std::vector<uint32_t>& raw_shapes() const;

		/**
		 * ���������е�����
		 * @return �����е�����
		 */
		Eigen::Tensor<float, 3>& data();

		/**
		 * ���������е�����
		 * @return �����е�����
		 */
		const Eigen::Tensor<float, 3>& data() const;

		/**
		 * ����������channelͨ���е�����
		 * @param channel ��Ҫ���ص�ͨ��
		 * @return ���ص�ͨ��
		 */
		Eigen::Tensor<float, 3> slice(uint32_t channel);

		/**
		 * ����������channelͨ���е�����
		 * @param channel ��Ҫ���ص�ͨ��
		 * @return ���ص�ͨ��
		 */
		const Eigen::Tensor<float, 3>& slice(uint32_t channel) const;

		/**
		 * �����ض�λ�õ�Ԫ��
		 * @param channel ͨ��
		 * @param row ����
		 * @param col ����
		 * @return �ض�λ�õ�Ԫ��
		 */
		float at(uint32_t channel, uint32_t row, uint32_t col) const;

		/**
		 * �����ض�λ�õ�Ԫ��
		 * @param channel ͨ��
		 * @param row ����
		 * @param col ����
		 * @return �ض�λ�õ�Ԫ��
		 */
		float& at(uint32_t channel, uint32_t row, uint32_t col);

		/**
		 * �������
		 * @param pads ��������ĳߴ�
		 * @param padding_value �������
		 */
		void Padding(const std::vector<uint32_t>& pads, float padding_value);

		/**
		 * ʹ��valueֵȥ��ʼ������
		 * @param value
		 */
		void Fill(float value);

		/**
		 * ʹ��values�е����ݳ�ʼ������
		 * @param values ������ʼ������������
		 */
		void Fill(std::vector<float>& values, bool row_major = true);

		/**
		 * ����Tensor�ڵ���������
		 * @param row_major �Ƿ����������е�
		 * @return Tensor�ڵ���������
		 */
		std::vector<float> values(bool row_major = true);

		/**
		 * �Գ���1��ʼ������
		 */
		void Ones();

		/**
		 * �����ֵ��ʼ������
		 */
		void Rand();

		/**
		 * ��ӡ����
		 */
		void Show(bool row_major = true);

		/**
		 * ������ʵ�ʳߴ��С��Reshape pytorch����
		 * @param shapes ������ʵ�ʳߴ��С
		 * @param row_major ���������������������reshape
		 */
		void Reshape(const std::vector<uint32_t>& shapes, bool row_major = false);

		/**
		 * չ������
		 */
		void Flatten(bool row_major = false);

		/**
		 * �������е�Ԫ�ؽ��й���
		 * @param filter ���˺���
		 */
		void Transform(const std::function<float(float)>& filter);

		/**
		 * �������ݵ�ԭʼָ��
		 * @return �������ݵ�ԭʼָ��
		 */
		float* raw_ptr();

		/**
		 * �������ݵ�ԭʼָ��
		 * @param offset ����ָ���ƫ����
		 * @return �������ݵ�ԭʼָ��
		 */
		float* raw_ptr(uint32_t offset);

		/**
		 * ���ص�index���������ʼ��ַ
		 * @param index ��index������
		 * @return ��index���������ʼ��ַ
		 */
		float* matrix_raw_ptr(uint32_t index);

		std::shared_ptr<Tensor<float>> Clone();

		void write_tensor(std::string txt);

	private:
		std::vector<uint32_t> raw_shapes_;  // �������ݵ�ʵ�ʳߴ��С
		Eigen::Tensor<float, 3> data_;		// ��������
	};

	using ftensor = Tensor<float>;
	using sftensor = std::shared_ptr<Tensor<float>>;

	void write_matrix(std::string txt, Eigen::MatrixXf matrix);
}  // namespace my_infer