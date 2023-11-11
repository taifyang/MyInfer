#include "tensor.hpp"
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <numeric>
//#include <iostream>
//#include <fstream>
//#include <iomanip>

namespace my_infer
{
	Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) 
	{
		data_ = Eigen::Tensor<float, 3>(rows, cols, channels);
		if (channels == 1 && rows == 1) 
			this->raw_shapes_ = std::vector<uint32_t>{ cols };
		else if (channels == 1) 
			this->raw_shapes_ = std::vector<uint32_t>{ rows, cols };
		else 
			this->raw_shapes_ = std::vector<uint32_t>{ channels, rows, cols };
	}

	Tensor<float>::Tensor(uint32_t size) 
	{
		data_ = Eigen::Tensor<float, 3>(1, size, 1);
		this->raw_shapes_ = std::vector<uint32_t>{ size };
	}

	Tensor<float>::Tensor(uint32_t rows, uint32_t cols) 
	{
		data_ = Eigen::Tensor<float, 3>(rows, cols, 1);
		if (rows == 1) 
			this->raw_shapes_ = std::vector<uint32_t>{ cols };
		else 
			this->raw_shapes_ = std::vector<uint32_t>{ rows, cols };
	}

	Tensor<float>::Tensor(const std::vector<uint32_t>& shapes) 
	{
		CHECK(!shapes.empty() && shapes.size() <= 3);

		uint32_t remaining = 3 - shapes.size();
		std::vector<uint32_t> shapes_(3, 1);
		std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

		uint32_t channels = shapes_.at(0);
		uint32_t rows = shapes_.at(1);
		uint32_t cols = shapes_.at(2);

		data_ = Eigen::Tensor<float, 3>(rows, cols, channels);
		if (channels == 1 && rows == 1) 
			this->raw_shapes_ = std::vector<uint32_t>{ cols };
		else if (channels == 1) 
			this->raw_shapes_ = std::vector<uint32_t>{ rows, cols };
		else 
			this->raw_shapes_ = std::vector<uint32_t>{ channels, rows, cols };
	}

	Tensor<float>::Tensor(const Tensor& tensor) 
	{
		if (this != &tensor) 
		{
			this->data_ = tensor.data_;
			this->raw_shapes_ = tensor.raw_shapes_;
		}
	}

	Tensor<float>::Tensor(Tensor<float>&& tensor) noexcept 
	{
		if (this != &tensor) 
		{
			this->data_ = std::move(tensor.data_);
			this->raw_shapes_ = tensor.raw_shapes_;
		}
	}

	Tensor<float>& Tensor<float>::operator=(const Tensor& tensor) 
	{
		if (this != &tensor) 
		{
			this->data_ = tensor.data_;
			this->raw_shapes_ = tensor.raw_shapes_;
		}
		return *this;
	}

	Tensor<float>& Tensor<float>::operator=(Tensor<float>&& tensor) noexcept
	{
		if (this != &tensor) 
		{
			this->data_ = std::move(tensor.data_);
			this->raw_shapes_ = tensor.raw_shapes_;
		}
		return *this;
	}

	uint32_t Tensor<float>::rows() const
	{
		return this->data_.dimension(0);
	}

	uint32_t Tensor<float>::cols() const 
	{
		return this->data_.dimension(1);
	}

	uint32_t Tensor<float>::channels() const 
	{
		return this->data_.dimension(2);
	}

	uint32_t Tensor<float>::size() const 
	{
		return this->data_.size();
	}

	void Tensor<float>::set_data(const Eigen::Tensor<float, 3>& data)
	{
		CHECK(data.dimension(0) == this->data_.dimension(0)) << data.dimension(0) << " != " << this->data_.dimension(0);
		CHECK(data.dimension(1) == this->data_.dimension(1)) << data.dimension(1) << " != " << this->data_.dimension(1);
		CHECK(data.dimension(2) == this->data_.dimension(2)) << data.dimension(2) << " != " << this->data_.dimension(2);
		this->data_ = std::move(data);
	}

	//bool Tensor<float>::empty() const { return this->data_.empty(); }

	float Tensor<float>::index(uint32_t offset) const 
	{
		CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
		return this->data_(offset);
	}

	float& Tensor<float>::index(uint32_t offset) 
	{
		CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
		return this->data_(offset);
	}

	std::vector<uint32_t> Tensor<float>::shapes() const 
	{
		return { this->channels(), this->rows(), this->cols() };
	}

	Eigen::Tensor<float, 3>& Tensor<float>::data() 
	{ 
		return this->data_; 
	}

	const Eigen::Tensor<float, 3>& Tensor<float>::data() const 
	{ 
		return this->data_; 
	}

	Eigen::Tensor<float, 3> Tensor<float>::slice(uint32_t channel) 
	{
		CHECK_LT(channel, this->channels());
		Eigen::array<Eigen::DenseIndex, 3> offsets = { 0, 0, channel };
		Eigen::array<Eigen::DenseIndex, 3> extents = { this->data_.dimension(0), this->data_.dimension(1), 1 };
		return this->data_.slice(offsets, extents);
	}

	const Eigen::Tensor<float, 3> Tensor<float>::slice(uint32_t channel) const 
	{
		CHECK_LT(channel, this->channels());
		Eigen::array<Eigen::DenseIndex, 3> offsets = { 0, 0, channel };
		Eigen::array<Eigen::DenseIndex, 3> extents = { this->data_.dimension(0), this->data_.dimension(1), 1 };
		return this->data_.slice(offsets, extents);
	}

	float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const 
	{
		CHECK_LT(row, this->rows());
		CHECK_LT(col, this->cols());
		CHECK_LT(channel, this->channels());
		return this->data_(row, col, channel);
	}

	float& Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) 
	{
		CHECK_LT(row, this->rows());
		CHECK_LT(col, this->cols());
		CHECK_LT(channel, this->channels());
		return this->data_(row, col, channel);
	}

	void Tensor<float>::Padding(const std::vector<uint32_t> & pads,	float padding_value) 
	{
		CHECK_EQ(pads.size(), 4);

		uint32_t pad_rows0 = pads.at(0);  // up
		uint32_t pad_rows1 = pads.at(1);  // bottom
		uint32_t pad_cols0 = pads.at(2);  // left
		uint32_t pad_cols1 = pads.at(3);  // right

		auto d0 = std::make_pair(pad_rows0, pad_rows1);
		auto d1 = std::make_pair(pad_cols0, pad_cols1);
		auto d2 = std::make_pair(0, 0);

		Eigen::array<std::pair<int, int>, 3> dims{ d0, d1, d2 };
		Eigen::Tensor<float, 3> padded = this->data_.pad(dims, padding_value);
		this->data_ = padded;
		this->raw_shapes_ = std::vector<uint32_t>{ this->channels(), this->rows(), this->cols() };
	}

	void Tensor<float>::Fill(float value) 
	{
		this->data_.setConstant(value);
	}

	void Tensor<float>::Fill(std::vector<float> & values, bool row_major)
	{
		const uint32_t total_elems = this->data_.size();
		CHECK_EQ(values.size(), total_elems);

		if (row_major) 
		{
			this->raw_shapes_ = { this->channels(), this->rows(), this->cols() };

			uint32_t index = 0;
			for (uint32_t i = 0; i < this->channels(); ++i)
			{
				for (uint32_t j = 0; j < this->rows(); ++j)
				{
					for (uint32_t k = 0; k < this->cols(); ++k)
					{
						this->at(i, j, k) = values[index++];
					}
				}
			}
		}
		else 
		{
			std::copy(values.begin(), values.end(), this->raw_ptr());
		}	
	}

	void Tensor<float>::Show(bool row_major) 
	{
		if (row_major) 
		{
			for (uint32_t i = 0; i < this->channels(); ++i) 
			{
				LOG(INFO) << "Channel: " << i;
				for (uint32_t j = 0; j < this->rows(); ++j) 
				{
					for (uint32_t k = 0; k < this->cols(); ++k) 
					{
						printf("%f\t", this->at(i, j, k));
					}
					printf("\n");
				}
			}
		}
		else 
		{
			Eigen::Tensor<float, 3, Eigen::RowMajor> temp = this->data_.swap_layout();
			for (uint32_t i = 0; i < this->channels(); ++i) 
			{
				LOG(INFO) << "Channel: " << i;
				for (uint32_t j = 0; j < this->cols(); ++j) 
				{
					for (uint32_t k = 0; k < this->rows(); ++k) 
					{
						printf("%f\t", temp(i, j, k));
					}
					printf("\n");
				}
			}
		}
	}

	void Tensor<float>::Flatten(bool row_major)
	{
		const uint32_t size = this->data_.size();
		this->Reshape({ size });
	}

	void Tensor<float>::Rand() 
	{
		this->data_.setRandom();
	}

	void Tensor<float>::Ones() 
	{
		this->Fill(1.f);
	}

	//void Tensor<float>::Transform(const std::function<float(float)> & filter) 
	//{
	//	this->data_.transform(filter);
	//}

	const std::vector<uint32_t>& Tensor<float>::raw_shapes() const 
	{
		CHECK(!this->raw_shapes_.empty());
		CHECK_LE(this->raw_shapes_.size(), 3);
		CHECK_GE(this->raw_shapes_.size(), 1);
		return this->raw_shapes_;
	}

	void Tensor<float>::Reshape(const std::vector<uint32_t> & shapes, bool row_major)
	{
		CHECK(!shapes.empty());
		const uint32_t origin_size = this->size();
		const uint32_t current_size = std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies<int>());
		CHECK(shapes.size() <= 3);
		CHECK(current_size == origin_size);

		std::vector<float> values(this->data_.size());
		uint32_t index = 0;

		if (row_major)	
		{
			for (uint32_t i = 0; i < this->channels(); i++)
			{
				for (uint32_t j = 0; j < this->rows(); j++)
				{
					for (uint32_t k = 0; k < this->cols(); k++)
					{
						values[index++] = this->at(i, j, k);
					}
				}
			}

			for (uint32_t i = 0; i < shapes.at(0); i++)
			{
				uint32_t w = 1, h = 1;
				if (shapes.size() == 3)
				{
					w = shapes.at(2);
					h = shapes.at(1);
				}
				else if (shapes.size() == 2)
				{
					h = shapes.at(1);
				}

				std::vector<float> v(values.begin() + i * w * h, values.begin() + (i + 1) * w * h);
				Eigen::Map<Eigen::MatrixXf> matrix(v.data(), w, h);
				Eigen::MatrixXf matrix_trans = matrix.transpose();
				Eigen::Map<Eigen::RowVectorXf> v_trans(matrix_trans.data(), matrix_trans.size());
				std::copy(v_trans.begin(), v_trans.end(), values.begin() + i * w * h);
			}
		}
		else			
		{
			for (uint32_t i = 0; i < this->channels(); i++)
			{
				for (uint32_t j = 0; j < this->cols(); j++)
				{
					for (uint32_t k = 0; k < this->rows(); k++)
					{
						values[index++] = this->at(i, k, j);
					}
				}
			}
		}

		if (shapes.size() == 3) 
		{
			this->raw_shapes_ = { shapes.at(0), shapes.at(1), shapes.at(2) }; //channel row col
			this->data_ = Eigen::Tensor<float, 3>(Eigen::TensorMap<Eigen::Tensor<float, 3>>(values.data(), shapes.at(1), shapes.at(2), shapes.at(0)));//row col channel
		}
		else if (shapes.size() == 2) 
		{
			this->raw_shapes_ = { shapes.at(0), shapes.at(1) }; //row col
			this->data_ = Eigen::Tensor<float, 3>(Eigen::TensorMap<Eigen::Tensor<float, 3>>(values.data(), shapes.at(0), shapes.at(1), 1));//row col 1
		}
		else 
		{
			this->raw_shapes_ = { shapes.at(0) };	//row
			this->data_ = Eigen::Tensor<float, 3>(Eigen::TensorMap<Eigen::Tensor<float, 3>>(values.data(), shapes.at(0), 1, 1));//row 1 1
		}
	}

	float* Tensor<float>::raw_ptr() 
	{
		return this->data_.data();
	}

	float* Tensor<float>::raw_ptr(uint32_t offset) 
	{
		const uint32_t size = this->size();
		CHECK(!this->data_.size());
		CHECK_LT(offset, size);
		return this->data_.data() + offset;
	}

	std::vector<float> Tensor<float>::values(bool row_major)
	{
		std::vector<float> values(this->data_.size(), 0);
		uint32_t index = 0;
		if (row_major) 
		{
			for (uint32_t i = 0; i < this->channels(); i++)
			{
				for (uint32_t j = 0; j < this->rows(); j++)
				{
					for (uint32_t k = 0; k < this->cols(); k++)
					{
						values[index++] = this->at(i, j, k);
					}
				}
			}
		}
		else 
		{
			for (uint32_t i = 0; i < this->channels(); i++)
			{
				for (uint32_t j = 0; j < this->rows(); j++)
				{
					for (uint32_t k = 0; k < this->cols(); k++)
					{
						values[index++] = this->at(i, j, k);
					}
				}
			}
		}
		return values;
	}

	std::shared_ptr<Tensor<float>> Tensor<float>::Clone() 
	{
		return std::make_shared<Tensor>(*this);
	}

	//void Tensor<float>::write_tensor(std::string txt)
	//{
	//	std::fstream fout(txt, 'w');
	//	for (uint32_t i = 0; i < this->channels(); ++i)
	//	{
	//		for (uint32_t j = 0; j < this->rows(); ++j)
	//		{
	//			for (uint32_t k = 0; k < this->cols(); ++k)
	//			{
	//				fout << std::fixed << std::setprecision(3) << this->at(i, j, k) << "\t";
	//			}
	//			fout << std::endl;
	//		}
	//	}
	//	fout.close();
	//}

	//void write_matrix(std::string txt, Eigen::MatrixXf matrix)
	//{
	//	std::fstream fout(txt, 'w');
	//	for (uint32_t i = 0; i < matrix.rows(); ++i)
	//	{
	//		for (uint32_t j = 0; j < matrix.cols(); ++j)
	//		{
	//			fout << std::fixed << std::setprecision(2) << matrix(i, j) << "\t";
	//		}
	//		fout << std::endl;
	//	}
	//	fout.close();
	//}
} 

