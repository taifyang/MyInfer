#include <glog/logging.h>
#include "tensor.hpp"
#include "tensor_utils.hpp"


namespace my_infer 
{
	bool TensorIsSame(const sftensor& a, const sftensor& b, float threshold) 
	{
		CHECK(a != nullptr);
		CHECK(b != nullptr);
		if (a->shapes() != b->shapes()) 
		{
			return false;
		}

		for (size_t i = 0; i < a->size(); i++)
		{
			if (std::fabs(a->index(i) - b->index(i)) > threshold)
				return false;
		}
		return true;
	}

	void TensorElementAdd(const sftensor& tensor1, const sftensor& tensor2, const sftensor& output_tensor) 
	{
		CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);
		if (tensor1->shapes() == tensor2->shapes()) 
		{
			CHECK(tensor1->shapes() == output_tensor->shapes());
			output_tensor->set_data(tensor1->data() + tensor2->data());
		}
		else 
		{
			CHECK(tensor1->channels() == tensor2->channels()) << "Tensors shape are not adapting";
			const auto & [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
			CHECK(output_tensor->shapes() == input_tensor1->shapes() && output_tensor->shapes() == input_tensor2->shapes());
			output_tensor->set_data(input_tensor1->data() + input_tensor2->data());
		}
	}

	void TensorElementMultiply(const sftensor & tensor1, const sftensor & tensor2, const sftensor & output_tensor) 
	{
		CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);
		if (tensor1->shapes() == tensor2->shapes())
		{
			CHECK(tensor1->shapes() == output_tensor->shapes());
			output_tensor->set_data(tensor1->data() * tensor2->data());
		}
		else 
		{
			CHECK(tensor1->channels() == tensor2->channels()) << "Tensors shape are not adapting";
			const auto & [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
			CHECK(output_tensor->shapes() == input_tensor1->shapes() && output_tensor->shapes() == input_tensor2->shapes());
			output_tensor->set_data(input_tensor1->data() * input_tensor2->data());
		}
	}

	sftensor TensorElementAdd(const sftensor & tensor1, const sftensor & tensor2) 
	{
		CHECK(tensor1 != nullptr && tensor2 != nullptr);
		if (tensor1->shapes() == tensor2->shapes()) 
		{
			sftensor output_tensor = TensorCreate(tensor1->shapes());
			output_tensor->set_data(tensor1->data() + tensor2->data());
			return output_tensor;
		}
		else 
		{
			const auto & [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
			CHECK(input_tensor1->shapes() == input_tensor2->shapes());
			sftensor output_tensor = TensorCreate(input_tensor1->shapes());
			output_tensor->set_data(input_tensor1->data() + input_tensor2->data());
			return output_tensor;
		}
	}

	sftensor TensorElementMultiply(const sftensor & tensor1, const sftensor & tensor2) 
	{
		CHECK(tensor1 != nullptr && tensor2 != nullptr);
		if (tensor1->shapes() == tensor2->shapes()) 
		{
			sftensor output_tensor = TensorCreate(tensor1->shapes());
			output_tensor->set_data(tensor1->data() * tensor2->data());
			return output_tensor;
		}
		else 
		{
			CHECK(tensor1->channels() == tensor2->channels()) << "Tensors shape are not adapting";
			const auto & [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
			CHECK(input_tensor1->shapes() == input_tensor2->shapes());
			sftensor output_tensor = TensorCreate(input_tensor1->shapes());
			output_tensor->set_data(input_tensor1->data() * input_tensor2->data());
			return output_tensor;
		}
	}

	sftensor TensorCreate(uint32_t channels, uint32_t rows, uint32_t cols)
	{
		return std::make_shared<Tensor<float>>(channels, rows, cols);
	}

	sftensor TensorCreate(const std::vector<uint32_t> & shapes) 
	{
		CHECK(shapes.size() == 3);
		return TensorCreate(shapes.at(0), shapes.at(1), shapes.at(2));
	}

	std::tuple<sftensor, sftensor> TensorBroadcast(const sftensor & tensor1, const sftensor & tensor2)
	{
		CHECK(tensor1 != nullptr && tensor2 != nullptr);
		if (tensor1->shapes() == tensor2->shapes()) 
		{
			return { tensor1, tensor2 };
		}
		else 
		{
			CHECK(tensor1->channels() == tensor2->channels());
			if (tensor2->rows() == 1 && tensor2->cols() == 1) 
			{
				CHECK(tensor2->size() == tensor2->channels());
				sftensor new_tensor = TensorCreate(tensor2->channels(), tensor1->rows(), tensor1->cols());
				Eigen::array<int, 3> cast = { tensor1->rows(), tensor1->cols(), 1 };
				new_tensor->data() = tensor2->data().broadcast(cast);
				return { tensor1, new_tensor };
			}
			else if (tensor1->rows() == 1 && tensor1->cols() == 1)
			{
				CHECK(tensor1->size() == tensor1->channels());
				sftensor new_tensor = TensorCreate(tensor1->channels(), tensor2->rows(), tensor2->cols());
				Eigen::array<int, 3> cast = { tensor2->rows(), tensor2->cols(), 1 };
				new_tensor->data() = tensor1->data().broadcast(cast);
				return { new_tensor, tensor2 };
			}
			else 
			{
				LOG(FATAL) << "Broadcast shape is not adapting!";
				return { tensor1, tensor2 };
			}
		}
	}
}  