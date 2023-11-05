#include <gtest/gtest.h>
#include <glog/logging.h>
#include "tensor.hpp"

TEST(test_tensor_size, tensor_size)
{
	LOG(INFO) << "start test tensor_size\n";
	using namespace my_infer;

	Tensor<float> f1(2, 3, 4);
	ASSERT_EQ(f1.channels(), 2);
	ASSERT_EQ(f1.rows(), 3);
	ASSERT_EQ(f1.cols(), 4);
	ASSERT_EQ(f1.size(), 2 * 3 * 4);
}

TEST(test_tensor, tensor_init1D)
{
	LOG(INFO) << "start test tensor_init1D\n";
	using namespace my_infer;

	Tensor<float> f1(2);
	f1.Fill(1.f);
	f1.Show();
	const auto& raw_shapes = f1.raw_shapes();
	const uint32_t size = raw_shapes.at(0);
	ASSERT_EQ(raw_shapes.size(), 1);
	ASSERT_EQ(size, 2);
}

TEST(test_tensor, tensor_init2D)
{
	LOG(INFO) << "start test tensor_init2D\n";
	using namespace my_infer;

	Tensor<float> f1(2, 3);
	f1.Fill(1.f);
	f1.Show(true);

	const auto& raw_shapes = f1.raw_shapes();
	const uint32_t rows = raw_shapes.at(0);
	const uint32_t cols = raw_shapes.at(1);
	ASSERT_EQ(raw_shapes.size(), 2);
	ASSERT_EQ(rows, 2);
	ASSERT_EQ(cols, 3);
}

TEST(test_tensor, tensor_init3D_3)
{
	LOG(INFO) << "start test tensor_init3D_3\n";
	using namespace my_infer;

	Tensor<float> f1(2, 3, 4);
	f1.Fill(1.f);
	f1.Show(true);

	const auto& raw_shapes = f1.raw_shapes();
	const uint32_t channels = raw_shapes.at(0);
	const uint32_t rows = raw_shapes.at(1);
	const uint32_t cols = raw_shapes.at(2);
	ASSERT_EQ(raw_shapes.size(), 3);
	ASSERT_EQ(channels, 2);
	ASSERT_EQ(rows, 3);
	ASSERT_EQ(cols, 4);
}

TEST(test_tensor, tensor_init3D_2)
{
	LOG(INFO) << "start test tensor_init3D_2\n";
	using namespace my_infer;

	Tensor<float> f1(1, 2, 3);
	f1.Fill(1.f);
	f1.Show(true);

	const auto& raw_shapes = f1.raw_shapes();
	const uint32_t rows = raw_shapes.at(0);
	const uint32_t cols = raw_shapes.at(1);
	ASSERT_EQ(raw_shapes.size(), 2);
	ASSERT_EQ(rows, 2);
	ASSERT_EQ(cols, 3);
}

TEST(test_tensor, tensor_init3D_1)
{
	LOG(INFO) << "start test tensor_init3D_1\n";
	using namespace my_infer;

	Tensor<float> f1(1, 1, 3);
	f1.Fill(1.f);
	f1.Show(true);

	const auto& raw_shapes = f1.raw_shapes();
	const uint32_t size = raw_shapes.at(0);
	ASSERT_EQ(raw_shapes.size(), 1);
	ASSERT_EQ(size, 3);
}

TEST(test_tensor, index1)
{
	LOG(INFO) << "start test index1\n";
	using namespace my_infer;

	Tensor<float> f1(2, 3, 4);
	f1.index(3) = 4;
	ASSERT_EQ(f1.index(3), 4);
	f1.Show();
}

TEST(test_tensor, index2)
{
	LOG(INFO) << "start test index2\n";
	using namespace my_infer;

	Tensor<float> f1(2, 3, 4);
	std::vector<float> values;
	for (int i = 0; i < 24; ++i)
	{
		values.push_back(1);
	}
	f1.Fill(values, true);
	for (int i = 0; i < 24; ++i)
	{
		ASSERT_EQ(f1.index(i), 1);
	}
	f1.Show();
}

TEST(test_tensor, slice)
{
	LOG(INFO) << "start test slice\n";
	using namespace my_infer;

	Tensor<float> f1(2, 3, 4);
	f1.Fill(1.f);
	auto slice = f1.slice(0);
	LOG(INFO) << "slice:\n" << slice << std::endl;
	ASSERT_EQ(slice.dimension(0), 3);
	ASSERT_EQ(slice.dimension(1), 4);
	ASSERT_EQ(slice.dimension(2), 1);
}

TEST(test_tensor_values, tensor_values)
{
	LOG(INFO) << "start test tensor_values\n";
	using namespace my_infer;

	Tensor<float> f1(2, 3, 4);
	f1.Rand();
	f1.Show(true);

	LOG(INFO) << "Data in the first channel: " << f1.slice(0);
	LOG(INFO) << "Data in the (0,1,2): " << f1.at(0, 1, 2);
	LOG(INFO) << "Data at index 3: " << f1.index(3);
}

TEST(test_tensor, set_data)
{
	LOG(INFO) << "start test set_data\n";
	using namespace my_infer;

	Tensor<float> f1(2, 3, 4);
	Eigen::Tensor<float, 3> data(3, 4, 2);
	data.setRandom();
	f1.set_data(data);
	LOG(INFO) << "data: \n" << data;
	LOG(INFO) << "Tensor: \n";
	f1.Show(true);
}

TEST(test_tensor, data)
{
	LOG(INFO) << "start test data\n";
	using namespace my_infer;

	Tensor<float> f1(2, 3, 4);
	f1.Fill(1.f);
	Eigen::Tensor<float, 3> data(3, 4, 2);
	data.setZero();
	f1.set_data(data);
	LOG(INFO) << "data: \n" << data;
	LOG(INFO) << "Tensor: \n";
	f1.Show(true);
}

TEST(test_fill_reshape, fill)
{
	LOG(INFO) << "start test fill\n";
	using namespace my_infer;

	Tensor<float> f1(2, 3, 4);
	std::vector<float> values(2 * 3 * 4);
	for (int i = 0; i < 24; ++i)
		values.at(i) = float(i + 1);
	f1.Fill(values);
	f1.Show();

	int index = 1;
	for (size_t i = 0; i < f1.channels(); i++)
	{
		for (size_t j = 0; j < f1.rows(); j++)
		{
			for (size_t k = 0; k < f1.cols(); k++)
			{
				ASSERT_EQ(f1.at(i, j, k), index++);
			}
		}
	}

	auto slice = f1.slice(1);
	LOG(INFO) << "slice:\n" << slice << std::endl;

	values = f1.values();
	for (int i = 0; i < 24; ++i)
		ASSERT_EQ(values.at(i), i + 1);
}

TEST(test_fill_reshape, reshape1)
{
	LOG(INFO) << "start test reshape1\n";
	using namespace my_infer;

	LOG(INFO) << "----------------------Reshape----------------------";
	Tensor<float> f1(2, 3, 4);
	std::vector<float> values(2 * 3 * 4);
	for (int i = 0; i < 24; ++i)
		values.at(i) = float(i + 1);
	f1.Fill(values);
	f1.Show();

	f1.Reshape({ 4, 3, 2 }, false);
	LOG(INFO) << "-------------------After Reshape-------------------";
	f1.Show();
}

TEST(test_fill_reshape, reshape2)
{
	LOG(INFO) << "start test reshape2\n";
	using namespace my_infer;

	LOG(INFO) << "----------------------Reshape----------------------";
	Tensor<float> f1(2, 3, 4);
	std::vector<float> values(2 * 3 * 4);
	for (int i = 0; i < 24; ++i)
		values.at(i) = float(i + 1);
	f1.Fill(values);
	f1.Show();

	f1.Reshape({ 4, 3, 2 }, true);
	LOG(INFO) << "-------------------After Reshape-------------------";
	f1.Show();
}

TEST(test_tensor, copy_construct1)
{
	LOG(INFO) << "start test copy_construct1\n";
	using namespace my_infer;

	Tensor<float> f1(2, 3, 4);
	f1.Rand();
	Tensor<float> f2(f1);
	ASSERT_EQ(f2.channels(), 2);
	ASSERT_EQ(f2.rows(), 3);
	ASSERT_EQ(f2.cols(), 4);
}

TEST(test_tensor, copy_construct2)
{
	LOG(INFO) << "start test copy_construct2\n";
	using namespace my_infer;

	Tensor<float> f1(2, 3, 4);
	Tensor<float> f2(3, 224, 224);
	f2.Rand();
	f1 = f2;
	ASSERT_EQ(f1.channels(), 3);
	ASSERT_EQ(f1.rows(), 224);
	ASSERT_EQ(f1.cols(), 224);
}

TEST(test_tensor, copy_construct3)
{
	LOG(INFO) << "start test copy_construct3\n";
	using namespace my_infer;

	Tensor<float> f1(2, 3, 4);
	Tensor<float> f2(std::vector<uint32_t>{3, 224, 224});
	f2.Rand();
	f1 = f2;
	ASSERT_EQ(f1.channels(), 3);
	ASSERT_EQ(f1.rows(), 224);
	ASSERT_EQ(f1.cols(), 224);
}

TEST(test_tensor, padding1)
{
	LOG(INFO) << "start test padding1\n";
	using namespace my_infer;

	Tensor<float> tensor(2, 3, 4);
	ASSERT_EQ(tensor.channels(), 2);
	ASSERT_EQ(tensor.rows(), 3);
	ASSERT_EQ(tensor.cols(), 4);

	LOG(INFO) << "----------------------before padding----------------------";
	tensor.Fill(1.f);
	tensor.Show();

	LOG(INFO) << "----------------------after padding----------------------";
	tensor.Padding({ 1, 2, 3, 4 }, 0);
	ASSERT_EQ(tensor.rows(), 6);
	ASSERT_EQ(tensor.cols(), 11);
	tensor.Show();

	for (int c = 0; c < tensor.channels(); ++c)
	{
		for (int r = 0; r < tensor.rows(); ++r)
		{
			for (int c_ = 0; c_ < tensor.cols(); ++c_)
			{
				if ((r >= 1 && r <= 3) && (c_ >= 3 && c_ <= 6))
					ASSERT_EQ(tensor.at(c, r, c_), 1.f);
				else
					ASSERT_EQ(tensor.at(c, r, c_), 0.f);
			}
		}
	}
}

TEST(test_tensor, raw_shapes1) 
{
	LOG(INFO) << "start test raw_shapes1\n";
	using namespace my_infer;

	Tensor<float> f1(2, 3, 4);
	f1.Reshape({ 24 });
	const auto& shapes = f1.raw_shapes();
	ASSERT_EQ(shapes.size(), 1);
	ASSERT_EQ(shapes.at(0), 24);
}

TEST(test_tensor, raw_shapes2) 
{
	LOG(INFO) << "start test raw_shapes2\n";
	using namespace my_infer;

	Tensor<float> f1(2, 3, 4);
	f1.Reshape({ 4, 6 });
	const auto& shapes = f1.raw_shapes();
	ASSERT_EQ(shapes.size(), 2);
	ASSERT_EQ(shapes.at(0), 4);
	ASSERT_EQ(shapes.at(1), 6);
}

TEST(test_tensor, raw_shapes3) 
{
	LOG(INFO) << "start test raw_shapes3\n";
	using namespace my_infer;

	Tensor<float> f1(2, 3, 4);
	f1.Reshape({ 4, 3, 2 });
	const auto& shapes = f1.raw_shapes();
	ASSERT_EQ(shapes.size(), 3);
	ASSERT_EQ(shapes.at(0), 4);
	ASSERT_EQ(shapes.at(1), 3);
	ASSERT_EQ(shapes.at(2), 2);
}

TEST(test_tensor, raw_view1)
{
	LOG(INFO) << "start test raw_view1\n";
	using namespace my_infer;

	Tensor<float> f1(2, 3, 4);
	f1.Reshape({ 24 }, true);
	const auto& shapes = f1.raw_shapes();
	ASSERT_EQ(shapes.size(), 1);
	ASSERT_EQ(shapes.at(0), 24);
}

TEST(test_tensor, raw_view2) 
{
	LOG(INFO) << "start test raw_view2\n";
	using namespace my_infer;

	Tensor<float> f1(2, 3, 4);
	f1.Reshape({ 4, 6 }, true);
	const auto& shapes = f1.raw_shapes();
	ASSERT_EQ(shapes.size(), 2);
	ASSERT_EQ(shapes.at(0), 4);
	ASSERT_EQ(shapes.at(1), 6);
}

TEST(test_tensor, raw_view3) 
{
	LOG(INFO) << "start test raw_view3\n";
	using namespace my_infer;

	Tensor<float> f1(2, 3, 4);
	f1.Reshape({ 4, 3, 2 }, true);
	const auto& shapes = f1.raw_shapes();
	ASSERT_EQ(shapes.size(), 3);
	ASSERT_EQ(shapes.at(0), 4);
	ASSERT_EQ(shapes.at(1), 3);
	ASSERT_EQ(shapes.at(2), 2);
}

TEST(test_tensor, clone)
{
	LOG(INFO) << "start test clone\n";
	using namespace my_infer;

	std::shared_ptr<ftensor> f1 = std::make_shared<ftensor>(2, 3, 4);
	f1->Rand();

	const auto& f2 = f1->Clone();
	ASSERT_EQ(f1->size(), f2->size());
	for (int i = 0; i < f2->size(); ++i)
		ASSERT_EQ(f1->index(i), f2->index(i));
}
