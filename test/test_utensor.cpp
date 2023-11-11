#include <glog/logging.h>
#include <gtest/gtest.h>
#include "tensor.hpp"
#include "tensor_utils.hpp"

TEST(test_tensor, create1)
{
	LOG(INFO) << "start test create1\n";
	using namespace my_infer;

	const std::shared_ptr<ftensor>& tensor_ptr = TensorCreate(2, 3, 4);
	ASSERT_EQ(tensor_ptr->channels(), 2);
	ASSERT_EQ(tensor_ptr->rows(), 3);
	ASSERT_EQ(tensor_ptr->cols(), 4);
}

TEST(test_utensor, create2)
{
	LOG(INFO) << "start test create2\n";
	using namespace my_infer;

	const std::shared_ptr<ftensor>& tensor_ptr = TensorCreate({ 2, 3, 4 });
	ASSERT_EQ(tensor_ptr->channels(), 2);
	ASSERT_EQ(tensor_ptr->rows(), 3);
	ASSERT_EQ(tensor_ptr->cols(), 4);
}

TEST(test_tensor, tensor_broadcast1) 
{
	LOG(INFO) << "start test tensor_broadcast1\n";
	using namespace my_infer;

	const std::shared_ptr<ftensor>& tensor1 = TensorCreate({ 2, 3, 4 });
	const std::shared_ptr<ftensor>& tensor2 = TensorCreate({ 2, 1, 1 });
	tensor1->Rand();
	tensor2->Rand();
	tensor1->Show();
	tensor2->Show();

	const auto& [tensor11, tensor21] = TensorBroadcast(tensor1, tensor2);
	ASSERT_EQ(tensor21->channels(), 2);
	ASSERT_EQ(tensor21->rows(), 3);
	ASSERT_EQ(tensor21->cols(), 4);

	ASSERT_EQ(tensor11->channels(), 2);
	ASSERT_EQ(tensor11->rows(), 3);
	ASSERT_EQ(tensor11->cols(), 4);
	tensor11->Show();
	tensor21->Show();
}

TEST(test_tensor, tensor_broadcast2)
{
	LOG(INFO) << "start test tensor_broadcast2\n";
	using namespace my_infer;

	const std::shared_ptr<ftensor>& tensor1 = TensorCreate({ 2, 3, 4 });
	const std::shared_ptr<ftensor>& tensor2 = TensorCreate({ 2, 1, 1 });
	tensor1->Rand();
	tensor2->Rand();
	tensor1->Show();
	tensor2->Show();

	const auto& [tensor11, tensor21] = TensorBroadcast(tensor1, tensor2);
	ASSERT_EQ(tensor21->channels(), 2);
	ASSERT_EQ(tensor21->rows(), 3);
	ASSERT_EQ(tensor21->cols(), 4);

	ASSERT_EQ(tensor11->channels(), 2);
	ASSERT_EQ(tensor11->rows(), 3);
	ASSERT_EQ(tensor11->cols(), 4);
	tensor11->Show();
	tensor21->Show();
}

TEST(test_tensor, tensor_broadcast3) 
{
	LOG(INFO) << "start test tensor_broadcast3\n";
	using namespace my_infer;

	const std::shared_ptr<ftensor>& tensor1 = TensorCreate({ 2, 1, 1 });
	const std::shared_ptr<ftensor>& tensor2 = TensorCreate({ 2, 3, 4 });
	tensor1->Rand();
	tensor2->Rand();
	tensor1->Show();
	tensor2->Show();

	const auto& [tensor11, tensor21] = TensorBroadcast(tensor1, tensor2);
	ASSERT_EQ(tensor21->channels(), 2);
	ASSERT_EQ(tensor21->rows(), 3);
	ASSERT_EQ(tensor21->cols(), 4);

	ASSERT_EQ(tensor11->channels(), 2);
	ASSERT_EQ(tensor11->rows(), 3);
	ASSERT_EQ(tensor11->cols(), 4);
	tensor11->Show();
	tensor21->Show();
}

TEST(test_tensor, add1) 
{
	LOG(INFO) << "start test add1\n";
	using namespace my_infer;

	const auto& f1 = std::make_shared<Tensor<float>>(2, 3, 4);
	f1->Fill(1.f);
	const auto& f2 = std::make_shared<Tensor<float>>(2, 3, 4);
	f2->Fill(2.f);
	const auto& f3 = TensorElementAdd(f1, f2);
	f1->Show();
	f2->Show();
	f3->Show();
	for (int i = 0; i < f3->size(); ++i) 
		ASSERT_EQ(f3->index(i), 3.f);
}

TEST(test_tensor, add2)
{
	LOG(INFO) << "start test add2\n";
	using namespace my_infer;

	const auto& f1 = std::make_shared<Tensor<float>>(2, 3, 4);
	f1->Fill(1.f);
	const auto& f2 = std::make_shared<Tensor<float>>(2, 1, 1);
	f2->Fill(2.f);
	const auto& f3 = TensorElementAdd(f1, f2);
	f1->Show();
	f2->Show();
	f3->Show();
	for (int i = 0; i < f3->size(); ++i)
		ASSERT_EQ(f3->index(i), 3.f);
}

TEST(test_tensor, add3)
{
	LOG(INFO) << "start test add3\n";
	using namespace my_infer;

	const auto& f1 = std::make_shared<Tensor<float>>(2, 3, 4);
	f1->Fill(1.f);
	const auto& f2 = std::make_shared<Tensor<float>>(2, 1, 1);
	f2->Fill(2.f);
	const auto& f3 = TensorElementAdd(f2, f1);
	f1->Show();
	f2->Show();
	f3->Show();
	for (int i = 0; i < f3->size(); ++i) 
		ASSERT_EQ(f3->index(i), 3.f);
}

TEST(test_tensor, mul1) 
{
	LOG(INFO) << "start test mul1\n";
	using namespace my_infer;

	const auto& f1 = std::make_shared<Tensor<float>>(2, 3, 4);
	f1->Fill(2.f);
	const auto& f2 = std::make_shared<Tensor<float>>(2, 3, 4);
	f2->Fill(3.f);
	const auto& f3 = TensorElementMultiply(f1, f2);
	for (int i = 0; i < f3->size(); ++i) 
		ASSERT_EQ(f3->index(i), 6.f);
}

TEST(test_tensor, mul2)
{
	LOG(INFO) << "start test mul2\n";
	using namespace my_infer;

	const auto& f1 = std::make_shared<Tensor<float>>(2, 3, 4);
	f1->Fill(2.f);
	const auto& f2 = std::make_shared<Tensor<float>>(2, 1, 1);
	f2->Fill(3.f);
	const auto& f3 = TensorElementMultiply(f1, f2);
	for (int i = 0; i < f3->size(); ++i)
		ASSERT_EQ(f3->index(i), 6.f);
}

TEST(test_tensor, mul3)
{
	LOG(INFO) << "start test mul3\n";
	using namespace my_infer;

	const auto& f1 = std::make_shared<Tensor<float>>(2, 3, 4);
	f1->Fill(2.f);
	const auto& f2 = std::make_shared<Tensor<float>>(2, 1, 1);
	f2->Fill(3.f);
	const auto& f3 = TensorElementMultiply(f2, f1);
	for (int i = 0; i < f3->size(); ++i)
		ASSERT_EQ(f3->index(i), 6.f);
}