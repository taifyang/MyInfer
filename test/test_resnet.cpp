#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "load_data.hpp"
#include "runtime_ir.hpp"
#include "softmax.hpp"

my_infer::sftensor PreProcessImage(cv::Mat& image)
{
	using namespace my_infer;
	assert(!image.empty());

	cv::resize(image, image, cv::Size(224, 224));
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	image.convertTo(image, CV_32FC3, 1. / 255.);
	cv::subtract(image, cv::Scalar(0.485, 0.456, 0.406), image);
	cv::divide(image, cv::Scalar(0.229, 0.224, 0.225), image);

	std::vector<cv::Mat> split_images;
	cv::split(image, split_images);
	uint32_t input_w = 224;
	uint32_t input_h = 224;
	uint32_t input_c = 3;
	sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);

	for (int i = 0; i < split_images.size(); ++i)
	{
		cv::Mat split_image = split_images[i];
		assert(split_image.total() == input_w * input_h);
		const cv::Mat& split_image_t = split_image.t();
		memcpy(input->raw_ptr() + i * split_image.total(), split_image_t.data, sizeof(float) * split_image.total());
	}
	//input->write_tensor("input.txt");

	return input;
}

TEST(test_model, resnet) 
{
	LOG(INFO) << "start test resnet\n";
	using namespace my_infer;

	const std::string& param_path = "./tmp/resnet18_batch1.pnnx.param";
	const std::string& bin_path = "./tmp/resnet18_batch1.pnnx.bin";
	RuntimeGraph graph(param_path, bin_path);
	graph.Build("pnnx_input_0", "pnnx_output_0");
	LOG(INFO) << "Start resnet inference";
	std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 224, 224);
	input->Fill(1.);

	std::vector<std::shared_ptr<Tensor<float>>> inputs;
	inputs.push_back(input);

	std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs, false);
	ASSERT_EQ(outputs.size(), 1);

	auto target = CSVDataLoader::LoadData("./tmp/out.csv");
	auto output = outputs.front();
	//std::cout << output->data() << std::endl;
	ASSERT_EQ(output->size(), target->size());
	for (uint32_t i = 0; i < output->size(); ++i)
	{
		ASSERT_LE(std::abs(output->index(i) - target->index(i)), 1e-5);
	}
}

TEST(test_model, resnet_classify_demo)
{
	LOG(INFO) << "start test resnet_classify_demo\n";
	using namespace my_infer;

	cv::Mat image = cv::imread("./tmp/dog.jpg");
	sftensor input = PreProcessImage(image);

	std::vector<sftensor> inputs;
	inputs.push_back(input);

	const std::string& param_path = "./tmp/resnet18_batch1.pnnx.param";
	const std::string& bin_path = "./tmp/resnet18_batch1.pnnx.bin";
	RuntimeGraph graph(param_path, bin_path);
	graph.Build("pnnx_input_0", "pnnx_output_0");

	const std::vector<sftensor> outputs = graph.Forward(inputs, true);

	const uint32_t batch_size = 1;
	std::vector<sftensor> outputs_softmax(batch_size);
	SoftmaxLayer softmax_layer;
	softmax_layer.Forward(outputs, outputs_softmax);
	assert(outputs_softmax.size() == batch_size);

	for (int i = 0; i < outputs_softmax.size(); ++i)
	{
		const sftensor& output_tensor = outputs_softmax.at(i);
		assert(output_tensor->size() == 1 * 1000);

		float max_prob = -1;
		int max_index = -1;
		for (int j = 0; j < output_tensor->size(); ++j)
		{
			float prob = output_tensor->index(j);
			if (max_prob <= prob)
			{
				max_prob = prob;
				max_index = j;
			}
		}
		printf("class with max prob is %f index %d\n", max_prob, max_index);
	}
}

