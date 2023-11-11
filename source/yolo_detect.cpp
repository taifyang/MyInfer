#include "yolo_detect.hpp"
#include "layer_factory.hpp"
#include "tensor_utils.hpp"
#include <iostream>

namespace my_infer 
{
	YoloDetectLayer::YoloDetectLayer(int32_t stages, 
		int32_t num_classes, 
		const std::vector<float>& strides,
		const std::vector<Eigen::MatrixXf>& anchor_grids,
		const std::vector<Eigen::MatrixXf>& grids,
		const std::vector<std::shared_ptr<ConvolutionLayer>>& conv_layers)
		: Layer("yolo"),
		stages_(stages),
		num_classes_(num_classes),
		strides_(strides),
		anchor_grids_(anchor_grids),
		grids_(grids),
		conv_layers_(conv_layers) {}

	InferStatus YoloDetectLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, std::vector<std::shared_ptr<Tensor<float>>>& outputs)
	{
		if (inputs.empty()) 
		{
			LOG(ERROR) << "The input feature map of yolo detect layer is empty";
			return InferStatus::kInferFailedInputEmpty;
		}

		const uint32_t stages = stages_;
		const uint32_t classes_info = num_classes_ + 5;
		const uint32_t input_size = inputs.size();
		const uint32_t batch_size = outputs.size();

		if (input_size / batch_size != stages_ || input_size % batch_size != 0) 
		{
			LOG(ERROR) << "The input and output number of yolo detect layer is wrong";
			return InferStatus::kInferFailedYoloStageNumberError;
		}

		CHECK(!this->conv_layers_.empty() && this->conv_layers_.size() == stages) << "The convolution layers in yolo detection layer is empty or do not have a correct number";

		std::vector<std::vector<std::shared_ptr<Tensor<float>>>> batches(stages);
		for (uint32_t i = 0; i < input_size; ++i) 
		{
			const uint32_t index = i / batch_size;
			const auto& input = inputs.at(i);
			//input->write_tensor("input.txt");
			if (input == nullptr) 
			{
				LOG(ERROR) << "The input feature map of yolo detect layer is empty";
				return InferStatus::kInferFailedInputEmpty;
			}
			CHECK(index <= batches.size());
			batches.at(index).push_back(input);
		}

		std::vector<std::vector<sftensor>> stage_outputs(stages);
#pragma omp parallel for num_threads(stages)
		for (int stage = 0; stage < stages; ++stage) 
		{
			const std::vector<std::shared_ptr<Tensor<float>>>& stage_input = batches.at(stage);

			CHECK(stage_input.size() == batch_size) << "The number of stage do not equal to batch size";

			std::vector<std::shared_ptr<Tensor<float>>> stage_output(batch_size);
			const auto status = this->conv_layers_.at(stage)->Forward(stage_input, stage_output);

			CHECK(status == InferStatus::kInferSuccess) << "Infer failed, error code: " << int(status);
			CHECK(stage_output.size() == batch_size) << "The number of stage output do not equal to batch size";
			stage_outputs.at(stage) = stage_output;
		}

		uint32_t concat_rows = 0;
		std::vector<std::shared_ptr<Tensor<float>>> zs(stages);
		for (uint32_t stage = 0; stage < stages; ++stage) 
		{
			const std::vector<sftensor> stage_output = stage_outputs.at(stage);
			const uint32_t nx_ = stage_output.front()->rows();
			const uint32_t ny_ = stage_output.front()->cols();
			for (uint32_t i = 0; i < stage_output.size(); ++i) 
			{
				CHECK(stage_output.at(i)->rows() == nx_ && stage_output.at(i)->cols() == ny_);
			}

			std::shared_ptr<Tensor<float>> x_stages_tensor;
			x_stages_tensor = TensorCreate(batch_size, stages * nx_ * ny_, uint32_t(classes_info));

#pragma omp parallel for num_threads(batch_size)
			for (int b = 0; b < batch_size; ++b) 
			{
				const std::shared_ptr<Tensor<float>>& input = stage_output.at(b);
				//input->write_tensor("input.txt");
				CHECK(input != nullptr);
				const uint32_t nx = input->rows();
				const uint32_t ny = input->cols();
				input->Reshape({ stages, uint32_t(classes_info), ny * nx }, true);
				//input->write_tensor("input.txt");

				for (uint32_t i = 0; i < input->size(); i++)
				{
					input->index(i) = 1.f / (1.f + expf(-input->index(i)));
				}
				//input->write_tensor("input.txt");

				for (uint32_t s = 0; s < stages; ++s) 
				{
					for (uint32_t i = 0, r = ny * nx * s; r < ny * nx * (s + 1); r++, i++)
					{
						for (uint32_t j = 0, c = 0; c < classes_info; c++, j++)
						{
							x_stages_tensor->at(b, r, c) = input->at(s, j, i);
						}
					}
				}
				
				Eigen::array<Eigen::DenseIndex, 3> xy_offsets = { 0, 0, b };
				Eigen::array<Eigen::DenseIndex, 3> xy_extents = { x_stages_tensor->rows(), 2, 1 };
				Eigen::Tensor<float, 3> xy = x_stages_tensor->data().slice(xy_offsets, xy_extents);	//4800*2*1
				//std::cout << xy << std::endl;

				Eigen::array<Eigen::DenseIndex, 3> wh_offsets = { 0, 2, b };
				Eigen::array<Eigen::DenseIndex, 3> wh_extents = { x_stages_tensor->rows(), 2, 1 };
				Eigen::Tensor<float, 3> wh = x_stages_tensor->data().slice(wh_offsets, wh_extents);	//4800*2*1
				//std::cout << wh << std::endl;

				Eigen::Tensor<float, 3> grid(grids_[stage].rows(), grids_[stage].cols(), 1);
				std::copy(grids_[stage].data(), grids_[stage].data() + grids_[stage].size(), grid.data());
				Eigen::Tensor<float, 3> temp1 = (xy * 2.0f + grid) * strides_[stage];
				
				Eigen::Tensor<float, 3> anchor(anchor_grids_[stage].rows(), anchor_grids_[stage].cols(), 1);
				std::copy(anchor_grids_[stage].data(), anchor_grids_[stage].data() + anchor_grids_[stage].size(), anchor.data());
				Eigen::Tensor<float, 3> temp2 = (wh * 2.0f).pow(2.0f) * anchor;

				for (uint32_t r = 0; r < x_stages_tensor->rows(); r++)
				{
					for (uint32_t c = 0; c < 2; c++)
					{
						x_stages_tensor->at(b, r, c) = temp1(r, c, 0);
					}
					for (uint32_t c = 2; c < 4; c++)
					{
						x_stages_tensor->at(b, r, c) = temp2(r, c - 2, 0);
					}
				}
			}
			concat_rows += x_stages_tensor->rows();
			zs.at(stage) = x_stages_tensor;
			//zs.at(stage)->write_tensor("zs.txt");
		}

		uint32_t current_rows = 0;
		std::shared_ptr<Tensor<float>> f1 = std::make_shared<Tensor<float>>(batch_size, concat_rows, classes_info);
		for (const auto& z : zs) 
		{
			for (uint32_t i = 0; i < z->channels(); ++i)
			{
				for (uint32_t j = 0; j < z->rows(); ++j)
				{
					for (uint32_t k = 0; k < z->cols(); ++k)
					{
						f1->at(i, j + current_rows, k) = z->at(i, j, k);
					}
				}
			}
			current_rows += z->rows();
		}
		//f1->write_tensor("f1.txt");

		for (int i = 0; i < f1->channels(); ++i) 
		{
			std::shared_ptr<Tensor<float>> output = outputs.at(i);
			if (output == nullptr) 
			{
				output = std::make_shared<Tensor<float>>(1, concat_rows, classes_info);
				outputs.at(i) = output;
			}
			CHECK(output->rows() == f1->slice(i).dimension(0));
			CHECK(output->cols() == f1->slice(i).dimension(1));
			output->data() = f1->slice(i);
			//output->write_tensor("output.txt");
		}

		return InferStatus::kInferSuccess;
	}

	ParseParameterAttrStatus YoloDetectLayer::GetInstance(const std::shared_ptr<RuntimeOperator> & op, std::shared_ptr<Layer> & yolo_detect_layer) 
	{
		CHECK(op != nullptr) << "Yolo detect operator is nullptr";

		const auto & attrs = op->attribute;
		CHECK(!attrs.empty()) << "Operator attributes is empty!";

		if (attrs.find("pnnx_5") == attrs.end()) 
		{
			LOG(ERROR) << "Can not find the in yolo strides attribute";
			return ParseParameterAttrStatus::kAttrMissingYoloStrides;
		}

		const auto& stages_attr = attrs.at("pnnx_5");
		if (stages_attr->shape.empty())
		{
			LOG(ERROR) << "Can not find the in yolo strides attribute";
			return ParseParameterAttrStatus::kAttrMissingYoloStrides;
		}

		int stages_number = stages_attr->shape.at(0);
		CHECK(stages_number == 3) << "Only support three stages yolo detect head";
		const std::vector<float> & strides = stages_attr->get<float>();
		CHECK(strides.size() == stages_number) << "Stride number is not equal to strides";

		std::vector<std::shared_ptr<ConvolutionLayer>> conv_layers(stages_number);
		int32_t num_classes = -1;
		for (int i = 0; i < stages_number; ++i) 
		{
			const std::string& weight_name = "m." + std::to_string(i) + ".weight";
			if (attrs.find(weight_name) == attrs.end()) 
			{
				LOG(ERROR) << "Can not find the in weight attribute " << weight_name;
				return ParseParameterAttrStatus::kAttrMissingWeight;
			}

			const auto& conv_attr = attrs.at(weight_name);
			const auto& out_shapes = conv_attr->shape;
			CHECK(out_shapes.size() == 4) << "Stage output shape must equal to four";

			const int out_channels = out_shapes.at(0);
			if (num_classes == -1) 
			{
				CHECK(out_channels % stages_number == 0) << "The number of output channel is wrong, it should divisible by stages number";
				num_classes = out_channels / stages_number - 5;
				CHECK(num_classes > 0) << "The number of object classes must greater than zero";
			}
			const int in_channels = out_shapes.at(1);
			const int kernel_h = out_shapes.at(2);
			const int kernel_w = out_shapes.at(3);
			conv_layers.at(i) = std::make_shared<ConvolutionLayer>(out_channels, in_channels, kernel_h, kernel_w, 0, 0, 1, 1, 1);
			const std::vector<float>& weights = conv_attr->get<float>();
			conv_layers.at(i)->set_weights(weights);

			const std::string& bias_name = "m." + std::to_string(i) + ".bias";
			if (attrs.find(bias_name) == attrs.end())
			{
				LOG(ERROR) << "Can not find the in bias attribute";
				return ParseParameterAttrStatus::kAttrMissingBias;
			}
			const auto& bias_attr = attrs.at(bias_name);
			const std::vector<float>& bias = bias_attr->get<float>();
			conv_layers.at(i)->set_bias(bias);
		}

		std::vector<Eigen::MatrixXf> anchor_grids;
		for (int i = 4; i >= 0; i -= 2) 
		{
			const std::string& pnnx_name = "pnnx_" + std::to_string(i);
			const auto& anchor_grid_attr = attrs.find(pnnx_name);
			if (anchor_grid_attr == attrs.end()) 
			{
				LOG(ERROR) << "Can not find the in yolo anchor grides attribute";
				return ParseParameterAttrStatus::kAttrMissingYoloAnchorGrides;
			}
			const auto& anchor_grid = anchor_grid_attr->second;
			const auto& anchor_shapes = anchor_grid->shape;
			std::vector<float> anchor_weight_data = anchor_grid->get<float>();
			CHECK(!anchor_shapes.empty() && anchor_shapes.size() == 5 && anchor_shapes.front() == 1) << "Anchor shape has a wrong size";

			const uint32_t anchor_rows = anchor_shapes.at(1) * anchor_shapes.at(2) * anchor_shapes.at(3);
			const uint32_t anchor_cols = anchor_shapes.at(4);
			CHECK(anchor_weight_data.size() == anchor_cols * anchor_rows) << "Anchor weight has a wrong size";

			Eigen::MatrixXf anchor_grid_matrix = Eigen::Map<Eigen::MatrixXf>(anchor_weight_data.data(), anchor_cols, anchor_rows);
			anchor_grids.emplace_back(anchor_grid_matrix.transpose());
		}

		std::vector<Eigen::MatrixXf> grids;
		std::vector<int32_t> grid_indexes{ 6, 3, 1 };
		for (const auto grid_index : grid_indexes) 
		{
			const std::string& pnnx_name = "pnnx_" + std::to_string(grid_index);
			const auto& grid_attr = attrs.find(pnnx_name);
			if (grid_attr == attrs.end())
			{
				LOG(ERROR) << "Can not find the in yolo grides attribute";
				return ParseParameterAttrStatus::kAttrMissingYoloGrides;
			}

			const auto& grid = grid_attr->second;
			const auto& shapes = grid->shape;
			std::vector<float> weight_data = grid->get<float>();
			CHECK(!shapes.empty() && shapes.size() == 5 && shapes.front() == 1) << "Grid shape has a wrong size";

			const uint32_t grid_rows = shapes.at(1) * shapes.at(2) * shapes.at(3);
			const uint32_t grid_cols = shapes.at(4);
			CHECK(weight_data.size() == grid_cols * grid_rows) << "Grid weight has a wrong size";

			Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(weight_data.data(), grid_cols, grid_rows);
			grids.emplace_back(matrix.transpose());
		}
		yolo_detect_layer = std::make_shared<YoloDetectLayer>(stages_number, num_classes, strides, anchor_grids, grids, conv_layers);
		return ParseParameterAttrStatus::kParameterAttrParseSuccess;
	}

	LayerRegistererWrapper kYoloGetInstance("models.yolo.Detect", YoloDetectLayer::GetInstance);
}  