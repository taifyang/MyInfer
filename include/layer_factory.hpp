#pragma once

#include <map>
#include "layer.hpp"
#include "runtime_op.hpp"

namespace my_infer
{
	class LayerRegisterer
	{
	public:
		typedef ParseParameterAttrStatus(*Creator)(const std::shared_ptr<RuntimeOperator>& op, std::shared_ptr<Layer>& layer);

		typedef std::map<std::string, Creator> CreateRegistry;

		static void RegisterCreator(const std::string& layer_type, const Creator& creator);

		static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<RuntimeOperator>& op);

		static CreateRegistry& Registry();
	};

	class LayerRegistererWrapper
	{
	public:
		LayerRegistererWrapper(const std::string& layer_type, const LayerRegisterer::Creator& creator)
		{
			LayerRegisterer::RegisterCreator(layer_type, creator);
		}
	};
}