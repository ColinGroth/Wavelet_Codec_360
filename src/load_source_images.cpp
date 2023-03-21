#include "glm_include.h"
#include "rendergraph.h"

#include "img_io.h"
#include "scene_config.h"
#include "scene_config_parser.h"

#include "load_source_images.h"
#include "wavelet_transform.h"

#include <filesystem>
#include <set>

namespace
{
Array<String> image_paths;
String        image_directory;
uint32_t      first_image_idx;

}

/**
 * 		Load filename of all frames of our video
 */
void load_entries()
{
	std::set<std::string> entries;
	bool load_directory{std::filesystem::exists(image_directory.begin()) &&
	                    std::filesystem::is_directory(image_directory.begin())};
	if (load_directory) {
		for (auto const &entry :
		     std::filesystem::directory_iterator(image_directory.begin())) {
			if (entry.is_regular_file() &&
			    (entry.path().extension() == ".png" ||
			     entry.path().extension() == ".jpg")) {
#ifdef WIN32
				char *str = new char[4096];
				wcstombs(str, entry.path().c_str(), 4096);
				image_paths.add(str);
#else
				entries.insert(entry.path().string());
#endif
			}
		}
	}

	for (auto &entry : entries) { image_paths.add(entry.c_str()); }
}

/**
 * 		Load the next set of interframes into memory to be considered for
 * 		encoding. Returns false if no more frames available (so all frames are
 * 		transformed already).
 */
bool load_next_textures(App &app, RenderGraph::Graph &graph, Renderer &renderer)
{
	int ifc = cast<WaveletTransformInterface>(
	              app.experiments.get("wavelet_transform").settings[0])
	              .interframe_count;

	if (image_paths.size() < current_image_idx + ifc || ifc == 0) return false;

	/// load all specified textures and convert them to float if necessary
	Array<Texture> float_textures;
	for (int i = 0; i < ifc; ++i) {
		Texture tx;
		printf("Loading %s\n", image_paths[current_image_idx].begin());
		tx.load(image_paths[current_image_idx]);
		int w = tx.width();
		int h = tx.height();

		auto &imgNode = graph.getImageNode("source_image");
		if (w != imgNode.width || h != imgNode.height) {
			printf("Wrong image size!");
			assert(false);
			exit(0);
		}

		/// convert texture to float
		float_textures.add(Texture(w, h, VK_FORMAT_R32G32B32A32_SFLOAT));
		Texture &cur      = float_textures.last();
		uint8_t *src_data = (uint8_t *)tx.data();
		float   *dst_data = (float *)cur.data();
		for (size_t p = 0; p < w * h; p++) {
			float rgb_data[3]   = {float(src_data[p * 3]) / 255.f,
                                 float(src_data[p * 3 + 1]) / 255.f,
                                 float(src_data[p * 3 + 2]) / 255.f};
			dst_data[p * 4]     = rgb_data[0];
			dst_data[p * 4 + 1] = rgb_data[1];
			dst_data[p * 4 + 2] = rgb_data[2];
			dst_data[p * 4 + 3] = 1.0f;
		}
		++current_image_idx;
	}

	/// load textures into image node
	for (size_t i = 0; i < ifc; i++) {
		auto &image = graph.imageStore[first_image_idx + i];
		RenderGraph::uploadTexture(&renderer, float_textures[i], image);
	}

	return true;
}

void init_image_loading(App                &app,
                        GLFWwindow         *window,
                        RenderGraph::Graph &graph,
                        Renderer           &renderer,
                        const SceneConfig  &cfg)
{
	/// parse image path and/or directory
	for (const auto &node : cfg.nodes) {
		if (node.type_string != "WaveletTransformTest") continue;

		for (auto kv : node.settings.list) {
			if (kv.key == "image") {
				image_paths.add(kv.value);
			} else if (kv.key == "image_directory") {
				image_directory = kv.value;
			}
		}
	}

	load_entries();

	WaveletTransformInterface &transformInterface =
	    cast<WaveletTransformInterface>(
	        app.experiments.get("wavelet_transform").settings[0]);

	images_total = image_paths.size() -
	               image_paths.size() % transformInterface.interframe_count;

	auto &images_node      = graph.getImageNode("source_image");
	images_node.array_size = transformInterface.interframe_count;
	images_node.format     = vk::Format::eR32G32B32A32Sfloat;
	images_node.create(&graph, &renderer, false);
	first_image_idx = images_node.idx[0];
}
