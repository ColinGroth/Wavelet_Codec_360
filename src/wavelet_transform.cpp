#include "glm_include.h"

#include "debug_helper.h"
#include "img_io.h"
#include "scene_config.h"
#include "scene_config_parser.h"
#include "wavelet_transform.h"

namespace
{
WaveletTransformInterface      *settings = nullptr;
Array<WaveletTransformSettings> wavelet_transforms;
}  // namespace

extern "C" {

void MT_DLL_EXPORT init_threshold_command_buffer(Renderer           &renderer,
                                                 RenderGraph::Graph &graph,
                                                 String              pass_name,
                                                 int                 idx,
                                                 int useHeightFactor);

void MT_DLL_EXPORT
init_wavelet_transform_command_buffer(Renderer                       &renderer,
                                      RenderGraph::Graph             &graph,
                                      const WaveletTransformSettings &transform,
                                      bool                            changed,
                                      int                             idx,
                                      int                             nlevel,
                                      int *compute_threads_inv,
                                      int  useHeightFactor,
                                      int *block_size);

void MT_DLL_EXPORT init_interframe_tranform_command_buffer(
    Renderer                       &renderer,
    RenderGraph::Graph             &graph,
    const WaveletTransformSettings &transform,
    bool                            changed,
    int                             idx);

void MT_DLL_EXPORT init(App                &app,
                        GLFWwindow         *window,
                        RenderGraph::Graph &graph,
                        Renderer           &renderer,
                        const SceneConfig  &cfg);

void MT_DLL_EXPORT update(App                &app,
                          RenderGraph::Graph &graph,
                          Renderer           &renderer,
                          uint32_t            idx,
                          float               delta);

void init_threshold_command_buffer(Renderer           &renderer,
                                   RenderGraph::Graph &graph,
                                   String              pass_name,
                                   int                 idx,
                                   int                 useHeightFactor)
{
	auto &imgNode = graph.getImageNode("img_wavelet");
	auto &rpNode  = graph.getRenderPassNode(pass_name);
	auto &pass    = renderer.passes[rpNode.idx];

	renderer.waitIdle();
	for (auto &cmd : pass.commandBuffers) { cmd.reset(); }

	int compute_threads =
	    imgNode.width * imgNode.height * settings->interframe_count;

	if (!pass.commandBuffers[idx].initialized) {
		auto cmd = pass.beginSubCommandBuffer(idx);

		rpNode.preTransitions(&graph, &renderer, idx);

		auto &pNode           = graph.pipelineNodes[rpNode.pipelines[0]];
		auto &computePipeline = renderer.computePipelines[pNode.idx];
		computePipeline.bind(cmd);

		pNode.bindDescriptorSets(&graph, &renderer, idx, cmd);
		cmd.pushConstants(computePipeline.pipelineLayout,
		                  vk::ShaderStageFlagBits::eCompute,
		                  0,
		                  sizeof(int),
		                  &useHeightFactor);
		cmd.dispatch((compute_threads + 64) / 64, 1, 1);
		rpNode.postTransitions(&graph, &renderer, idx);

		auto &feedbackImgWavelet = graph.imageStore[imgNode.idx[0]];
		auto &imguiOutImage =
		    graph.imageStore[graph.getImageNode("imgui/output").idx[0]];
		feedbackImgWavelet.blit(cmd, vk::Filter::eNearest, imguiOutImage);

		CHECK_VK_ERROR(cmd.end());
	}
}

void init_wavelet_transform_command_buffer(
    Renderer                       &renderer,
    RenderGraph::Graph             &graph,
    const WaveletTransformSettings &transform,
    bool                            changed,
    int                             idx,
    int                             nlevel,
    int                            *compute_threads_inv,
    int                             useHeightFactor,
    int                            *block_sizes)
{
	auto &rpNode = graph.getRenderPassNode(transform.pass_name);
	auto &pass   = renderer.passes[rpNode.idx];

	if (changed) {
		renderer.waitIdle();
		for (auto &cmd : pass.commandBuffers) { cmd.reset(); }
	}

	int compute_threads =
	    rpNode.width * rpNode.height * settings->interframe_count;

	if (!pass.commandBuffers[idx].initialized) {
		auto cmd = pass.beginSubCommandBuffer(idx);

		int count = settings->interframe_count;
		if (transform.type == RECONSTRUCTION) { count = 0; }
		for (int i = 0; i < count; ++i) {
			auto &inputImageNode    = graph.getImageNode(transform.input_image);
			auto &tmpImgWaveletNode = graph.getImageNode(transform.tmp_image);
			auto &inputImage        = graph.imageStore[inputImageNode.idx[i]];
			auto &tmpImgWavelet = graph.imageStore[tmpImgWaveletNode.idx[i]];

			inputImage.blit(cmd, vk::Filter::eNearest, tmpImgWavelet);
		}

		int iterations = settings->forward_iterations * 2;
		if (transform.type == RECONSTRUCTION) { iterations = nlevel * 2; }
		for (int j = 0; j < iterations; ++j) {
			rpNode.preTransitions(&graph, &renderer, idx);

			auto &pNode           = graph.pipelineNodes[rpNode.pipelines[0]];
			auto &computePipeline = renderer.computePipelines[pNode.idx];
			computePipeline.bind(cmd);

			pNode.bindDescriptorSets(&graph, &renderer, idx, cmd);
			int invocation = j;
			if (transform.type == RECONSTRUCTION) {
				invocation = nlevel * 2 - j - 1;
			}
			int reconstructAll = 0;
			if (transform.type == RECONSTRUCTION &&
			    compute_threads_inv[nlevel - 1] >=
			        rpNode.width * rpNode.height) {
				reconstructAll = 1;
			}
			// block size when encoding is not used, hence nullptr
			int block_size = (block_sizes) ? block_sizes[int(j / 2)] : 0;
			int pushData[] = {invocation,
			                  settings->interframe_count,
			                  block_size,
			                  reconstructAll};
			cmd.pushConstants(computePipeline.pipelineLayout,
			                  vk::ShaderStageFlagBits::eCompute,
			                  0,
			                  sizeof(pushData),
			                  &pushData);
			if (transform.type == RECONSTRUCTION) {
				compute_threads = compute_threads_inv[int(j / 2)];
			}
			cmd.dispatch((compute_threads + 64) / 64, 1, 1);
			rpNode.postTransitions(&graph, &renderer, idx);
			if (transform.type != RECONSTRUCTION && j % 2 == 1) {
				compute_threads >>= 2;
			}
		}

		/// use this to blit the output to "imgui/output" for simpler
		/// visualisation. However, this will slow down the playback speed!
		//		if (transform.type == RECONSTRUCTION) {
		//		auto &tmpImgWaveletNode =
		//graph.getImageNode(transform.tmp_image); 		auto &outputImageNode   =
		//graph.getImageNode(transform.output_image); 			auto &tmpImgWavelet =
		//graph.imageStore[tmpImgWaveletNode.idx[0]]; 			auto &outputImage   =
		//graph.imageStore[outputImageNode.idx[0]];
		//
		//			tmpImgWavelet.blit(cmd, vk::Filter::eNearest, outputImage);
		//		}

		CHECK_VK_ERROR(cmd.end());

		if (transform.type != RECONSTRUCTION) {
			init_threshold_command_buffer(
			    renderer, graph, "approx/rp", idx, useHeightFactor);
		}
	}
}

void init_interframe_tranform_command_buffer(
    Renderer                       &renderer,
    RenderGraph::Graph             &graph,
    const WaveletTransformSettings &transform,
    bool                            changed,
    int                             idx)
{
	auto &rpNode = graph.getRenderPassNode(transform.pass_name);
	auto &pass   = renderer.passes[rpNode.idx];

	if (changed) {
		renderer.waitIdle();
		for (auto &cmd : pass.commandBuffers) { cmd.reset(); }
	}

	int compute_threads = 0;
	if (settings->useInterframe) {
		compute_threads =
		    rpNode.width * rpNode.height * settings->interframe_count;
	}

	if (!pass.commandBuffers[idx].initialized) {
		auto cmd = pass.beginSubCommandBuffer(idx);

		int iterations = log2(settings->interframe_count);

		for (int j = 0; j < iterations; ++j) {
			rpNode.preTransitions(&graph, &renderer, idx);

			auto &pNode           = graph.pipelineNodes[rpNode.pipelines[0]];
			auto &computePipeline = renderer.computePipelines[pNode.idx];
			computePipeline.bind(cmd);

			pNode.bindDescriptorSets(&graph, &renderer, idx, cmd);
			int pushData[] = {j, settings->interframe_count};
			cmd.pushConstants(computePipeline.pipelineLayout,
			                  vk::ShaderStageFlagBits::eCompute,
			                  0,
			                  sizeof(pushData),
			                  &pushData);
			cmd.dispatch((compute_threads + 64) / 64, 1, 1);
			rpNode.postTransitions(&graph, &renderer, idx);

			compute_threads >>= 1;
		}

		for (int i = 0; i < settings->interframe_count; ++i) {
			int side = 0;
			if (i > 0) { side = int(log2(i) + iterations) % 2; }
			auto &tmpImgWaveletNode = graph.getImageNode(transform.tmp_image);
			auto &tmpImgWavelet =
			    graph.imageStore[tmpImgWaveletNode.idx
			                         [i + settings->interframe_count * side]];
			auto &outputImageNode = graph.getImageNode(transform.output_image);
			auto &outputImage     = graph.imageStore[outputImageNode.idx[i]];

			tmpImgWavelet.blit(cmd, vk::Filter::eNearest, outputImage);
		}

		CHECK_VK_ERROR(cmd.end());

		init_threshold_command_buffer(
		    renderer, graph, "interframe_approx/rp", idx, 0);
	}
}

void init(App                &app,
          GLFWwindow         *window,
          RenderGraph::Graph &graph,
          Renderer           &renderer,
          const SceneConfig  &cfg)

{
	app.experiments.get("wavelet_transform")
	    .settings.add(WaveletTransformInterface());

	/// parse .prj file
	for (const auto &node : cfg.nodes) {
		if (node.type_string != "WaveletTransform") continue;

		WaveletTransformSettings transform = {};
		for (auto kv : node.settings.list) {
			kv.value_parser.str = kv.value;
			if (kv.key == "type") {
				transform.type = string_to_wtt(kv.value);
			} else if (kv.key == "input") {
				transform.input_image = kv.value;
			} else if (kv.key == "pass") {
				transform.pass_name = kv.value;
			} else if (kv.key == "output") {
				transform.output_image = kv.value;
			} else if (kv.key == "tmp") {
				transform.tmp_image = kv.value;
			} else if (kv.key == "kernel_name") {
				transform.kernel_name = kv.value;
			} else if (kv.key == "lp_kernel" && transform.type != INTERFRAME) {
				transform.kernel_size = 0;
				auto &size            = transform.kernel_size;
				while (kv.value_parser.i < kv.value.size()) {
					auto token = cfg::parse_next_token(kv.value_parser);
					cfg::assert_token_type(
					    kv.value_parser, token, cfg::Token::Type::Number);
					transform.kernel_buffer.lp[size++] =
					    cfg::parse_float(token.str);
				}
				fprintf(stdout,
				        "LowPass Kernel of size '%d'\n",
				        transform.kernel_size);
			} else if (kv.key == "hp_kernel" && transform.type != INTERFRAME) {
				transform.kernel_size = 0;
				auto &size            = transform.kernel_size;
				while (kv.value_parser.i < kv.value.size()) {
					auto token = cfg::parse_next_token(kv.value_parser);
					cfg::assert_token_type(
					    kv.value_parser, token, cfg::Token::Type::Number);
					transform.kernel_buffer.hp[size++] =
					    cfg::parse_float(token.str);
				}
				fprintf(stdout,
				        "HighPass Kernel of size '%d'\n",
				        transform.kernel_size);
			} else {
				auto err = String().format(
				    "Error: WaveletTransform has no setting: %s",
				    kv.key.begin());
				cfg::report_error_and_exit(
				    cfg::report_parser_error(kv.value_parser, err));
			}
		}
		app.experiments.get("wavelet_transform").settings.add(transform);
		wavelet_transforms.add(transform);
	}
	// kernels are typically given with sum sqrt(2)
	int min_size = std::min(wavelet_transforms[0].kernel_size,
	                        wavelet_transforms[1].kernel_size);
	for (int i = 0; i < min_size; ++i) {
		wavelet_transforms[0].kernel_buffer.lp[i] /= 1.4142135623730947;
		wavelet_transforms[1].kernel_buffer.lp[i] *= 1.4142135623730947;
	}

	for (const auto &transform : wavelet_transforms) {
		auto &imgNode = graph.getImageNode(transform.input_image);
		auto &rpNode  = graph.getRenderPassNode(transform.pass_name);
		auto &pNode   = graph.pipelineNodes[rpNode.pipelines[0]];

		rpNode.width                        = imgNode.width;
		rpNode.height                       = imgNode.height;
		rpNode.inherit_swapchain_resolution = false;

		if (transform.type != INTERFRAME) {
			pNode.shaders[0].specializationMap.add({});
			pNode.shaders[0].specializationMap.last().setSize(sizeof(uint32_t));
			pNode.shaders[0]
			    .specializationInfo.setDataSize(sizeof(uint32_t))
			    .setPData(&transform.kernel_size);

			auto &kernelBuffer = graph.getBufferNode(transform.kernel_name);
			kernelBuffer.size  = sizeof(transform.kernel_buffer.lp[0]) *
			                    transform.kernel_size * 2;
			kernelBuffer.create(&graph, &renderer);
			{
				BufferMapper mapper(kernelBuffer.buffers[0]);
				std::memcpy(mapper.data,
				            &transform.kernel_buffer.lp,
				            transform.kernel_size *
				                sizeof(transform.kernel_buffer.lp[0]));
				std::memcpy((uint8_t *)mapper.data +
				                transform.kernel_size *
				                    sizeof(transform.kernel_buffer.lp[0]),
				            &transform.kernel_buffer.hp,
				            transform.kernel_size *
				                sizeof(transform.kernel_buffer.hp[0]));
			}
		}
	}

	auto &settingsBuffer = graph.getBufferNode("settings");
	settingsBuffer.size  = sizeof(WaveletTransformInterface);

	auto &imgNode    = graph.getImageNode("img_wavelet");
	auto &bufferNode = graph.getBufferNode("compacted");
	bufferNode.size =
	    imgNode.width * imgNode.height * sizeof(CompressedWavelet);

	auto rpNode    = &graph.getRenderPassNode("compaction/rp");
	rpNode->width  = imgNode.width;
	rpNode->height = imgNode.height;
	rpNode->inherit_swapchain_resolution = false;
}

void update(App                &app,
            RenderGraph::Graph &graph,
            Renderer           &renderer,
            uint32_t            idx,
            float               delta)
{
	if (settings == nullptr) {
		settings = &cast<WaveletTransformInterface>(
		    app.experiments.get("wavelet_transform").settings[0]);
	}
	auto &src_img = graph.getImageNode("source_image");

	static int max_forward_iterations = (int)log2(int(src_img.width / 32)) - 2;
	settings->forward_iterations =
	    std::min(settings->forward_iterations, max_forward_iterations);

	bool forwardChanged  = false;
	bool backwardChanged = false;
	if (ImGui::Begin("Properties")) {
		if (ImGui::CollapsingHeader("Wavelet Transformation")) {
			forwardChanged |= ImGui::SliderInt("Input Levels",
			                                   &settings->forward_iterations,
			                                   1,
			                                   max_forward_iterations);
			bool threshold_img_changed =
			    ImGui::DragFloat("Image based Threshold",
			                     &settings->image_threshold,
			                     0.01f,
			                     0.f,
			                     1.f);
			ImGui::DragFloat("Inter-frame Threshold",
			                 &settings->interframe_threshold,
			                 0.01f,
			                 0.f,
			                 1.f);
			ImGui::DragInt(
			    "Interframe Count", &settings->interframe_count, 2, 2, 8);

			auto        &imgNode = graph.getImageNode("prefix_sum_images/cdf");
			auto        &img     = graph.imageStore[imgNode.idx[idx]];
			static float percentage = -1;
			if (percentage < 0 || threshold_img_changed) {
				downloadImage(renderer,
				              img,
				              &percentage,
				              glm::ivec2(img.width - 1, img.height - 1),
				              glm::ivec2(1));
				percentage /= img.width * (img.height - 1);
			}

			ImGui::Text("Approx: %.2f%%", percentage * 100.f);
		}
	}
	ImGui::End();

	{
		auto        &settingsBuffer = graph.getBufferNode("settings");
		BufferMapper mapper(settingsBuffer.buffers[0]);
		memcpy(mapper.data, settings, sizeof(WaveletTransformInterface));
	}
}
}
