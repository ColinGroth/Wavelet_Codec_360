#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <mutex>
#include <thread>

#include "glm_include.h"
#include "img_io.h"
#include "imgui.h"
#include "rendergraph.h"
#include "renderpass.h"

#include "scene_config.h"
#include "scene_config_parser.h"
#include "wavelet_transform.h"

#include "load_source_images.h"
#include "wavelet_video_compression.h"

namespace
{
GuiSettings settings = GuiSettings();

WaveletTransformInterface *transformInterface = nullptr;

init_wavelet_transform_command_buffer_t init_wavelet_transform_command_buffer =
    nullptr;
init_interframe_tranform_command_buffer_t
    init_interframe_tranform_command_buffer = nullptr;
}  // namespace

unsigned int images_total;
unsigned int current_image_idx = 0;

bool test_multiple_mode  = false;
bool start_test_multiple = false;
int  current_test        = 0;
int  current_frame       = -1;

struct PredictionMisses
{
	size_t predictions    = 0;
	size_t total_wavelets = 0;
	size_t pred_last      = 0;
	size_t total_last     = 0;
};
PredictionMisses predictionMisses;

bool         currently_saving     = false;
unsigned int saved_frames_start   = 0;
unsigned int saved_frames_current = 0;

extern "C" {
#ifdef BENCH
#define BENCH_START(name) bench_start(name)
#define BENCH_END bench_end()
#else
#define BENCH_START(b)
#define BENCH_END
#endif

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

void fillCmdBuffer(RenderGraph::Graph &graph,
                   Renderer           &renderer,
                   uint32_t           &idx,
                   String              nodeName,
                   int                 compute_threads,
                   void               *pushData,
                   size_t              pushDataSize)
{
	auto &rpNode = graph.getRenderPassNode(nodeName);
	auto &pass   = renderer.passes[rpNode.idx];
	auto  cmd    = pass.beginSubCommandBuffer(idx);

	rpNode.preTransitions(&graph, &renderer, idx);
	auto &pNode           = graph.pipelineNodes[rpNode.pipelines[0]];
	auto &computePipeline = renderer.computePipelines[pNode.idx];
	computePipeline.bind(cmd);
	pNode.bindDescriptorSets(&graph, &renderer, idx, cmd);
	if (pushDataSize > 0) {
		cmd.pushConstants(computePipeline.pipelineLayout,
		                  vk::ShaderStageFlagBits::eCompute,
		                  0,
		                  pushDataSize,
		                  pushData);
	}
	cmd.dispatch((compute_threads + 64) / 64, 1, 1);
	rpNode.postTransitions(&graph, &renderer, idx);
	CHECK_VK_ERROR(cmd.end());
}

/**
 ** 	Here we encode the entire video. Please not, that for the development of
 **	`	the codec our priority was not fast encoding but rather fast playback of
 **		360 video content. Therefore, there is a lot of optimization potential
 **		here.
 */
void encodeFrames(App                &app,
                  RenderGraph::Graph &graph,
                  Renderer           &renderer,
                  uint32_t           &idx)
{
	//	BENCH_START("Compression total");

	int &ifc          = transformInterface->interframe_count;
	int  levels_total = transformInterface->forward_iterations;

	frames.resize(ifc);
	auto &waveletImgNode = graph.getImageNode("img_wavelet");

	if (settings.encodingTotal < ifc) { settings.encodingTotal = images_total; }
	if (settings.encodingTotal % ifc != 0) {
		settings.encodingTotal -= settings.encodingTotal % ifc;
	}

	size_t end_of_file = 0;
	/// first write the header
	{
		header = {header.VERSION,
		          glm::ivec2(waveletImgNode.width, waveletImgNode.height),
		          settings.encodingTotal,
		          levels_total,
		          ifc,
		          transformInterface->image_threshold,
		          transformInterface->interframe_threshold};
		std::ofstream waveletFile{settings.savePath.begin(), std::ios::binary};
		assert(waveletFile.is_open());
		waveletFile.write((const char *)&header, sizeof(header));

		metaData_size = sizeof(header) +
		                (sizeof(FrameMetaData) - sizeof(Array<LevelMetaData>)) *
		                    settings.encodingTotal +
		                (sizeof(LevelMetaData) - sizeof(Array<uint32_t>)) *
		                    settings.encodingTotal * (levels_total + 1);
		end_of_file = metaData_size;
	}

	current_image_idx = 0;
	/// only load one interframe set of frames at a time, perform the encoding
	/// (forward wavelet transforms) and write data to file. Otherwise the GPU
	/// buffer overflows. load_next_textures returns false if all textures are
	/// encoded already.
	while (current_image_idx < settings.encodingTotal &&
	       load_next_textures(app, graph, renderer)) {
		BENCH_START("Compression time one set");
		/*
		 *  	Wavelet transformation
		 */
		WaveletTransformSettings &transformSettings =
		    cast<WaveletTransformSettings>(
		        app.experiments.get("wavelet_transform").settings[1]);
		init_wavelet_transform_command_buffer(renderer,
		                                      graph,
		                                      transformSettings,
		                                      true,
		                                      idx,
		                                      0,
		                                      nullptr,
		                                      int(settings.useHeightFactor),
		                                      nullptr);

		WaveletTransformSettings &interframeSettings =
		    cast<WaveletTransformSettings>(
		        app.experiments.get("wavelet_transform").settings[3]);
		init_interframe_tranform_command_buffer(
		    renderer, graph, interframeSettings, true, idx);

		/**
		 *  	Forward wavelet transforms per frame, interframe wavelet
		 *  	transform ones for all transformed frames of the set and
		 *		compaction of the data
		 */
		size_t frame_end = end_of_file;
		for (int i = 0; i < ifc; ++i) {
			auto &frame = frames[i];
			frame.metaData.levelData.resize(levels_total + 1);
			frame.dataPerLevel.resize(levels_total + 1);

			auto        &bufferNode    = graph.getBufferNode("compacted");
			auto        &clusterBuffer = graph.getBufferNode("clustering");
			BufferMapper clusterBufferMapper(clusterBuffer.buffers[0]);
			auto        &normBuffer = graph.getBufferNode("normalize");
			BufferMapper normBufferMapper(normBuffer.buffers[0]);
			auto        &cdf = graph.getImageNode("prefix_sum_images/cdf");

			for (int j = 0; j < 12 * ifc; ++j) {
				*((float *)normBufferMapper.data + j) = -FLT_MAX;
				if (j % 6 < 3) { *((float *)normBufferMapper.data + j) = -0; }
			}

			int    blocks_total = 0;
			size_t wvlt_count   = 0;

			for (int j = levels_total; j >= 0; --j) {
				int            lvlIdx    = levels_total - j;
				LevelMetaData &levelData = frame.metaData.levelData[lvlIdx];

				if (waveletImgNode.width >> (j + 1) <= levelData.block_size) {
					levelData.block_size = std::max(
					    int((waveletImgNode.width >> (j + 2)) + 0.5), 2);
				}
				int blocks_horiz = 0;
				int blocks_vert  = 0;
				if (j != levels_total) {
					blocks_horiz = (waveletImgNode.width >> (j + 1)) /
					               levelData.block_size;
					blocks_vert = (waveletImgNode.height >> (j + 1)) /
					              levelData.block_size;
				}
				printf(
				    "Frame: %d Level: %d block_horiz: %d block_vert: %d "
				    "block_size: %d\n",
				    i,
				    lvlIdx,
				    blocks_horiz,
				    blocks_vert,
				    levelData.block_size);
				int blocks             = blocks_horiz * blocks_vert * 3;
				levelData.blocks_horiz = blocks_horiz;
				levelData.blocks_vert  = blocks_vert;
				levelData.block_start_marker =
				    frame_end + blocks_total * sizeof(levelData.block_ends[0]);
				levelData.block_ends.resize(blocks);
				blocks_total += blocks;
				levelData.block_end_marker =
				    frame_end + blocks_total * sizeof(levelData.block_ends[0]);

				*(int *)clusterBufferMapper.data = levelData.block_size;
				for (int b = 1; b <= blocks + 1; ++b) {
					*((int *)clusterBufferMapper.data + b) = 0;
				}

				int   pushData[] = {j, levels_total, i};
				auto &maskNode   = graph.getImageNode("prefix_sum_images/pdf");
				fillCmdBuffer(graph,
				              renderer,
				              idx,
				              "masking/rp",
				              maskNode.width * maskNode.height,
				              &pushData,
				              sizeof(pushData));
				fillCmdBuffer(graph,
				              renderer,
				              idx,
				              "compaction/rp",
				              (maskNode.width * maskNode.height) >> (2 * j),
				              &pushData,
				              sizeof(pushData));

				/// here the shaders are executed
				{
					OneShotCommandBuffer cmd{renderer.ctx.dev,
					                         renderer.ctx.commandPool,
					                         renderer.ctx.graphicsQueue};
					renderer.executeQueue(cmd.cmd, graph.queues[queueIdxComp]);
				}

				/// get compacted wavelet data
				BufferMapper compBufferMapper(bufferNode.buffers[0]);
				auto *data = ((CompressedWavelet *)compBufferMapper.data);

				/// read wavelet count from cdf (last pixel at bottom right)
				float value_;
				downloadImage(renderer,
				              graph.imageStore[cdf.idx[0]],
				              &value_,
				              glm::ivec2(cdf.width - 1, cdf.height - 1),
				              glm::ivec2(1));
				int level_count = value_;
				//								assert(level_count > 0);

				/// copy data to our frame information structure
				frame.dataPerLevel[lvlIdx].resize(level_count);
				std::memcpy(frame.dataPerLevel[lvlIdx].begin(),
				            data,
				            level_count * sizeof(CompressedWavelet));

				std::memcpy(levelData.block_ends.begin(),
				            (int *)clusterBufferMapper.data + 2,
				            blocks * sizeof(int));

				uint32_t block_sum = 0;
				for (auto &block : levelData.block_ends) {
					if (block == 0) {
						block = block_sum;
					} else {
						block_sum = block;
					}
				}

				wvlt_count += level_count;
				frame.metaData.levelData[lvlIdx].level_end =
				    (wvlt_count - 1) * sizeof(CompressedWavelet) + frame_end;
			}

			/**
			 * set all the meta data information
			 */
			frame.metaData.total_blocks_size =
			    blocks_total *
			    sizeof(frame.metaData.levelData[0].block_ends[0]);

			for (LevelMetaData &levelData : frame.metaData.levelData) {
				levelData.level_end += frame.metaData.total_blocks_size;
			}

			frame.metaData.frame_start_marker = frame_end;
			if (wvlt_count <= 0) {
				assert(false);
				exit(EXIT_FAILURE);
			}
			frame_end += frame.metaData.total_blocks_size +
			             wvlt_count * sizeof(CompressedWavelet);
			frame.metaData.frame_end_marker = frame_end - 1;

			frame.metaData.approxNorm =
			    *((NormalizationData *)normBufferMapper.data + i * 2);
			frame.metaData.waveletsNorm =
			    *((NormalizationData *)normBufferMapper.data + i * 2 + 1);

			// sign bits were flipped in shader for mins
			frame.metaData.waveletsNorm.min_r *= -1;
			frame.metaData.waveletsNorm.min_g *= -1;
			frame.metaData.waveletsNorm.min_b *= -1;
			frame.metaData.approxNorm.min_r *= -1;
			frame.metaData.approxNorm.min_g *= -1;
			frame.metaData.approxNorm.min_b *= -1;
		}
		BENCH_END;

		/**
		 **		Write data to file
		 **/
		{
			std::fstream waveletFile{
			    settings.savePath.begin(),
			    std::ios::out | std::ios::in | std::ios::binary};

			/// write meta data
			size_t meta_data_pointer =
			    sizeof(header) +
			    (sizeof(FrameMetaData) - sizeof(Array<LevelMetaData>)) *
			        (current_image_idx - ifc) +
			    (sizeof(LevelMetaData) - sizeof(Array<uint32_t>)) *
			        (current_image_idx - ifc) * (levels_total + 1);
			waveletFile.seekp(meta_data_pointer);

			for (Frame &frame : frames) {
				waveletFile.write(
				    (const char *)&frame.metaData,
				    sizeof(FrameMetaData) - sizeof(Array<LevelMetaData>));
				for (LevelMetaData &levelMetaData : frame.metaData.levelData) {
					waveletFile.write((const char *)&levelMetaData,
					                  sizeof(levelMetaData) -
					                      sizeof(levelMetaData.block_ends));
				}
			}

			/// write block_ends and wavelet data
			waveletFile.seekp(end_of_file);
			for (Frame &frame : frames) {
				for (LevelMetaData &levelMetaData : frame.metaData.levelData) {
					waveletFile.write(
					    (const char *)levelMetaData.block_ends.begin(),
					    levelMetaData.block_ends.size() *
					        sizeof(levelMetaData.block_ends[0]));
				}
				for (auto &levelData : frame.dataPerLevel) {
					waveletFile.write((const char *)levelData.begin(),
					                  levelData.size() * sizeof(levelData[0]));
				}
			}

			end_of_file = waveletFile.tellp();
		}
	}
	current_image_idx = 0;
	//	BENCH_END;
}

/**
 **		All the meta data is preloaded before the playback is started. This
 **		includes the file header as well as the respective meta data for each
 **		frame.
 ***/
void preloadMetaData(RenderGraph::Graph &graph)
{
	std::ifstream waveletFile(settings.savePath.begin(), std::ios::binary);

	/// load file header first
	waveletFile.read((char *)&header, sizeof(WaveletFileHeader));
	WaveletFileHeader currentHeader;
	if (header.VERSION != currentHeader.VERSION) {
		fprintf(stderr,
		        "WRONG file header version! Re-encode or manually adjust file "
		        "header.\n");
		assert(false);
		exit(EXIT_FAILURE);
	}
	assert(transformInterface->interframe_count == header.interframe_count);

	/// next load the meta data of all frames into the memory
	frames.resize(header.frame_count);
	for (Frame &frame : frames) {
		frame.metaData.levelData.resize(header.level + 1);
		waveletFile.read((char *)&frame.metaData,
		                 sizeof(FrameMetaData) - sizeof(Array<LevelMetaData>));

		for (int i = header.level; i >= 0; --i) {
			LevelMetaData &levelMetaData =
			    frame.metaData.levelData[header.level - i];
			waveletFile.read(
			    (char *)&levelMetaData,
			    sizeof(levelMetaData) - sizeof(levelMetaData.block_ends));
		}
	}
	metaData_size = waveletFile.tellg();

	/// here we declare memory of the size of 3 interframe sets (assuming max
	/// wavelet count) to be used for the real-time wavelet loading later
	int op_frames = header.interframe_count * 3;
	if (frames.size() < op_frames) { op_frames = header.interframe_count; }
	for (int i = 0; i < op_frames; ++i) {
		frames[i].preloadInfo.resize(header.level + 1);
		frames[i].dataPerLevel.resize(header.level + 1);
		frames[i].maskPerLevel.resize(header.level + 1);
		for (int j = 0; j <= header.level; ++j) {
			frames[i].dataPerLevel[header.level - j].resize(
			    (header.size.x * header.size.y) >> (2 * j));
			auto &lvlData = frames[i].metaData.levelData[header.level - j];
			lvlData.block_ends.resize(lvlData.blocks_horiz *
			                          lvlData.blocks_vert * 3);
			frames[i].maskPerLevel[header.level - j].resize(
			    (header.size.x / lvlData.block_size *
			     (header.size.y / lvlData.block_size)) >>
			    (2 * j));
		}
	}
}

/**
 *  	Loads one line of blocks of wavelet coefficients
 * 		Returns:
 * 		true  - the loaded line is the last line needed in the level
 *				(concidering the eye data).
 * 		false - otherwise; more lines need to be loaded
 **/
bool loadLine(int lvl, int frameIdx, bool preloading = false)
{
	const std::lock_guard<std::mutex> lock(driveBusyMutex);

	int lvlIdx = header.level - lvl;

	Frame &frame =
	    frames[frameIdx % (transformInterface->interframe_count * 3)];
	LevelMetaData    &lvlMetaData = frames[frameIdx].metaData.levelData[lvlIdx];
	auto             &blockEnds   = frame.metaData.levelData[lvlIdx].block_ends;
	LevelPreloadInfo &pre         = frame.preloadInfo[lvlIdx];

	std::ifstream waveletFile(settings.savePath.begin(), std::ios::binary);

	/// first load block_ends for level of interframe if not done yet
	if (!pre.areBlockEndsLoaded && lvlIdx) {
		waveletFile.seekg(lvlMetaData.block_start_marker);
		waveletFile.read((char *)blockEnds.begin(),
		                 blockEnds.size() * sizeof(blockEnds[0]));
		pre.areBlockEndsLoaded = true;
		return false;
	}

	if (lvl == header.level) {
		/**
		 * 	for the highest level (approximation layer) we want to preload the
		 * 	full viewport
		 */
		if (pre.wavelet_count != 0) { return true; }

		size_t wvltStartMarker = frames[frameIdx].metaData.frame_start_marker +
		                         frames[frameIdx].metaData.total_blocks_size;
		size_t approxSize =
		    lvlMetaData.level_end - wvltStartMarker + sizeof(CompressedWavelet);
		size_t wvltCount = approxSize / sizeof(CompressedWavelet);

		waveletFile.seekg(wvltStartMarker);
		waveletFile.read((char *)frame.dataPerLevel[0].begin(), approxSize);
		pre.wavelet_count = wvltCount;

		return true;
	} else {
		/**
		**	load line in arbitrary wavelet level by given section
		**/
		int const &blocks_horiz = eyeData.level_block_amount[lvlIdx].x;
		int const &blocks_vert  = eyeData.level_block_amount[lvlIdx].y;

		/// The section of blocks that we need to load is defined by the 'Gaze'
		/// module and available over the sec variable. sec_vp defines the
		/// expansion of the whole viewport per level, while sec is the relative
		/// expansion per level. When we do not use foveation sec_vp = sec!
		auto &sec    = eyeData.sec[lvlIdx];
		auto &sec_vp = eyeData.sec_viewport[lvlIdx];

		if (sec_vp.end.x == pre.last_block.x &&
		    sec_vp.end.y <= pre.last_block.y) {
			return true;
		}
		if (pre.first_block.y == -1 || sec_vp.end.x != pre.last_block.x ||
		    sec_vp.start.y < pre.first_block.y) {
			pre.former_first = pre.first_block;
			pre.former_last  = pre.last_block;
			pre.first_block  = sec_vp.start;
			pre.last_block   = {sec_vp.end.x, pre.first_block.y - 1};
		}

		/// Here we come to the exact line of blocks that we want to load. b1
		/// and b2 define the index of the start and end block of our line.
		int      b1_idx;
		int      b2_idx;
		uint32_t line_start;
		uint32_t line_size;

		/// k defines the index of the line we load in this function call
		int k = ++pre.last_block.y;
		/// Per level we have 3 wavelet coefficient sections (LH, HL, HH). From
		/// every section we need to load the exact same line for the
		/// reconstruction. j defines the section.
		for (int j = 0; j < 3; ++j) {
			b1_idx = k * blocks_horiz + pre.first_block.x;
			b2_idx = k * blocks_horiz + pre.last_block.x;
			if (j > 0) {
				b1_idx += blocks_horiz * 2 * blocks_vert + k * blocks_horiz +
				          blocks_horiz * (j - 1);
				b2_idx += blocks_horiz * 2 * blocks_vert + k * blocks_horiz +
				          blocks_horiz * (j - 1);
			}
			/// Double checking if any part of the section we want to load was
			/// not already loaded by former processes. If so, we only load the
			/// missing part.
			if (pre.former_first.x != -1 && k >= pre.former_first.y &&
			    k <= pre.former_last.y) {
				if (pre.first_block.x < pre.former_first.x) {
					b2_idx -= pre.last_block.x - pre.former_first.x + 1;
				} else if (pre.first_block.x > pre.former_first.x) {
					b1_idx += pre.former_last.x - pre.first_block.x + 1;
				}
			}

			if (b1_idx > b2_idx) {
				fprintf(
				    stderr,
				    "ERROR read line: start index is taller than end index\n");
				exit(0);
			}
			int const b_size = b2_idx - b1_idx + 1;

			/// We need to load the same area for both eyes of a stereo frame.
			/// Here we assume horizontally split stereo frames.
			for (int e = 0; e < 2; ++e) {
				if (k >= sec.start.y && k <= sec.end.y) {
					line_start = 0;
					line_size  = 0;

					int b1_fov = b1_idx + sec.start.x - sec_vp.start.x;
					int b2_fov = b2_idx + sec.end.x - sec_vp.end.x;

					if (b1_fov != 0) {
						line_start = blockEnds[b1_fov - 1];
					} else {
						line_start = blockEnds[b1_fov];
					}
					if (b2_fov >=
					    blocks_horiz * blocks_vert * 6 - 1) {  /// last block
						line_size = (lvlMetaData.level_end -
						             frames[frameIdx]
						                 .metaData.levelData[lvlIdx - 1]
						                 .level_end) /
						                sizeof(CompressedWavelet) -
						            line_start;
					} else {
						line_size = blockEnds[b2_fov] - line_start;
					}

					if (line_size > header.size.x * 32) {
						fprintf(
						    stderr,
						    "ERROR read line: line size (probably) too big\n");
						assert(false);
						exit(EXIT_FAILURE);
					}

					/// go to the actual position in our file
					waveletFile.seekg(frames[frameIdx]
					                      .metaData.levelData[lvlIdx - 1]
					                      .level_end +
					                  (1 + line_start) *
					                      sizeof(CompressedWavelet));
					/// load only what we need from the seeked position (blocks
					/// of line are in order)
					waveletFile.read(
					    (char *)frame.dataPerLevel[lvlIdx].begin() +
					        pre.wavelet_count * sizeof(CompressedWavelet),
					    line_size * sizeof(CompressedWavelet));

					predictionMisses.total_wavelets += line_size;
					if (preloading) {
						predictionMisses.predictions += line_size;
					}

					pre.wavelet_count += line_size;
				}

				/**
				 * create the render mask that we need to reconstruct only
				 * the blocks loaded here. The render mask is passed to the GPU
				 * and used by the inverse wavelet transform.
				 **/
				if (frameIdx % transformInterface->interframe_count == 0 &&
				    j == 0) {
					Array<int> _idx;
					_idx.resize(b_size * 4);
					int x   = (b1_idx + 1) % blocks_horiz;
					_idx[0] = (b1_idx - x + 1) * 4 + x * 2 - 2;
					_idx[1] = _idx[0] + blocks_horiz * 2;
					for (int b = 1; b < 2 * b_size; b++) {
						_idx[b * 2]     = _idx[0] + b;
						_idx[b * 2 + 1] = _idx[1] + b;
					}

					std::memcpy((int *)frame.maskPerLevel[lvlIdx].begin() +
					                pre.mask_count,
					            _idx.begin(),
					            _idx.size() * sizeof(int));
					pre.mask_count += b_size * 4;
				}

				/// prepare block indices for next wavelet section
				b1_idx += blocks_horiz * blocks_vert;
				b2_idx += blocks_horiz * blocks_vert;
				if (j > 0) {
					b1_idx += blocks_horiz * blocks_vert;
					b2_idx += blocks_horiz * blocks_vert;
				}
			}
		}
		return false;
	}
}

/**
 **		Clear-up function that runs in parallel with the reconstruction threads.
 **		Frees the preload info for all levels of the frames not needed any
 **		longer.
 ***/
void clearLastFrames()
{
	int currentFrameIdx = frameIdx;
	for (int i = 1; i <= transformInterface->interframe_count; ++i) {
		int fIdx =
		    (currentFrameIdx - i) % (transformInterface->interframe_count * 3);
		if (currentFrameIdx == 0) {
			fIdx = (header.frame_count - i) %
			       (transformInterface->interframe_count * 3);
		}

		for (int j = 0; j <= header.level; ++j) {
			frames[fIdx].preloadInfo[j] = LevelPreloadInfo();
		}
	}
}

/**
 **		Function that runs in parallel with the reconstruction of the current
 **		frame and loads/updates the viewport data of all frames c of the next
 **		inter-frame set
 ***/
void prepareNextInterframeBuffer()
{
	int &ifc         = transformInterface->interframe_count;
	int  next_if_set = frameIdx + ifc - frameIdx % ifc;
	if (next_if_set >= header.frame_count) { next_if_set = 0; }
	Gaze oldEyeData = eyeData;

	for (int i = header.level; i >= 0; --i) {
		int c = 0;
		while (c < ifc) {
			/// When the current frame reached the inter-frame set that should
			/// be preloaded, we do not want to preload anymore as uploadData()
			/// will take care of that now.
			if (frameIdx >= next_if_set && frameIdx <= next_if_set + ifc) {
				return;
			}

			/// In case that the focus point off the eye changed, all preloaded
			/// data has to be updated again (therefore c = 0). The last layer
			/// (approximation layer) is loaded entirely and needs no update in
			/// that case.
			if (eyeData != oldEyeData) {
				if (i < header.level) {
					i = header.level - 1;
					c = 0;
				}
				oldEyeData = eyeData;
			}

			/// loadLine returns true when the full viewport of the respective
			/// frame is loaded into memory
			if (loadLine(i, next_if_set + c, true)) { ++c; }
		}
	}
}

/**
 **		Loading of the missing wavelets at render-time and upload of all data of
 **		one inter-frame to the GPU
 ***/
void uploadData(RenderGraph::Graph &graph,
                int                 prev_if_idx,
                int                 if_lvlIdx,
                int                 if_lvlCount,
                int                *wvltInfo,
                int                *maskData,
                int                 buffer_offset)
{
	if (if_lvlIdx > if_lvlCount) { return; }

	int if_fullLvlIdx = (frameIdx % transformInterface->interframe_count) >>
	                    (if_lvlCount - if_lvlIdx);
	int interframe_idx = 0;
	if (if_lvlIdx > 0) {
		interframe_idx = (1 << (if_lvlIdx - 1)) + int(if_fullLvlIdx / 2);
	}
	if (!transformInterface->useInterframe) {  // Debug
		interframe_idx = frameIdx;
	}

	for (int lvlIdx = 0; lvlIdx <= header.level; ++lvlIdx) {
		Frame &frame = frames[(prev_if_idx + interframe_idx) %
		                      (transformInterface->interframe_count * 3)];
		int    i     = header.level - lvlIdx;

		/// In case that some data has not been preloaded (function:
		/// prepareNextInterframeBuffer) we want to load it directly before
		/// rendering. Like the name suggests, the loadLine function loads one
		/// line of blocks (not pixels!) from the drive every time it is called.
		/// When all required lines are loaded, TRUE is returned and the loop is
		/// exited.
		while (true) {
			if (loadLine(i, prev_if_idx + interframe_idx)) { break; }
		}

		///
		/// upload preloaded wavelets
		///
		int wavelet_count = frame.preloadInfo[lvlIdx].wavelet_count;

		/// the compacted buffer is used to store the wavelets in GPU shared
		/// memory
		auto        &compBuffer = graph.getBufferNode("compacted");
		BufferMapper compBufferMapper(compBuffer.buffers[0]);

		std::memcpy((CompressedWavelet *)compBufferMapper.data + wvltInfo[0] +
		                buffer_offset,
		            frame.dataPerLevel[lvlIdx].begin(),
		            wavelet_count * sizeof(CompressedWavelet));
		wvltInfo[0] += wavelet_count;
		wvltInfo[lvlIdx + 1] = wvltInfo[0];
		compute_threads += wavelet_count;

		///
		/// upload the mask that tells the shader what areas to decompose
		///
		if ((if_lvlIdx == if_lvlCount - 1 ||
		     transformInterface->interframe_count == 1) &&
		    !settings.reconstructAll) {
			int _idx = prev_if_idx % (transformInterface->interframe_count * 3);
			std::memcpy(
			    maskData + 16 + maskData[i + 1],
			    frames[_idx].maskPerLevel[lvlIdx].begin(),
			    frames[_idx].preloadInfo[lvlIdx].mask_count * sizeof(int));
			if (lvlIdx) {
				maskData[i] = maskData[i + 1] +
				              frames[_idx].preloadInfo[lvlIdx].mask_count;
			} else {
				maskData[i] = frames[_idx].preloadInfo[lvlIdx].mask_count;
			}
		}
	}
}

/**
 **		GPU-based rebuild of the wavelet image and inverse wavelet transform
 **   	for last relevant inter-frame (c=log(n))
 ***/
void reconstructFrame(App                &app,
                      RenderGraph::Graph &graph,
                      Renderer           &renderer,
                      uint32_t           &idx,
                      int                 c,
                      int                 if_fullLvlIdx,
                      int                 if_lvlCount,
                      int                *maskData,
                      VkFence             fence = VK_NULL_HANDLE)
{
	int pushData[] = {header.level,
	                  c,
	                  if_fullLvlIdx,
	                  transformInterface->useInterframe,
	                  settings.showWvltImg};
	/// fills the command buffer for the shader that rebuilds the wavelet image
	/// with the data from file (rebuild.comp)
	fillCmdBuffer(graph,
	              renderer,
	              idx,
	              "reconstruction/rp",
	              compute_threads,
	              &pushData,
	              sizeof(pushData));

	/// It is only necessary to fill the buffers of the inverse wavelet
	/// transform if the information of all inter-frames is uploaded
	if (!settings.showWvltImg && c == if_lvlCount) {
		WaveletTransformSettings &transformSettings =
		    cast<WaveletTransformSettings>(
		        app.experiments.get("wavelet_transform").settings[2]);

		/// determine number of threads based on the given parameters
		Array<int> threads;
		threads.resize(header.level);
		Array<int> block_sizes;
		for (int j = 0; j < header.level; ++j) {
			if (settings.reconstructAll) {
				auto &imgNode = graph.getImageNode("img_wavelet");
				threads[j]    = (imgNode.width * imgNode.height) >>
				             ((header.level - j - 1) * 2);
			} else {
				threads[j] = (maskData[header.level - j - 1] -
				              maskData[header.level - j]) *
				             frames[0].metaData.levelData[j].block_size *
				             frames[0].metaData.levelData[j].block_size;
			}
			block_sizes[j] = frames[0].metaData.levelData[j].block_size;
		}

		/// initilize inverse wavelet transform and fill buffer
		init_wavelet_transform_command_buffer(renderer,
		                                      graph,
		                                      transformSettings,
		                                      true,
		                                      idx,
		                                      header.level,
		                                      threads.begin(),
		                                      int(settings.useHeightFactor),
		                                      block_sizes.begin());
		/// fill buffer to re-map image from equirectangular to normal
		/// representation
		fillCmdBuffer(
		    graph, renderer, idx, "mapping/rp", mapping_threads, nullptr, 0);
	}

	/// command buffer is executed at end of scope
	{
		cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

		/// check if the information of all inter-frames is uploaded (because
		/// then we need to run the inverse wavelet transform)
		if (!settings.showWvltImg && c == if_lvlCount) {
			renderer.executeQueue(cmd.cmd,
			                      graph.queues[queueIdxFullReconstruct]);
		} else {
			/// Debug - show which wavelets are loaded
			renderer.executeQueue(cmd.cmd, graph.queues[queueIdxRebuilt]);
		}
		cmd.end();
		cmd.submit((vk::Fence)fence);
	}
}

/**
 **		decoding (and presentation) of one frame defined by frameIdx
 ***/
void decodeFrame(App                &app,
                 RenderGraph::Graph &graph,
                 Renderer           &renderer,
                 uint32_t           &idx)
{
	int prev_if_idx =
	    frameIdx - frameIdx % transformInterface->interframe_count;

	auto        &normBuffer = graph.getBufferNode("normalize");
	BufferMapper normBufferMapper(normBuffer.buffers[0]);
	auto        &wvltInfoBuffer = graph.getBufferNode("wavelets_info");
	BufferMapper wvltInfoBufferMapper(wvltInfoBuffer.buffers[0]);
	auto        &imgNode       = graph.getImageNode("img_wavelet");
	auto        &clusterBuffer = graph.getBufferNode("clustering");
	BufferMapper maskBufferMapper(clusterBuffer.buffers[0]);
	int         *maskData          = (int *)maskBufferMapper.data;
	VkFence      reconstruct_fence = VK_NULL_HANDLE;

	{
		///
		/// execute image clear shader (runs decoupled!)
		///
		fillCmdBuffer(graph,
		              renderer,
		              idx,
		              "clear_image/rp",
		              imgNode.width * imgNode.height,
		              nullptr,
		              0);
		cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		renderer.executeQueue(
		    cmd.cmd, graph.queues[graph.createRenderQueue("clear_image/rp")]);
		cmd.end();
		cmd.submit();
	}

	/// fill buffer with normalization data to undo in shader later
	for (int i = 0; i < header.interframe_count; ++i) {
		Frame &frame = frames[prev_if_idx + i];

		std::memcpy((NormalizationData *)normBufferMapper.data + i * 2,
		            &frame.metaData.approxNorm,
		            sizeof(NormalizationData));
		std::memcpy((NormalizationData *)normBufferMapper.data + i * 2 + 1,
		            &frame.metaData.waveletsNorm,
		            sizeof(NormalizationData));
	}

	Array<int> wvltInfo;
	wvltInfo.resize(header.level + 1);

	int if_buffer_offset = 0;

	int if_lvlCount = glm::findMSB(header.interframe_count);
	if (!transformInterface->useInterframe) {  // debug
		if_lvlCount = 1;
	}

	///
	/// This is were the magic happens:
	///	one part of an image is reconstructed by decoding the same part in all
	///	relevant inter-frames. The data for the next inter-frame (c+1) is
	/// uploaded simultaniously with the inter-frame decoding of c
	///
	for (int c = 0; c <= if_lvlCount; ++c) {
		for (int j = 0; j <= header.level; ++j) { wvltInfo[j] = 0; }
		int if_fullLvlIdx = (frameIdx % transformInterface->interframe_count) >>
		                    (if_lvlCount - c);

		/// upload of the first inter-frame image (c_0) to the GPU prior to the
		/// first rekonstruction step and set wavelet info (description below)
		if (c == 0) {
			compute_threads = 0;
			uploadData(graph,
			           prev_if_idx,
			           c,
			           if_lvlCount,
			           wvltInfo.begin(),
			           maskData,
			           if_buffer_offset);
			*((int *)wvltInfoBufferMapper.data) = if_buffer_offset;
			memcpy((int *)wvltInfoBufferMapper.data + 1,
			       wvltInfo.begin(),
			       (header.level + 1) * sizeof(int));
			if_buffer_offset += compute_threads;
			for (int j = 0; j <= header.level; ++j) { wvltInfo[j] = 0; }
		}

		/// reconstruction of inter-frame c (runs decoupled!)
		reconstructFrame(app,
		                 graph,
		                 renderer,
		                 idx,
		                 c,
		                 if_fullLvlIdx,
		                 if_lvlCount,
		                 maskData,
		                 reconstruct_fence);

		compute_threads = 0;
		/// upload of data for the next inter-frame (c+1) in parallel with the
		/// current inter-frame c
		uploadData(graph,
		           prev_if_idx,
		           c + 1,
		           if_lvlCount,
		           wvltInfo.begin(),
		           maskData,
		           if_buffer_offset);

		/// wait for the GPU to finish shading when necessary
		(void)renderer.ctx.dev.waitIdle();

		/// update wavelet info used by the
		/// reconstruction shader (rebuild.comp)
		*((int *)wvltInfoBufferMapper.data) = if_buffer_offset;
		memcpy((int *)wvltInfoBufferMapper.data + 1,
		       &wvltInfo,
		       (header.level + 1) * sizeof(int));
		if_buffer_offset += compute_threads;
	}
}

inline void ui_encoding(App                &app,
                        RenderGraph::Graph &graph,
                        Renderer           &renderer,
                        uint32_t            idx)
{
	ImGui::Checkbox("Use Height Factor", &settings.useHeightFactor);
	ImGui::InputInt("Encode Number Frames", &settings.encodingTotal);
	ImGui::InputText(
	    "Save Path", settings.savePath.begin(), settings.savePath.size() + 100);

	if (ImGui::Button("Write Video File")) {
		encodeFrames(app, graph, renderer, idx);
	}
}

inline void ui_decoding(App                &app,
                        RenderGraph::Graph &graph,
                        Renderer           &renderer,
                        uint32_t            idx)
{
	ImGui::Checkbox("Reconstruct entire image", &settings.reconstructAll);
	ImGui::Checkbox("Use Foveation", &settings.foveate);
	ImGui::Checkbox("Show Wavelets", &settings.showWvltImg);
	ImGui::Checkbox("Loop", &settings.loop);

	ImGui::InputInt("fps", &settings.fps);
	ImGui::DragFloat("head x", &eyeData.head_pos.x, 0.1, 0.0, 1.);
	ImGui::DragFloat("head y", &eyeData.head_pos.y, 0.1, 0.0, 1.);
	// eyeData.focus_left_eye = glm::vec2(settings.gazeSpeed);

	ImGui::Text(settings.savePath.begin());
	if (ImGui::Button(settings.playVideo ? "Stop" : "Start")) {
		settings.playVideo = !settings.playVideo;
		if (settings.playVideo) {
			total_frame_count = 0;
			start_time        = std::chrono::system_clock::now();
		}
	}
	/// file header info
	if (isMetaDataLoaded) {
		ImGui::Text("Size: %ix%i\n", header.size.x, header.size.y);
		ImGui::Text("Frames: %i\n", header.frame_count);
		ImGui::Text("Level: %i\n", header.level);
		ImGui::Text("Inter-frames: %i\n", header.interframe_count);
		ImGui::Text("Image threshold: %f\n", header.image_threshold);
		ImGui::Text("IF threshold: %f\n", header.if_threshold);
	}

	if (settings.playVideo) {
		int &ifc = transformInterface->interframe_count;
		if (!isMetaDataLoaded) {
			preloadMetaData(graph);
			eyeData.init(header.size,
			             header.level,
			             frames[0].metaData.levelData.begin());

			frameIdx = -ifc;
			prepareNextInterframeBuffer();
			start_time       = std::chrono::system_clock::now();
			frameIdx         = 0;
			isMetaDataLoaded = true;
		}
		if (fps_clock + std::chrono::milliseconds(1000 / settings.fps) <
		    std::chrono::system_clock::now()) {
			if (frames.size() >= header.interframe_count * 3) {
				if (frameIdx % ifc == 0) {
					std::thread preIFthr(prepareNextInterframeBuffer);
					preIFthr.detach();
					std::thread clearFramesThr(clearLastFrames);
					clearFramesThr.detach();
				}
			}

			eyeData.update(
			    header.level, settings.reconstructAll, settings.foveate, false);

			decodeFrame(app, graph, renderer, idx);

			total_frame_count++;
			current_frame = frameIdx;
			if (++frameIdx >= header.frame_count - header.frame_count % ifc) {
				frameIdx           = 0;
				settings.playVideo = settings.loop;
			}

			fps_clock = std::chrono::system_clock::now();
		}
	}
}

void init(App                &app,
          GLFWwindow         *window,
          RenderGraph::Graph &graph,
          Renderer           &renderer,
          const SceneConfig  &cfg)

{
	app.windows.insert("Video Compression");

	auto &transformModule = app.experiments.get("wavelet_transform");
	transformInterface =
	    &cast<WaveletTransformInterface>(transformModule.settings[0]);

	init_wavelet_transform_command_buffer =
	    transformModule.get_proc_addr<init_wavelet_transform_command_buffer_t>(
	        "init_wavelet_transform_command_buffer");
	init_interframe_tranform_command_buffer =
	    transformModule
	        .get_proc_addr<init_interframe_tranform_command_buffer_t>(
	            "init_interframe_tranform_command_buffer");

	queueIdxComp            = graph.createRenderQueue("compaction/rp");
	queueIdxRebuilt         = graph.createRenderQueue("reconstruction/rp");
	queueIdxFullReconstruct = graph.createRenderQueue("mapping/rp");

	cmd = CommandBuffer(
	    renderer.ctx.dev, renderer.ctx.commandPool, renderer.ctx.graphicsQueue);

	/**
	 * Set size for all buffers we need
	 */
	auto mappingNode = graph.getImageNode("mapped_target");
	mapping_threads  = mappingNode.width * mappingNode.height;

	auto &normBuffer = graph.getBufferNode("normalize");
	normBuffer.size =
	    2 * transformInterface->interframe_count * sizeof(NormalizationData);

	auto &wvltInfoBuffer = graph.getBufferNode("wavelets_info");
	wvltInfoBuffer.size  = 15 * sizeof(int);

	auto	     &clusterBuffer  = graph.getBufferNode("clustering");
	auto	     &waveletImgNode = graph.getImageNode("img_wavelet");
	LevelMetaData ld             = LevelMetaData();
	clusterBuffer.size           = (waveletImgNode.width / ld.block_size *
                              waveletImgNode.height / ld.block_size * 3 +
                          16) *
	                     sizeof(int);

	/// read save path from .prj file
	for (const auto &node : cfg.nodes) {
		if (node.type_string != "VideoCompression") continue;

		for (auto kv : node.settings.list) {
			kv.value_parser.str = kv.value;
			if (kv.key == "filepath") { settings.savePath = kv.value; }
		}
	}

	init_image_loading(app, window, graph, renderer, cfg);

	eyeData.readData(
	    "../projects/Wavelet_Codec_360/resources/EyeGazeExample.txt");
}

void update(App                &app,
            RenderGraph::Graph &graph,
            Renderer           &renderer,
            uint32_t            idx,
            float               delta)
{
	if (transformInterface == nullptr) {
		transformInterface = &cast<WaveletTransformInterface>(
		    app.experiments.get("wavelet_transform").settings[0]);
	}

	if (ImGui::Begin("Properties")) {
		ImGui::Text("FPS: %.2f",
		            total_frame_count /
		                std::chrono::duration<double>(
		                    std::chrono::system_clock::now() - start_time)
		                    .count());
		ImGui::Text("Frame Idx: %i", frameIdx - 1);

		if (ImGui::CollapsingHeader("Wavelet Compression")) {
			ImGui::Checkbox("Encode", &settings.encode);
			ImGui::Checkbox("Use inferframe transform",
			                &transformInterface->useInterframe);

			if (settings.encode) {
				ui_encoding(app, graph, renderer, idx);
			} else {
				ui_decoding(app, graph, renderer, idx);
			}
		}
	}
	ImGui::End();
}
}
