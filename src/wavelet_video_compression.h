#pragma once

#include <mutex>
#include "Gaze.h"

struct GuiSettings
{
	String savePath;
	bool   encode          = true;
	bool   loop            = true;
	bool   playVideo       = false;
	bool   showWvltImg     = false;
	int    fps             = -1;
	int    debug           = 0;
	bool   reconstructAll  = false;
	bool   foveate         = false;
	bool   useHeightFactor = true;
	int    encodingTotal   = 12;
};

struct WBenchmark
{
	String description;
#if (WIN32)
	std::chrono::time_point<std::chrono::steady_clock> start_time =
	    std::chrono::high_resolution_clock::now();
	std::chrono::time_point<std::chrono::steady_clock,
	                        std::chrono::duration<double>>
	    end_time;
#else
	std::chrono::time_point<std::chrono::system_clock> start_time =
	    std::chrono::high_resolution_clock::now();
	std::chrono::time_point<std::chrono::system_clock,
	                        std::chrono::duration<double>>
	    end_time;
#endif
};

inline WBenchmark benchmarks;

inline void bench_start(String name)
{
	WBenchmark new_bench  = WBenchmark();
	new_bench.description = name;
	benchmarks = new_bench;
}

inline void bench_end()
{
	// benchmarks.back().end_time = std::chrono::high_resolution_clock::now();
	benchmarks.end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration =
	    (benchmarks.end_time - benchmarks.start_time) * 1000;
	std::cout << benchmarks.description.begin() << ": " << duration.count()
	          << " ms" << std::endl;
}

struct LevelPreloadInfo
{
	size_t     wavelet_count      = 0;
	int        mask_count         = 0;
	glm::ivec2 first_block        = {-1, -1};
	glm::ivec2 last_block         = {-1, -1};
	glm::ivec2 former_first       = {-1, -1};
	glm::ivec2 former_last        = {-1, -1};
	bool       areBlockEndsLoaded = false;
};

struct LevelMetaData
{
	size_t          level_end;
	int             block_size = 32;
	int             blocks_horiz;
	int             blocks_vert;
	size_t          block_start_marker;
	size_t          block_end_marker;
	Array<uint32_t> block_ends = {};
};

struct FrameMetaData
{
	size_t               frame_start_marker;
	size_t               frame_end_marker;
	int                  total_blocks_size;
	NormalizationData    approxNorm;
	NormalizationData    waveletsNorm;
	Array<LevelMetaData> levelData;
};

struct Frame
{
	FrameMetaData                   metaData;
	Array<Array<CompressedWavelet>> dataPerLevel;
	Array<Array<int>>               maskPerLevel;
	Array<LevelPreloadInfo>         preloadInfo;
};

struct WaveletFileHeader
{
	int        VERSION = 1;
	glm::ivec2 size;
	int        frame_count;
	int        level;
	int        interframe_count;
	float      image_threshold;
	float      if_threshold;
};

inline uint32_t      queueIdxComp;
inline uint32_t      queueIdxRebuilt;
inline uint32_t      queueIdxFullReconstruct;
inline int           frameIdx          = 0;
inline size_t        total_frame_count = 0;
inline bool          isMetaDataLoaded  = false;
inline std::mutex    driveBusyMutex;
inline int           compute_threads = 0;
inline int           mapping_threads;
inline CommandBuffer cmd;

inline auto fps_clock  = std::chrono::system_clock::now();
inline auto start_time = std::chrono::system_clock::now();

inline WaveletFileHeader header;
inline Array<Frame>      frames;
inline size_t            metaData_size;
inline Gaze              eyeData;
