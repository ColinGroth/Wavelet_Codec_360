#pragma once

#include <chrono>
#include <regex>

#include "glm_include.h"

#include "list.h"
#include "str.h"
#include "vec.h"
#include "wavelet_transform.h"

#include <memory>
#include <queue>

struct LevelMetaData;

struct GazeData
{
	std::chrono::milliseconds timestamp;
	glm::vec2                 position_left;
	glm::vec2                 position_right;
	float                     prob_left_open;
	float                     prob_right_open;
};

class Gaze
{
private:
	Array<GazeData> gaze_data;

	size_t                                             current_interval;
	float                                              refresh_rate;
	std::chrono::milliseconds                          refresh_duration;
	std::chrono::time_point<std::chrono::steady_clock> last_update;

	/// regular expressions to parse the example eye data file
	inline static const std::regex TIMESTAMP_REGEX =
	    std::regex("(\\d+)\\.(\\d+).(\\d+):");
	inline static const std::regex POSITION_REGEX = std::regex(
	    "X=(-?\\d\\.\\d+) Y=(-?\\d\\.\\d+) Z=(-?\\d\\.\\d+);X=(-?\\d\\.\\d+) "
	    "Y=(-?\\d\\.\\d+) Z=(-?\\d\\.\\d+)");
	inline static const std::regex EYES_OPEN_REGEX =
	    std::regex("\\| (\\d.\\d+);(\\d.\\d+)");

public:
	Gaze() = default;

	struct Section
	{
		///  BLOCK-based area of the FOV in the frame given a start and end
		///  block
		glm::ivec2 start, end;
	};

	Array<glm::ivec2> level_block_amount;  /// horizontal, vertical

	glm::vec2      head_pos;
	Array<Section> sec_viewport;  /// sections of the viewport (full FOV)

	struct EyePosition
	{
		glm::vec2 pos_right_eye{}, pos_left_eye{};
		EyePosition(glm::vec2 right_eye, glm::vec2 left_eye)
		{
			pos_right_eye = right_eye;
			pos_left_eye  = left_eye;
		}
	};
	std::queue<EyePosition> eye_trajectory;
	Array<float>            level_expansion_eye;
	/// sections (FoV in blocks) of the eye per level. sec == sec_viewport if
	/// foviation is not used
	Array<Section> sec;

	/// actual expansion (FOVs) of the different "quality layers" of the human
	/// eye
	float real_eye_zones[6] = {
	    90.f / 360, 35.f / 360, 18.f / 360, 8.f / 360, 5.f / 360, 1.5f / 360};

	static void printGazeData(GazeData g);
	void        readData(String file_path);

	void init(glm::ivec2 size, int max_level, LevelMetaData const *levelDatas);
	void update(int max_level, bool reconstruct_all, bool foveate, bool force);

	/// check if eye position changed to update the data accordingly
	bool operator!=(Gaze &otherEyeData)
	{
		return (glm::length(this->eye_trajectory.back().pos_left_eye -
		                    otherEyeData.eye_trajectory.back().pos_left_eye) >
		            0.01 ||
		        glm::length(this->eye_trajectory.back().pos_right_eye -
		                    otherEyeData.eye_trajectory.back().pos_right_eye) >
		            0.01);
	}
};
