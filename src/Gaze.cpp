//
// Created by colin on 05.05.22.
//

#include "Gaze.h"

#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <cmath>

#include "wavelet_video_compression.h"

/**
 * 		Read eye gaze data from file. Mostly for testing.
 */
void Gaze::readData(String file_path)
{
	std::ifstream gaze_file(file_path.begin(), std::ios::in);

	std::string gaze_line;

	int line = 0;
	while (std::getline(gaze_file, gaze_line)) {
		GazeData data;
		line++;

		std::smatch timestamp;
		std::smatch position;
		std::smatch eyes_open_prob;

		if (!std::regex_search(gaze_line, timestamp, TIMESTAMP_REGEX)) {
			fprintf(stderr,
			        "ERROR reading gaze line %i: Incorrect timestamp format.\n",
			        line);
			continue;
		}

		if (!std::regex_search(gaze_line, position, POSITION_REGEX)) {
			fprintf(stderr,
			        "ERROR reading gaze line %i: Incorrect position format.\n",
			        line);
			continue;
		}

		if (!std::regex_search(gaze_line, eyes_open_prob, EYES_OPEN_REGEX)) {
			fprintf(stderr,
			        "ERROR reading gaze line %i: Incorrect format for whether "
			        "eyes are open.\n",
			        line);
			continue;
		}

		// Write timestamp
		data.timestamp = std::chrono::minutes(std::stoi(timestamp[1])) +
		                 std::chrono::seconds(std::stoi(timestamp[2])) +
		                 std::chrono::milliseconds(std::stoi(timestamp[3]));

		// Read raw position
		Vec3f raw_pos_left;
		Vec3f raw_pos_right;

		raw_pos_left.x = std::stof(position[1]);
		raw_pos_left.y = std::stof(position[2]);
		raw_pos_left.z = std::stof(position[3]);

		raw_pos_right.x = std::stof(position[4]);
		raw_pos_right.y = std::stof(position[5]);
		raw_pos_right.z = std::stof(position[6]);

		// Convert to Spherical Coordinates
		double left_inclination  = std::acos(raw_pos_left.z);
		double right_inclination = std::acos(raw_pos_right.z);

		double left_azimuth;
		if (raw_pos_left.x == 0.F && raw_pos_left.y == 0.F) {
			// Undefined
			left_azimuth = NAN;
		} else if (raw_pos_left.x == 0.F && raw_pos_left.y < 0.F) {
			left_azimuth = -M_PI_2;
		} else if (raw_pos_left.x == 0.F && raw_pos_left.y > 0.F) {
			left_azimuth = +M_PI_2;
		} else if (raw_pos_left.x < 0.F && raw_pos_left.y < 0.F) {
			left_azimuth = std::atan(raw_pos_left.y / raw_pos_left.x) - M_PI;
		} else if (raw_pos_left.x < 0.F && raw_pos_left.y >= 0.F) {
			left_azimuth = std::atan(raw_pos_left.y / raw_pos_left.x) + M_PI;
		} else {
			left_azimuth = std::atan(raw_pos_left.y / raw_pos_left.x);
		}

		double right_azimuth;
		if (raw_pos_right.x == 0.F && raw_pos_right.y == 0.F) {
			// Undefined
			right_azimuth = NAN;
		} else if (raw_pos_right.x == 0.F && raw_pos_right.y < 0.F) {
			right_azimuth = -M_PI_2;
		} else if (raw_pos_right.x == 0.F && raw_pos_right.y > 0.F) {
			right_azimuth = +M_PI_2;
		} else if (raw_pos_right.x < 0.F && raw_pos_right.y < 0.F) {
			right_azimuth = std::atan(raw_pos_right.y / raw_pos_right.x) - M_PI;
		} else if (raw_pos_right.x < 0.F && raw_pos_right.y >= 0.F) {
			right_azimuth = std::atan(raw_pos_right.y / raw_pos_right.x) + M_PI;
		} else {
			right_azimuth = std::atan(raw_pos_right.y / raw_pos_right.x);
		}

		/// Convert to 2D Coordinates
		/// Values in range [0,1]
		data.position_left.x = left_azimuth / M_PI / 2.F + 0.5F;
		data.position_left.y = left_inclination / M_PI;

		data.position_right.x = right_azimuth / M_PI / 2.F + 0.5F;
		data.position_right.y = right_inclination / M_PI;

		if (std::isnan(data.position_left.x)) data.position_left.x = 0.0;
		if (std::isnan(data.position_left.y)) data.position_left.y = 0.0;
		if (std::isnan(data.position_right.x)) data.position_right.x = 0.0;
		if (std::isnan(data.position_right.y)) data.position_right.y = 0.0;

		/// Write probability of eyes open
		data.prob_left_open  = std::stof(eyes_open_prob[1]);
		data.prob_right_open = std::stof(eyes_open_prob[2]);

		this->gaze_data.add(data);

		//		Gaze::printGazeData(data);
	}

	this->current_interval = this->gaze_data.size() - 1;
	this->refresh_duration = std::chrono::milliseconds(8);
	this->refresh_rate =
	    line * 1000.F /
	    (this->gaze_data.last().timestamp - this->gaze_data[0].timestamp)
	        .count();
}

void Gaze::printGazeData(GazeData g)
{
	auto minutes =
	    std::chrono::duration_cast<std::chrono::minutes>(g.timestamp);
	auto seconds =
	    std::chrono::duration_cast<std::chrono::seconds>(g.timestamp - minutes);
	auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
	    g.timestamp - minutes - seconds);

	printf(
	    "Timestamp: %i:%i:%i | Position Left: %f,%f | "
	    "Position Right:  %f,%f | Are Eyes Open: %f, %f\n",
	    minutes,
	    seconds,
	    milliseconds,
	    g.position_left.x,
	    g.position_left.y,
	    g.position_right.x,
	    g.position_right.y,
	    g.prob_left_open,
	    g.prob_right_open);
}

void Gaze::init(glm::ivec2 size, int max_level, LevelMetaData const *levelDatas)
{
	this->head_pos = glm::vec2(0.5, 0.5);
	eye_trajectory.push(EyePosition(glm::vec2(0.5, 0.5), glm::vec2(0.5, 0.5)));
	this->level_expansion_eye.resize(max_level + 1);
	this->sec.resize(max_level + 1);
	this->sec_viewport.resize(max_level + 1);

	this->level_expansion_eye[0] = 0.25f;

	level_block_amount.resize(max_level + 1);
	for (int i = 0; i <= max_level; i++) {
		int lvl = max_level - i;
		level_block_amount[i].x =
		    (size.x >> (lvl + 1)) / levelDatas[i].block_size;
		/// half the blocks in vertical direction due to the two eyes
		level_block_amount[i].y =
		    (size.y >> (lvl + 2)) / levelDatas[i].block_size;
	}

	this->update(max_level, false, false, true);
}

void Gaze::update(int  max_level,
                  bool reconstruct_all,
                  bool foveate,
                  bool force = false)
{
	auto delta_t = std::chrono::steady_clock::now() - this->last_update;
	if (delta_t > this->refresh_duration) {
		int update_by = static_cast<int>(delta_t / this->refresh_duration);

		if ((this->current_interval += update_by) >= this->gaze_data.size()) {
			this->current_interval = 0;
		}

		this->last_update += update_by * this->refresh_duration;
	}

	/// Here the size of the FOV is set per level. In the case that we do not
	/// foveate, the expansion of all levels equals the viewport FOV.
	for (int i = 1; i <= max_level; ++i) {
		this->level_expansion_eye[i] = this->level_expansion_eye[0];
		if (foveate) {
			if (i <= 6) {
				this->level_expansion_eye[i] = real_eye_zones[i - 1];
			} else {
				this->level_expansion_eye[i] = real_eye_zones[5];
			}
		}
	}

	/**
	 * 	In this implementation we read the eye position from a file with real
	 * eye recordings (which may be useful for testing/experiments). In a real
	 * VR scenario, we would consider the live tracking data of the eye tracker
	 * here (maybe with some trajectory extrapolation to counteract latency).
	 * The implementation with the eye tracker of course depends on the hardware
	 * and has to be included by yourself.
	 *
	 * Values in range [-.5,.5]
	 */
	eye_trajectory.push(EyePosition(
	    this->gaze_data[this->current_interval].position_left - glm::vec2(0.5),
	    this->gaze_data[this->current_interval].position_right));
	if (eye_trajectory.size() > 10) { eye_trajectory.pop(); }

	/// Here we set the sections of blocks that have to be reconstructed
	/// (depending if we use foveation). sec_vp defines the expansion of the
	/// whole viewport per level, while sec is the relative FOV per level
	/// deriving from real_eye_zones[]. When we do not use foveation sec_vp =
	/// sec.
	for (int i = 0; i <= max_level; i++) {
		if (reconstruct_all) { this->level_expansion_eye[i] = 1; }
		glm::vec2 start = {
		    std::max(this->head_pos.x - this->level_expansion_eye[0] / 2, 0.f),
		    std::max(this->head_pos.y - this->level_expansion_eye[0], 0.f)};
		glm::vec2 end = {
		    std::min(this->head_pos.x + this->level_expansion_eye[0] / 2,
		             .999f),
		    std::min(this->head_pos.y + this->level_expansion_eye[0], .999f)};

		this->sec_viewport[i].start = {level_block_amount[i].x * start.x,
		                               level_block_amount[i].y * start.y};
		this->sec_viewport[i].end   = {level_block_amount[i].x * end.x,
                                     level_block_amount[i].y * end.y};

		if (foveate) {
			start = {std::max(this->head_pos.x +
			                      eye_trajectory.back().pos_left_eye.x *
			                          this->level_expansion_eye[0] -
			                      this->level_expansion_eye[i] / 2,
			                  0.f),
			         std::max(this->head_pos.y +
			                      eye_trajectory.back().pos_left_eye.y *
			                          this->level_expansion_eye[0] -
			                      this->level_expansion_eye[i],
			                  0.f)};
			end   = {std::min(this->head_pos.x +
			                      eye_trajectory.back().pos_left_eye.x *
			                          this->level_expansion_eye[0] +
			                      this->level_expansion_eye[i] / 2,
			                  .999f),
                   std::min(this->head_pos.y +
			                      eye_trajectory.back().pos_left_eye.y *
			                          this->level_expansion_eye[0] +
			                      this->level_expansion_eye[i],
			                  .999f)};
		}

		this->sec[i].start = {level_block_amount[i].x * start.x,
		                      level_block_amount[i].y * start.y};
		this->sec[i].end   = {level_block_amount[i].x * end.x,
                            level_block_amount[i].y * end.y};
	}
}
