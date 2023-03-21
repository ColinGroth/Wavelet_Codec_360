#pragma once

#include "rendergraph.h"

#define MAX_KERNEL_SIZE 20

struct CompressedWavelet
{
	uint32_t index;
	uint32_t coefficients;
	CompressedWavelet() = default;
	CompressedWavelet(uint32_t idx, uint32_t coeff)
	{
		index        = idx;
		coefficients = coeff;
	}
};

struct NormalizationData
{
	float min_r;
	float min_g;
	float min_b;
	float max_r;
	float max_g;
	float max_b;
};

struct WaveletTransformInterface
{
	int   forward_iterations   = 6;
	float image_threshold      = 0.02f;
	float interframe_threshold = 0.00f;
	int   interframe_count     = 4;
	bool  useInterframe        = true;
};

enum WaveletTransformType {
	RECONSTRUCTION,
	DECOMPOSITION,
	INTERFRAME,
	NONE
};
const static struct {
	WaveletTransformType type;
	String 				 str;
}
wtt_conversion [] = {
    {RECONSTRUCTION, "reconstruction"},
    {DECOMPOSITION, "decomposition"},
    {INTERFRAME, "interframe"},
};
inline WaveletTransformType string_to_wtt(String const& name) {
	for (int i = 0; i < sizeof(wtt_conversion) / sizeof(wtt_conversion[0]); ++i) {
		if (name == wtt_conversion[i].str) return wtt_conversion[i].type;
	}
	fprintf(stderr, "Unknown Wavelet Transform Type '%s'\n", name.begin());
	assert(false);
	exit(EXIT_FAILURE);
}

struct WaveletTransformSettings
{
	WaveletTransformType type = WaveletTransformType::NONE;
	String input_image;
	String output_image;
	String tmp_image;
	String pass_name;
	String kernel_name;
	struct KernelBuffer
	{
		float lp[MAX_KERNEL_SIZE];
		float hp[MAX_KERNEL_SIZE];
	} kernel_buffer;
	uint32_t kernel_size;
};

typedef void (*init_wavelet_transform_command_buffer_t)(
    Renderer                       &renderer,
    RenderGraph::Graph             &graph,
    const WaveletTransformSettings &transform,
    bool                            changed,
    int                             idx,
    int                             level,
    int                            *compute_threads_inv,
    int                             useHeightFactor,
    int							   *block_size);

typedef void (*init_interframe_tranform_command_buffer_t)(
    Renderer                       &renderer,
    RenderGraph::Graph             &graph,
    const WaveletTransformSettings &transform,
    bool                            changed,
    int                             idx);

