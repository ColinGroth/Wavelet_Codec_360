[Image:source_image]
	size                 = $Ignore:variables:size $Ignore:variables:size;
	format               = rgba32;
	transfer_source      = true;
	transfer_destination = true;

[Image:img_wavelet]
	size                 = $Ignore:variables:size $Ignore:variables:size;
	format               = rgba32;
	arraysize            = 8;       #increase when using more interframes. arraysize must be interframe_size * 2!
	transfer_destination = true;
	transfer_source      = true;

[DescriptorSet:output_source]
	textures = source_image:allgraphics;

[Buffer:settings]
	hostvisible = true;

[Buffer:kernel]
	hostvisible = true;

[Buffer:normalize]
    hostvisible = true;

[DescriptorSet:output_img_wavelet]
	textures = img_wavelet:allgraphics;
	sampler = nearest;

[Include:wavelet] path = modules/wavelet_transform.rg;
	tmp      = img_wavelet;
	kernel   = kernel;

[Include:approx] path = modules/compute.rg;
	shaderpath = shaders/wavelet_threshold.comp.spv;
	dependency = wavelet/wavelet/rp;

	[Pipeline:approx/pipe]
		pushconstants  = 4:compute;

	[DescriptorSet:approx/desc]
		storageimages  = img_wavelet:compute;
		ubos           = settings:compute;

[Include:interframe] path = modules/compute.rg;
	shaderpath = shaders/interframe_transform.comp.spv;
	dependency = approx/rp;

	[Pipeline:interframe/pipe]
		pushconstants  = 8:compute;

	[DescriptorSet:interframe/desc]
		storageimages  = img_wavelet:compute;

[Include:interframe_approx] path = modules/compute.rg;
	shaderpath = shaders/interframe_threshold.comp.spv;
	dependency = interframe/rp;

	[Pipeline:interframe_approx/pipe]
		pushconstants  = 4:compute;

	[DescriptorSet:interframe_approx/desc]
		storageimages  = img_wavelet:compute;
		ubos           = settings:compute;
		storagebuffers = normalize:compute;

[Include:prefix_sum_images] path = modules/cdf_images.rg;
	[Image:prefix_sum_images/pdf]
		size = $Ignore:variables:size $Ignore:variables:size;

	[Image:prefix_sum_images/cdf]
		transfer_source = true;

[Include:masking] path = modules/compute.rg;
	shaderpath = shaders/mark_non_zero.comp.spv;
	dependency = interframe_approx/rp;

	[Pipeline:masking/pipe]
		descriptorsets = masking/desc;
		pushconstants = 12:compute;

	[DescriptorSet:masking/desc]
		storageimages = img_wavelet:compute
		                prefix_sum_images/pdf:compute;

[Include:prefix_sum] path = modules/compute_cdf.rg;
	pdf_image     = prefix_sum_images/pdf;
	cdf_image     = prefix_sum_images/cdf;
	tmp_cdf_image = prefix_sum_images/cdf_tmp;
	dependency    = masking/rp;

[Buffer:compacted]
	hostvisible = true;

[Buffer:clustering]
    hostvisible = true;

[Include:compaction] path = modules/compute.rg;
	shaderpath = shaders/compaction.comp.spv;
	dependency = prefix_sum/cdf_v/rp;

	[Pipeline:compaction/pipe]
		descriptorsets = compaction/desc
		                 prefix_sum_images/cdf_images;
		pushconstants = 12:compute;

	[DescriptorSet:compaction/desc]
		textures       = img_wavelet:compute;
		storagebuffers = compacted:compute
		                 normalize:compute
		                 clustering:compute;

#############################################################
# Reconstruction
#############################################################

[Include:clear_image] path = modules/compute.rg;
	shaderpath = shaders/clear_image.comp.spv;

	[DescriptorSet:clear_image/desc]
		storageimages  = img_wavelet:compute;

[Buffer:wavelets_info]
	hostvisible = true;

[Include:reconstruction] path = modules/compute.rg;
	shaderpath = shaders/rebuild.comp.spv;

    [DescriptorSet:reconstruction/desc]
		storageimages  = img_wavelet:compute;
		storagebuffers = compacted:compute
		                 wavelets_info:compute
		                 normalize:compute;

	[Pipeline:reconstruction/pipe]
		descriptorsets = reconstruction/desc;
	    pushconstants  = 20:compute;

[Buffer:synthesis_kernel]
    hostvisible = true;

[Include:inverse] path = modules/inverse_wavelet_transform.rg;
	kernel     = synthesis_kernel;
	mask       = clustering;
	tmp        = img_wavelet;
	dependency = reconstruction/rp;

[Image:mapped_target]
	size                 = 1024 1024;
	format               = rgba8;
	transfer_source		 = true;

[DescriptorSet:output_mapped_target]
	textures = mapped_target:allgraphics;
