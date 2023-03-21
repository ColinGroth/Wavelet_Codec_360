[Include:inverse_wavelet] path = modules/compute.rg;
	shaderpath = shaders/inverse_wavelet_transform.comp.spv;
	dependency = $dependency;

	[RenderPass:inverse_wavelet/rp]
		inherit_swapchain_resolution = false;
		size = $Ignore:variables:size $Ignore:variables:size;

	[Pipeline:inverse_wavelet/pipe]
		pushconstants = 8:compute;

	[DescriptorSet:inverse_wavelet/desc]
		storageimages  = $tmp:compute;
		storagebuffers = $kernel:compute
		                 $mask:compute;