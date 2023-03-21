[Include:wavelet] path = modules/compute.rg;
	shaderpath = shaders/wavelet_transform.comp.spv;

	[Pipeline:wavelet/pipe]
		pushconstants  = 16:compute;

	[DescriptorSet:wavelet/desc]
		storageimages  = $tmp:compute;
		storagebuffers = $kernel:compute;
	
	[RenderPass:wavelet/rp]
		dependencies = $dependency;

