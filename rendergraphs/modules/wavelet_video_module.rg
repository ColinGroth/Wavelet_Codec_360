[Include] path = modules/wavelet_video_bare.rg;

[Include:mapping] path = modules/compute.rg;
    shaderpath = shaders/mapping.comp.spv;
    dependency = inverse/inverse_wavelet/rp;

    [DescriptorSet:mapping/desc]
        storageimages  = img_wavelet:compute
                         mapped_target:compute;

    [Pipeline:mapping/pipe]
        descriptorsets = mapping/desc;

