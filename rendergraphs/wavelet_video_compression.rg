[Ignore:variables]
#	size = 8192;
	size = 4096;
#	size = 1024;

[Include] path = modules/wavelet_video_module.rg;

######## Imgui output
## Buffers for the GUI itself
[Include:imgui] path = modules/imgui.rg;

	[Image:imgui/output]
		transfer_destination = true;
		size                 = $Ignore:variables:size $Ignore:variables:size;

