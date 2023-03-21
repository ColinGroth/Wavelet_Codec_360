# Fast Wavelet-Based Codec for 360° Videos

<img src='resources/banner.jpg'>

### [Website](https://graphics.tu-bs.de/publications/groth2023wavelet-based) | [Paper](https://graphics.tu-bs.de/upload/publications/Groth2023Wavelets_CR.pdf) | [Video](https://www.youtube.com/watch?v=VqgEgkRDEiE) <br>

Official implementation of the wavelet-based video codec described in our paper 'Wavelet-Based Fast Decoding of 360° Videos' published in IEEE Transactions on Visualization and Computer Graphics.

In this work, wavelet transforms are used for inter- and intra-frame coding of video footage. 
The focus of the codec is fast video playback of high-quality, high frame-rate videos with a wide field of view, e.g. 360° videos.
Generally, the output of the rendering (the decoded frames) may be displayed on a monitor or shown inside some VR glasses. 
However, a dynamic display output is not provided here and may be customized to user-specific hardware.  

## Prerequisites
- Linux or Windows
- GPU framework (see 'Graphics Interface')

## How to Use
### General Structure
The *wavelet_video_compression* source file drives the main control for the video coding. It also sets the playback specific video player attributes for the imgui interface.
The transform is handled in the *wavelet_transform.cpp* where also the GPU buffer are filled with information necessary for the shaders. 
The *load_source_images* source handles the loading of the video frames into the graphics card memory for encoding.

A good overview of the shaders and the order in which they are executed is given in the *wavelet_video_compression.rg* rendergraph file. In the file, relationships are defined by the *dependency* parameter.
All input parameters are parsed from the *wavelet_video_compression.prj* project file. Here, e.g., the input and output directories can be set.

NOTE: the default output in which the decoded frames are written is *output_img_wavelet* and NOT *imgui/output*!

### Graphics Interface
Note, that a codec is not a full application. Rather we provide the shaders and CPU-based implementation necessary for the encoding and decoding logic as described in our paper.
In our work, we used a Vulkan-based framework for GPU based processing and execution. 
This  framework is called *mtstudio* and developed by [Sascha Fricke](https://graphics.tu-bs.de/people/fricke). 
The framework is publicly available under the following links:

- For the core framework: https://git.cg.cs.tu-bs.de/Sascha/mtstudio
- For the Vulkan renderer: https://git.cg.cs.tu-bs.de/Sascha/vkrenderer

Please note, that the framework has no proper documentation yet and that we cannot handle all question on how to use it. 
Also, the module to compute the cumulative distribution function (cdf) can not be provided at this point and has to be implemented by yourself. We are confident to provide all utility modules in the near future.    
However, although our implementation includes the framework-specific function calls you may want to replace them with your own GPU backend using Vulkan, OpenGL or similar. 

## Citation
If you use our code for your publications, please cite our [paper](https://graphics.tu-bs.de/upload/publications/Groth2023Wavelets_CR.pdf) using the following BibTeX:
```
@InProceedings{Groth23wavelets,
  title = {Wavelet-Based Fast Decoding of 360\textdegree  Videos},
  author = {Groth, Colin and Fricke, Sascha and Castillo, Susana  and Magnor, Marcus},
  journal = {{IEEE} Transactions on Visualization and Computer Graphics ({TVCG}, Proc. {IEEE} {VR})},
  year = {2023}
}
```

## Acknowledgments
This work was partially funded by the German Science Foundation (DFG MA2555/15-1 'Immersive Digital Reality'), and under Germany’s Excellence Strategy within the Cluster of Excellence PhoenixD (EXC 2122, Project ID 390833453).
