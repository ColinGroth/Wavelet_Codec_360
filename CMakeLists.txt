
add_module(wavelet_transform
		src/wavelet_transform.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/wavelet_transform.comp
		"${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/wavelet_transform.comp OUTPUT inverse_wavelet_transform DEFINES INVERSE"
		"${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/wavelet_transform.comp OUTPUT inverse_wavelet_transform_mask DEFINES INVERSE MASK"
		${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/wavelet_threshold.comp
		${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/interframe_transform.comp
		${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/interframe_threshold.comp)

add_module(wavelet_video_compression
        src/load_source_images.cpp
        src/wavelet_video_compression.cpp
        src/Gaze.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/compaction.comp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/rebuild.comp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/mark_non_zero.comp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/clear_image.comp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/mapping.comp)



