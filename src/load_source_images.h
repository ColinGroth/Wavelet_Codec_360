#pragma once


void load_entries();

bool load_next_textures(App                &app,
                        RenderGraph::Graph &graph,
                        Renderer           &renderer);

void init_image_loading(App                &app,
                        GLFWwindow         *window,
                        RenderGraph::Graph &graph,
                        Renderer           &renderer,
                        const SceneConfig  &cfg);

extern unsigned int  images_total;
extern unsigned int  current_image_idx;
