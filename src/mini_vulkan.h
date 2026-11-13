/*
 * mini_vulkan.h — High-performance multi-threaded Vulkan renderer backend.
 *
 * Implements secondary command buffer recording across worker threads,
 * triple buffering / mailbox presentation, pipeline state caching,
 * dynamic vertex staging, and texture atlas uploading.
 */
#ifndef MINI_VULKAN_H
#define MINI_VULKAN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct MiniRenderer;
typedef struct MiniVkContext MiniVkContext;

/* Initialize multi-threaded Vulkan renderer for the given GLFW window */
MiniVkContext *mini_vk_create(void *glfw_window, int width, int height, int vsync, int sample_count);

/* Tear down Vulkan instance, device, pipelines, and worker threads */
void mini_vk_destroy(MiniVkContext *vk);

/* Begin a new frame */
void mini_vk_begin_frame(MiniVkContext *vk, int width, int height);

/* Multi-threaded flush of all MiniVectorBuffer commands */
void mini_vk_flush(MiniVkContext *vk, struct MiniRenderer *r);

/* Finish frame and present to the display */
void mini_vk_end_frame(MiniVkContext *vk);

/* Update dirty region in the 2048x2048 texture atlas */
void mini_vk_update_atlas(MiniVkContext *vk, int x, int y, int w, int h,
                          const uint32_t *src_rgba, int atlas_w, int atlas_h);

/* Read back current frame image as PNG */
int mini_vk_screenshot_png(MiniVkContext *vk, uint8_t **out_png, size_t *out_len);

/* Notify backend of framebuffer resize */
void mini_vk_resize(MiniVkContext *vk, int width, int height);

#ifdef __cplusplus
}
#endif

#endif /* MINI_VULKAN_H */
