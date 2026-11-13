/*
 * mini_vk_shaders.h — Precompiled SPIR-V shaders for MiniRenderer Vulkan backend.
 * Compiled with Vulkan SDK (D:\vulkan\Bin\glslc.exe -O).
 */
#ifndef MINI_VK_SHADERS_H
#define MINI_VK_SHADERS_H

#include <stdint.h>
#include <stddef.h>

static const uint32_t MINI_VK_VERT_SPV[] =
#include "shaders/vert_spv.inc"
;
static const size_t MINI_VK_VERT_SPV_SIZE = sizeof(MINI_VK_VERT_SPV);

static const uint32_t MINI_VK_FRAG_SPV[] =
#include "shaders/frag_spv.inc"
;
static const size_t MINI_VK_FRAG_SPV_SIZE = sizeof(MINI_VK_FRAG_SPV);

#endif /* MINI_VK_SHADERS_H */
