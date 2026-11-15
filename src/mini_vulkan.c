/*
 * mini_vulkan.c — High-performance multi-threaded Vulkan renderer backend.
 *
 * Implements secondary command buffer recording across worker threads,
 * triple buffering / mailbox presentation, pipeline state caching,
 * dynamic vertex staging, and texture atlas uploading.
 */

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mini_vulkan.h"
#include "mini_renderer.h"
#include "mini_png.h"
#include "mini_log.h"
#include "mini_vk_shaders.h"

#define MAX_FRAMES_IN_FLIGHT 2
#define MAX_FLUSHES_PER_FRAME 16
#define MINI_VK_MAX_THREADS 4
#define MINI_VK_MAX_VERTICES_PER_WORKER (256 * 1024) /* 256K vertices per thread per frame */
#define ATLAS_WIDTH 2048
#define ATLAS_HEIGHT 2048
#define PI 3.14159265359f

typedef struct MiniVkVertex {
    float x, y;
    float u, v;
    float r, g, b, a;
} MiniVkVertex;

typedef struct MiniVkPoint {
    float x, y;
} MiniVkPoint;

typedef struct MiniVkPushConstants {
    float projection[16];
    float model[16];
} MiniVkPushConstants;

/* Forward declaration */
struct MiniVkContext;

typedef struct MiniVkWorker {
#ifdef _WIN32
    HANDLE thread_handle;
    HANDLE start_event;
    HANDLE done_event;
#else
    pthread_t thread_handle;
    pthread_mutex_t mutex;
    pthread_cond_t start_cond;
    pthread_cond_t done_cond;
    int work_ready;
    int work_done;
#endif
    int thread_id;
    int should_exit;
    struct MiniVkContext *vk;

    /* Per-frame command pools & secondary buffers */
    VkCommandPool cmd_pool[MAX_FRAMES_IN_FLIGHT];
    VkCommandBuffer secondary_cb[MAX_FRAMES_IN_FLIGHT][MAX_FLUSHES_PER_FRAME];
    VkCommandBuffer secondary_cb_current;
    VkRenderPass render_pass;

    /* Task parameters */
    const struct MiniRenderer *r;
    uint32_t cmd_start;
    uint32_t cmd_count;
    uint32_t frame_index;
    uint32_t image_index;
    int viewport_w, viewport_h;

    /* Vertex output slice */
    MiniVkVertex *vertex_out;
    uint32_t max_vertices;
    uint32_t vertex_offset_in_global_buffer;
    uint32_t vertices_written;
} MiniVkWorker;

struct MiniVkContext {
    GLFWwindow *window;
    int width, height;
    int vsync;
    int sample_count;

    VkInstance instance;
    VkSurfaceKHR surface;
    VkPhysicalDevice physical_device;
    VkDevice device;
    uint32_t graphics_queue_family;
    VkQueue graphics_queue;
    VkQueue present_queue;

    /* Swapchain */
    VkSwapchainKHR swapchain;
    VkFormat swapchain_format;
    VkExtent2D swapchain_extent;
    uint32_t swapchain_image_count;
    VkImage swapchain_images[8];
    VkImageView swapchain_image_views[8];
    VkFramebuffer framebuffers[8];

    /* Render pass (clear & load) */
    VkRenderPass render_pass;
    VkRenderPass render_pass_clear;
    VkRenderPass render_pass_load;

    /* Descriptor set layout & pipeline */
    VkDescriptorSetLayout desc_set_layout;
    VkDescriptorPool desc_pool;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline_alpha;
    VkPipeline pipeline_diff;

    /* Atlas Texture (2048x2048 RGBA8) */
    VkImage atlas_image;
    VkDeviceMemory atlas_memory;
    VkImageView atlas_view;
    VkSampler atlas_sampler;
    VkDescriptorSet atlas_desc_set;

    /* Atlas Staging buffer & dirty tracking */
    int atlas_dirty;
    int dirty_min_x, dirty_min_y, dirty_max_x, dirty_max_y;
    VkBuffer atlas_staging_buffer;
    VkDeviceMemory atlas_staging_memory;
    uint32_t *atlas_staging_mapped;

    /* Dynamic vertex buffers (per frame in flight) */
    VkBuffer vertex_buffer[MAX_FRAMES_IN_FLIGHT];
    VkDeviceMemory vertex_memory[MAX_FRAMES_IN_FLIGHT];
    MiniVkVertex *vertex_mapped[MAX_FRAMES_IN_FLIGHT];

    /* Primary command pools & buffers */
    VkCommandPool primary_cmd_pool[MAX_FRAMES_IN_FLIGHT];
    VkCommandBuffer primary_cb[MAX_FRAMES_IN_FLIGHT];

    /* Sync objects */
    VkSemaphore image_available_sem[MAX_FRAMES_IN_FLIGHT];
    VkSemaphore render_finished_sem[MAX_FRAMES_IN_FLIGHT];
    VkFence in_flight_fence[MAX_FRAMES_IN_FLIGHT];

    /* Multi-threaded Worker Pool */
    MiniVkWorker workers[MINI_VK_MAX_THREADS];
    int num_workers;

    /* Pre-pass command state buffers */
    struct Mat4 *cmd_matrices;
    VkRect2D *cmd_scissors;
    uint8_t *cmd_has_xform;
    uint32_t cmd_states_cap;

    uint32_t current_frame;
    uint32_t current_image_index;
    int in_frame;
    uint32_t flush_count_in_frame;
    uint32_t worker_vert_used[MINI_VK_MAX_THREADS];
    int framebuffer_resized;
    float clear_color[4];
};

/* ------------------------------------------------------------------ */
/* Matrix Mathematics                                                 */
/* ------------------------------------------------------------------ */
typedef struct Mat4 {
    float m[16];
} Mat4;

static inline Mat4 mat4_ident(void) {
    Mat4 r;
    memset(r.m, 0, sizeof(r.m));
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

static inline Mat4 mat4_mul(Mat4 a, Mat4 b) {
    Mat4 out;
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) {
            out.m[c * 4 + r] =
                a.m[0 * 4 + r] * b.m[c * 4 + 0] +
                a.m[1 * 4 + r] * b.m[c * 4 + 1] +
                a.m[2 * 4 + r] * b.m[c * 4 + 2] +
                a.m[3 * 4 + r] * b.m[c * 4 + 3];
        }
    }
    return out;
}

static inline Mat4 mat4_make_translation(float tx, float ty, float tz) {
    Mat4 r = mat4_ident();
    r.m[12] = tx;
    r.m[13] = ty;
    r.m[14] = tz;
    return r;
}

static inline Mat4 mat4_make_scale(float sx, float sy, float sz) {
    Mat4 r = mat4_ident();
    r.m[0] = sx;
    r.m[5] = sy;
    r.m[10] = sz;
    return r;
}

static inline Mat4 mat4_make_rot_z(float rad) {
    Mat4 r = mat4_ident();
    float c = cosf(rad);
    float s = sinf(rad);
    r.m[0] = c;  r.m[4] = -s;
    r.m[1] = s;  r.m[5] = c;
    return r;
}

static inline void mat4_transform_pt(const Mat4 *m, float *x, float *y) {
    float px = *x;
    float py = *y;
    float rx = px * m->m[0] + py * m->m[4] + m->m[12];
    float ry = px * m->m[1] + py * m->m[5] + m->m[13];
    float rw = px * m->m[3] + py * m->m[7] + m->m[15];
    if (rw != 1.0f && rw != 0.0f) {
        rx /= rw;
        ry /= rw;
    }
    *x = rx;
    *y = ry;
}

static void mat4_identity(float *m) {
    memset(m, 0, 16 * sizeof(float));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

static void mat4_ortho(float *m, float width, float height) {
    memset(m, 0, 16 * sizeof(float));
    /* Map (0, 0) -> (-1, -1), (width, height) -> (1, 1) in Vulkan clip space */
    m[0]  = (width > 0.0f) ? (2.0f / width) : 1.0f;
    m[5]  = (height > 0.0f) ? (2.0f / height) : 1.0f;
    m[10] = 1.0f / 4000.0f;
    m[12] = -1.0f;
    m[13] = -1.0f;
    m[14] = 0.5f;
    m[15] = 1.0f;
}

static void mat4_multiply(float *out, const float *a, const float *b) {
    float res[16];
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            res[r * 4 + c] =
                a[r * 4 + 0] * b[0 * 4 + c] +
                a[r * 4 + 1] * b[1 * 4 + c] +
                a[r * 4 + 2] * b[2 * 4 + c] +
                a[r * 4 + 3] * b[3 * 4 + c];
        }
    }
    memcpy(out, res, 16 * sizeof(float));
}

/* ------------------------------------------------------------------ */
/* Vulkan Memory Helper                                               */
/* ------------------------------------------------------------------ */
static uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return 0;
}

static int create_buffer(MiniVkContext *vk, VkDeviceSize size, VkBufferUsageFlags usage,
                         VkMemoryPropertyFlags properties, VkBuffer *buffer, VkDeviceMemory *memory) {
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    if (vkCreateBuffer(vk->device, &buffer_info, NULL, buffer) != VK_SUCCESS)
        return -1;

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(vk->device, *buffer, &mem_reqs);

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = find_memory_type(vk->physical_device, mem_reqs.memoryTypeBits, properties)
    };
    if (vkAllocateMemory(vk->device, &alloc_info, NULL, memory) != VK_SUCCESS)
        return -1;

    vkBindBufferMemory(vk->device, *buffer, *memory, 0);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Geometry Tessellation Helpers                                       */
/* ------------------------------------------------------------------ */
static inline void emit_v(MiniVkVertex *v, float x, float y, float u, float v_coord, float r, float g, float b, float a) {
    v->x = x; v->y = y;
    v->u = u; v->v = v_coord;
    v->r = r; v->g = g; v->b = b; v->a = a;
}

static inline void emit_quad(MiniVkVertex *out, uint32_t *v_count,
                            float x0, float y0, float u0, float v0,
                            float x1, float y1, float u1, float v1,
                            float r, float g, float b, float a) {
    MiniVkVertex *v = out + (*v_count);
    emit_v(&v[0], x0, y0, u0, v0, r, g, b, a);
    emit_v(&v[1], x1, y0, u1, v0, r, g, b, a);
    emit_v(&v[2], x1, y1, u1, v1, r, g, b, a);

    emit_v(&v[3], x0, y0, u0, v0, r, g, b, a);
    emit_v(&v[4], x1, y1, u1, v1, r, g, b, a);
    emit_v(&v[5], x0, y1, u0, v1, r, g, b, a);
    *v_count += 6;
}

static inline void emit_quad_fade(MiniVkVertex *out, uint32_t *v_count,
                                  float p0x, float p0y, float a0,
                                  float p1x, float p1y, float a1,
                                  float p2x, float p2y, float a2,
                                  float p3x, float p3y, float a3,
                                  float r, float g, float b) {
    MiniVkVertex *v = out + (*v_count);
    emit_v(&v[0], p0x, p0y, -1.0f, -1.0f, r, g, b, a0);
    emit_v(&v[1], p1x, p1y, -1.0f, -1.0f, r, g, b, a1);
    emit_v(&v[2], p2x, p2y, -1.0f, -1.0f, r, g, b, a2);

    emit_v(&v[3], p0x, p0y, -1.0f, -1.0f, r, g, b, a0);
    emit_v(&v[4], p2x, p2y, -1.0f, -1.0f, r, g, b, a2);
    emit_v(&v[5], p3x, p3y, -1.0f, -1.0f, r, g, b, a3);
    *v_count += 6;
}

static inline void calc_grad_color(float px, float py,
                                   float bx, float by, float bw, float bh,
                                   int type, float angle_deg,
                                   float r1, float g1, float b1, float a1,
                                   float r2, float g2, float b2, float a2,
                                   float *out_r, float *out_g, float *out_b, float *out_a) {
    float cx = bx + bw * 0.5f;
    float cy = by + bh * 0.5f;
    float dx = px - cx;
    float dy = py - cy;
    float t = 0.5f;

    if (type == 1) {
        /* Radial gradient */
        float D = sqrtf((bw * 0.5f) * (bw * 0.5f) + (bh * 0.5f) * (bh * 0.5f));
        t = sqrtf(dx * dx + dy * dy) / (D > 1e-6f ? D : 1.0f);
    } else if (type == 2) {
        /* Conic gradient */
        float pa = atan2f(dx, -dy);
        if (pa < 0.0f) pa += 6.28318530718f;
        float deg = pa * 57.2957795f - angle_deg;
        deg = fmodf(deg, 360.0f);
        if (deg < 0.0f) deg += 360.0f;
        t = deg / 360.0f;
    } else {
        /* Linear gradient with CSS angle semantics */
        float a_rad = angle_deg * (PI / 180.0f);
        float dirx = sinf(a_rad);
        float diry = -cosf(a_rad);
        float L = fabsf(bw * dirx) + fabsf(bh * diry);
        if (L < 1e-6f) L = 1e-6f;
        t = (dx * dirx + dy * diry) / L + 0.5f;
    }

    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    *out_r = r1 * (1.0f - t) + r2 * t;
    *out_g = g1 * (1.0f - t) + g2 * t;
    *out_b = b1 * (1.0f - t) + b2 * t;
    *out_a = a1 * (1.0f - t) + a2 * t;
}

static inline void emit_xform_quad_aa(MiniVkVertex *v_base, uint32_t *count, uint32_t max_v,
                                      const Mat4 *m, float x0, float y0, float x1, float y1,
                                      float r, float g, float b, float a) {
    if (*count + 6 > max_v) return;
    int is_rot = (fabsf(m->m[1]) > 1e-4f || fabsf(m->m[4]) > 1e-4f);
    if (!is_rot || (x1 - x0) <= 1.0f || (y1 - y0) <= 1.0f || *count + 30 > max_v) {
        float p0x = x0, p0y = y0, p1x = x1, p1y = y0, p2x = x1, p2y = y1, p3x = x0, p3y = y1;
        mat4_transform_pt(m, &p0x, &p0y);
        mat4_transform_pt(m, &p1x, &p1y);
        mat4_transform_pt(m, &p2x, &p2y);
        mat4_transform_pt(m, &p3x, &p3y);
        MiniVkVertex *v = v_base + *count;
        emit_v(&v[0], p0x, p0y, -1, -1, r, g, b, a);
        emit_v(&v[1], p1x, p1y, -1, -1, r, g, b, a);
        emit_v(&v[2], p2x, p2y, -1, -1, r, g, b, a);
        emit_v(&v[3], p0x, p0y, -1, -1, r, g, b, a);
        emit_v(&v[4], p2x, p2y, -1, -1, r, g, b, a);
        emit_v(&v[5], p3x, p3y, -1, -1, r, g, b, a);
        *count += 6;
        return;
    }
    float ix0 = x0 + 0.5f, iy0 = y0 + 0.5f, ix1 = x1 - 0.5f, iy1 = y1 - 0.5f;
    float ox0 = x0 - 0.5f, oy0 = y0 - 0.5f, ox1 = x1 + 0.5f, oy1 = y1 + 0.5f;
    float ip[4][2] = {{ix0, iy0}, {ix1, iy0}, {ix1, iy1}, {ix0, iy1}};
    float op[4][2] = {{ox0, oy0}, {ox1, oy0}, {ox1, oy1}, {ox0, oy1}};
    for (int k = 0; k < 4; k++) {
        mat4_transform_pt(m, &ip[k][0], &ip[k][1]);
        mat4_transform_pt(m, &op[k][0], &op[k][1]);
    }
    MiniVkVertex *v = v_base + *count;
    emit_v(&v[0], ip[0][0], ip[0][1], -1, -1, r, g, b, a);
    emit_v(&v[1], ip[1][0], ip[1][1], -1, -1, r, g, b, a);
    emit_v(&v[2], ip[2][0], ip[2][1], -1, -1, r, g, b, a);
    emit_v(&v[3], ip[0][0], ip[0][1], -1, -1, r, g, b, a);
    emit_v(&v[4], ip[2][0], ip[2][1], -1, -1, r, g, b, a);
    emit_v(&v[5], ip[3][0], ip[3][1], -1, -1, r, g, b, a);
    v += 6;
    for (int k = 0; k < 4; k++) {
        int nk = (k + 1) & 3;
        emit_v(&v[0], ip[k][0],  ip[k][1],  -1, -1, r, g, b, a);
        emit_v(&v[1], op[k][0],  op[k][1],  -1, -1, r, g, b, 0.0f);
        emit_v(&v[2], op[nk][0], op[nk][1], -1, -1, r, g, b, 0.0f);
        emit_v(&v[3], ip[k][0],  ip[k][1],  -1, -1, r, g, b, a);
        emit_v(&v[4], op[nk][0], op[nk][1], -1, -1, r, g, b, 0.0f);
        emit_v(&v[5], ip[nk][0], ip[nk][1], -1, -1, r, g, b, a);
        v += 6;
    }
    *count += 30;
}

static inline void emit_xform_stroke_aa(MiniVkVertex *v_base, uint32_t *count, uint32_t max_v,
                                        const Mat4 *m, float x, float y, float cw, float ch, float lw,
                                        float r, float g, float b, float a) {
    int is_rot = (fabsf(m->m[1]) > 1e-4f || fabsf(m->m[4]) > 1e-4f);
    if (!is_rot || cw <= lw * 2.0f + 1.0f || ch <= lw * 2.0f + 1.0f || *count + 72 > max_v) {
        float b4[4][4] = {
            {x, y, x + cw, y + lw},
            {x, y + ch - lw, x + cw, y + ch},
            {x, y + lw, x + lw, y + ch - lw},
            {x + cw - lw, y + lw, x + cw, y + ch - lw}
        };
        for (int bi = 0; bi < 4; bi++) {
            if (*count + 6 > max_v) break;
            float p0x = b4[bi][0], p0y = b4[bi][1];
            float p1x = b4[bi][2], p1y = b4[bi][1];
            float p2x = b4[bi][2], p2y = b4[bi][3];
            float p3x = b4[bi][0], p3y = b4[bi][3];
            mat4_transform_pt(m, &p0x, &p0y);
            mat4_transform_pt(m, &p1x, &p1y);
            mat4_transform_pt(m, &p2x, &p2y);
            mat4_transform_pt(m, &p3x, &p3y);
            MiniVkVertex *v = v_base + *count;
            emit_v(&v[0], p0x, p0y, -1, -1, r, g, b, a);
            emit_v(&v[1], p1x, p1y, -1, -1, r, g, b, a);
            emit_v(&v[2], p2x, p2y, -1, -1, r, g, b, a);
            emit_v(&v[3], p0x, p0y, -1, -1, r, g, b, a);
            emit_v(&v[4], p2x, p2y, -1, -1, r, g, b, a);
            emit_v(&v[5], p3x, p3y, -1, -1, r, g, b, a);
            *count += 6;
        }
        return;
    }
    float ox0 = x - 0.5f, oy0 = y - 0.5f, ox1 = x + cw + 0.5f, oy1 = y + ch + 0.5f;
    float cx0 = x + 0.5f, cy0 = y + 0.5f, cx1 = x + cw - 0.5f, cy1 = y + ch - 0.5f;
    float icx0 = x + lw - 0.5f, icy0 = y + lw - 0.5f, icx1 = x + cw - lw + 0.5f, icy1 = y + ch - lw + 0.5f;
    float iox0 = x + lw + 0.5f, ioy0 = y + lw + 0.5f, iox1 = x + cw - lw - 0.5f, ioy1 = y + ch - lw - 0.5f;
    float op[4][2]  = {{ox0, oy0},   {ox1, oy0},   {ox1, oy1},   {ox0, oy1}};
    float cp[4][2]  = {{cx0, cy0},   {cx1, cy0},   {cx1, cy1},   {cx0, cy1}};
    float icp[4][2] = {{icx0, icy0}, {icx1, icy0}, {icx1, icy1}, {icx0, icy1}};
    float iop[4][2] = {{iox0, ioy0}, {iox1, ioy0}, {iox1, ioy1}, {iox0, ioy1}};
    for (int k = 0; k < 4; k++) {
        mat4_transform_pt(m, &op[k][0],  &op[k][1]);
        mat4_transform_pt(m, &cp[k][0],  &cp[k][1]);
        mat4_transform_pt(m, &icp[k][0], &icp[k][1]);
        mat4_transform_pt(m, &iop[k][0], &iop[k][1]);
    }
    MiniVkVertex *v = v_base + *count;
    for (int k = 0; k < 4; k++) {
        int nk = (k + 1) & 3;
        /* Core mitered border quad */
        emit_v(&v[0], cp[k][0],   cp[k][1],   -1, -1, r, g, b, a);
        emit_v(&v[1], cp[nk][0],  cp[nk][1],  -1, -1, r, g, b, a);
        emit_v(&v[2], icp[nk][0], icp[nk][1], -1, -1, r, g, b, a);
        emit_v(&v[3], cp[k][0],   cp[k][1],   -1, -1, r, g, b, a);
        emit_v(&v[4], icp[nk][0], icp[nk][1], -1, -1, r, g, b, a);
        emit_v(&v[5], icp[k][0],  icp[k][1],  -1, -1, r, g, b, a);
        v += 6;
        /* Outer 1px AA feather */
        emit_v(&v[0], cp[k][0],  cp[k][1],  -1, -1, r, g, b, a);
        emit_v(&v[1], op[k][0],  op[k][1],  -1, -1, r, g, b, 0.0f);
        emit_v(&v[2], op[nk][0], op[nk][1], -1, -1, r, g, b, 0.0f);
        emit_v(&v[3], cp[k][0],  cp[k][1],  -1, -1, r, g, b, a);
        emit_v(&v[4], op[nk][0], op[nk][1], -1, -1, r, g, b, 0.0f);
        emit_v(&v[5], cp[nk][0], cp[nk][1], -1, -1, r, g, b, a);
        v += 6;
        /* Inner 1px AA feather */
        emit_v(&v[0], icp[k][0],  icp[k][1],  -1, -1, r, g, b, a);
        emit_v(&v[1], icp[nk][0], icp[nk][1], -1, -1, r, g, b, a);
        emit_v(&v[2], iop[nk][0], iop[nk][1], -1, -1, r, g, b, 0.0f);
        emit_v(&v[3], icp[k][0],  icp[k][1],  -1, -1, r, g, b, a);
        emit_v(&v[4], iop[nk][0], iop[nk][1], -1, -1, r, g, b, 0.0f);
        emit_v(&v[5], iop[k][0],  iop[k][1],  -1, -1, r, g, b, 0.0f);
        v += 6;
    }
    *count += 72;
}

/* ------------------------------------------------------------------ */
/* Worker Thread Worker Procedure                                     */
/* ------------------------------------------------------------------ */
#ifdef _WIN32
static unsigned __stdcall mini_vk_worker_proc(void *arg) {
#else
static void *mini_vk_worker_proc(void *arg) {
#endif
    MiniVkWorker *w = (MiniVkWorker *)arg;
    MiniVkContext *vk = w->vk;

    while (1) {
#ifdef _WIN32
        WaitForSingleObject(w->start_event, INFINITE);
#else
        pthread_mutex_lock(&w->mutex);
        while (!w->work_ready && !w->should_exit)
            pthread_cond_wait(&w->start_cond, &w->mutex);
        w->work_ready = 0;
        pthread_mutex_unlock(&w->mutex);
#endif
        if (w->should_exit)
            break;

        uint32_t f_idx = w->frame_index;
        VkCommandBuffer cb = w->secondary_cb_current;

        VkCommandBufferInheritanceInfo inheritance = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
            .renderPass = w->render_pass,
            .subpass = 0,
            .framebuffer = vk->framebuffers[w->image_index]
        };
        VkCommandBufferBeginInfo begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
            .pInheritanceInfo = &inheritance
        };
        vkBeginCommandBuffer(cb, &begin_info);

        /* Set initial dynamic state */
        VkViewport viewport = {
            .x = 0.0f, .y = 0.0f,
            .width = (float)w->viewport_w,
            .height = (float)w->viewport_h,
            .minDepth = 0.0f, .maxDepth = 1.0f
        };
        vkCmdSetViewport(cb, 0, 1, &viewport);

        const struct MiniRenderer *r = w->r;
        uint32_t start = w->cmd_start;
        uint32_t count = w->cmd_count;

        VkRect2D cur_scissor = {
            .offset = {0, 0},
            .extent = {(uint32_t)w->viewport_w, (uint32_t)w->viewport_h}
        };
        if (count > 0 && vk->cmd_scissors) {
            cur_scissor = vk->cmd_scissors[start];
        }
        vkCmdSetScissor(cb, 0, 1, &cur_scissor);

        /* Default bindings */
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, vk->pipeline_alpha);
        VkDeviceSize v_offset = 0;
        vkCmdBindVertexBuffers(cb, 0, 1, &vk->vertex_buffer[f_idx], &v_offset);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, vk->pipeline_layout, 0, 1, &vk->atlas_desc_set, 0, NULL);

        MiniVkPushConstants pc;
        mat4_ortho(pc.projection, (float)w->viewport_w, (float)w->viewport_h);
        mat4_identity(pc.model);
        vkCmdPushConstants(cb, vk->pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

        /* Tessellation and recording loop */
        MiniVkVertex *v_base = w->vertex_out;
        uint32_t local_v_count = 0;
        uint32_t batch_start_v = 0;
        uint32_t global_base = w->vertex_offset_in_global_buffer;

        for (uint32_t i = 0; i < count; i++) {
            uint32_t cmd_idx = start + i;
            const MiniCmd *c = &r->vbuf.cmds[cmd_idx];

            if (local_v_count + 128 > w->max_vertices)
                break; /* safety limit */

            /* Dynamic Scissor Check */
            if (vk->cmd_scissors) {
                VkRect2D req_scissor = vk->cmd_scissors[cmd_idx];
                if (req_scissor.offset.x != cur_scissor.offset.x ||
                    req_scissor.offset.y != cur_scissor.offset.y ||
                    req_scissor.extent.width != cur_scissor.extent.width ||
                    req_scissor.extent.height != cur_scissor.extent.height) {
                    if (local_v_count > batch_start_v) {
                        vkCmdDraw(cb, local_v_count - batch_start_v, 1, global_base + batch_start_v, 0);
                        batch_start_v = local_v_count;
                    }
                    cur_scissor = req_scissor;
                    vkCmdSetScissor(cb, 0, 1, &cur_scissor);
                }
            }

            int has_xform = vk->cmd_has_xform ? vk->cmd_has_xform[cmd_idx] : 0;
            const Mat4 *m = vk->cmd_matrices ? &vk->cmd_matrices[cmd_idx] : NULL;

            switch (c->type) {
            case MINI_CMD_CLEAR:
                /* Clears are handled in main render pass begin */
                break;

            case MINI_CMD_RECT: {
                float x0 = c->x, y0 = c->y;
                float x1 = c->x + c->w, y1 = c->y + c->h;
                if (has_xform && m) {
                    emit_xform_quad_aa(v_base, &local_v_count, w->max_vertices, m, x0, y0, x1, y1, c->r, c->g, c->b, c->a);
                } else {
                    emit_quad(v_base, &local_v_count,
                              x0, y0, -1.0f, -1.0f,
                              x1, y1, -1.0f, -1.0f,
                              c->r, c->g, c->b, c->a);
                }
                break;
            }


            case MINI_CMD_TEXTURE_RECT: {
                float x0 = c->x, y0 = c->y;
                float x1 = c->x + c->w, y1 = c->y + c->h;
                if (has_xform && m) {
                    float p0x = x0, p0y = y0;
                    float p1x = x1, p1y = y0;
                    float p2x = x1, p2y = y1;
                    float p3x = x0, p3y = y1;
                    mat4_transform_pt(m, &p0x, &p0y);
                    mat4_transform_pt(m, &p1x, &p1y);
                    mat4_transform_pt(m, &p2x, &p2y);
                    mat4_transform_pt(m, &p3x, &p3y);
                    MiniVkVertex *v = v_base + local_v_count;
                    emit_v(&v[0], p0x, p0y, c->u0, c->v0, 1.0f, 1.0f, 1.0f, c->a);
                    emit_v(&v[1], p1x, p1y, c->u1, c->v0, 1.0f, 1.0f, 1.0f, c->a);
                    emit_v(&v[2], p2x, p2y, c->u1, c->v1, 1.0f, 1.0f, 1.0f, c->a);
                    emit_v(&v[3], p0x, p0y, c->u0, c->v0, 1.0f, 1.0f, 1.0f, c->a);
                    emit_v(&v[4], p2x, p2y, c->u1, c->v1, 1.0f, 1.0f, 1.0f, c->a);
                    emit_v(&v[5], p3x, p3y, c->u0, c->v1, 1.0f, 1.0f, 1.0f, c->a);
                    local_v_count += 6;
                } else {
                    emit_quad(v_base, &local_v_count,
                              x0, y0, c->u0, c->v0,
                              x1, y1, c->u1, c->v1,
                              1.0f, 1.0f, 1.0f, c->a);
                }
                break;
            }


            case MINI_CMD_GLYPH_RUN: {
                int n = c->glyph_count, g_start = c->reserved;
                if (g_start >= 0 && g_start + n <= r->glyph_n && r->glyph_quads) {
                    const MiniGlyphQuad *q = r->glyph_quads + g_start;
                    for (int gi = 0; gi < n; gi++) {
                        if (local_v_count + 6 > w->max_vertices) break;
                        float gx = floorf(c->x + q[gi].dx + 0.5f);
                        float gy = floorf(c->y + q[gi].dy + 0.5f);
                        float gw = q[gi].gw;
                        float gh = q[gi].gh;
                        if (has_xform && m) {
                            float p0x = gx,      p0y = gy;
                            float p1x = gx + gw, p1y = gy;
                            float p2x = gx + gw, p2y = gy + gh;
                            float p3x = gx,      p3y = gy + gh;
                            mat4_transform_pt(m, &p0x, &p0y);
                            mat4_transform_pt(m, &p1x, &p1y);
                            mat4_transform_pt(m, &p2x, &p2y);
                            mat4_transform_pt(m, &p3x, &p3y);
                            MiniVkVertex *v = v_base + local_v_count;
                            emit_v(&v[0], p0x, p0y, q[gi].u0, q[gi].v0, c->r, c->g, c->b, c->a);
                            emit_v(&v[1], p1x, p1y, q[gi].u1, q[gi].v0, c->r, c->g, c->b, c->a);
                            emit_v(&v[2], p2x, p2y, q[gi].u1, q[gi].v1, c->r, c->g, c->b, c->a);
                            emit_v(&v[3], p0x, p0y, q[gi].u0, q[gi].v0, c->r, c->g, c->b, c->a);
                            emit_v(&v[4], p2x, p2y, q[gi].u1, q[gi].v1, c->r, c->g, c->b, c->a);
                            emit_v(&v[5], p3x, p3y, q[gi].u0, q[gi].v1, c->r, c->g, c->b, c->a);
                            local_v_count += 6;
                        } else {
                            emit_quad(v_base, &local_v_count,
                                      gx, gy, q[gi].u0, q[gi].v0,
                                      gx + gw, gy + gh, q[gi].u1, q[gi].v1,
                                      c->r, c->g, c->b, c->a);
                        }
                    }
                }
                break;
            }


            case MINI_CMD_RECT_STROKE: {
                float lw = (c->reserved > 0) ? (float)c->reserved : 1.0f;
                float r_tl = c->u0, r_tr = c->v0, r_br = c->u1, r_bl = c->v1;
                float max_r = (c->w < c->h ? c->w : c->h) * 0.5f;
                if (r_tr <= 0.0f && r_br <= 0.0f && r_bl <= 0.0f) {
                    r_tr = r_tl; r_br = r_tl; r_bl = r_tl;
                }
                if (r_tl > max_r) r_tl = max_r;
                if (r_tr > max_r) r_tr = max_r;
                if (r_br > max_r) r_br = max_r;
                if (r_bl > max_r) r_bl = max_r;

                if (r_tl <= 0.5f && r_tr <= 0.5f && r_br <= 0.5f && r_bl <= 0.5f) {
                    float x = c->x, y = c->y, cw = c->w, ch = c->h;
                    /* 4 outer border quads */
                    if (has_xform && m) {
                        emit_xform_stroke_aa(v_base, &local_v_count, w->max_vertices, m, x, y, cw, ch, lw, c->r, c->g, c->b, c->a);
                    } else {
                        emit_quad(v_base, &local_v_count, x, y, -1, -1, x + cw, y + lw, -1, -1, c->r, c->g, c->b, c->a);
                        emit_quad(v_base, &local_v_count, x, y + ch - lw, -1, -1, x + cw, y + ch, -1, -1, c->r, c->g, c->b, c->a);
                        emit_quad(v_base, &local_v_count, x, y + lw, -1, -1, x + lw, y + ch - lw, -1, -1, c->r, c->g, c->b, c->a);
                        emit_quad(v_base, &local_v_count, x + cw - lw, y + lw, -1, -1, x + cw, y + ch - lw, -1, -1, c->r, c->g, c->b, c->a);
                    }
                } else {
                    struct { float x, y; } pts[128];
                    int npts = 0;
                    const int segs = 16;

                    /* Top-Left Corner Arc: PI to 1.5*PI */
                    if (r_tl > 0.5f) {
                        for (int s = 0; s <= segs; s++) {
                            float ang = PI + (s * PI / (2.0f * segs));
                            pts[npts].x = c->x + r_tl + cosf(ang) * r_tl;
                            pts[npts].y = c->y + r_tl + sinf(ang) * r_tl;
                            npts++;
                        }
                    } else {
                        pts[npts].x = c->x; pts[npts].y = c->y; npts++;
                    }

                    /* Top-Right Corner Arc: 1.5*PI to 2.0*PI */
                    if (r_tr > 0.5f) {
                        for (int s = 0; s <= segs; s++) {
                            float ang = 1.5f * PI + (s * PI / (2.0f * segs));
                            pts[npts].x = c->x + c->w - r_tr + cosf(ang) * r_tr;
                            pts[npts].y = c->y + r_tr + sinf(ang) * r_tr;
                            npts++;
                        }
                    } else {
                        pts[npts].x = c->x + c->w; pts[npts].y = c->y; npts++;
                    }

                    /* Bottom-Right Corner Arc: 0 to 0.5*PI */
                    if (r_br > 0.5f) {
                        for (int s = 0; s <= segs; s++) {
                            float ang = 0.0f + (s * PI / (2.0f * segs));
                            pts[npts].x = c->x + c->w - r_br + cosf(ang) * r_br;
                            pts[npts].y = c->y + c->h - r_br + sinf(ang) * r_br;
                            npts++;
                        }
                    } else {
                        pts[npts].x = c->x + c->w; pts[npts].y = c->y + c->h; npts++;
                    }

                    /* Bottom-Left Corner Arc: 0.5*PI to PI */
                    if (r_bl > 0.5f) {
                        for (int s = 0; s <= segs; s++) {
                            float ang = 0.5f * PI + (s * PI / (2.0f * segs));
                            pts[npts].x = c->x + r_bl + cosf(ang) * r_bl;
                            pts[npts].y = c->y + c->h - r_bl + sinf(ang) * r_bl;
                            npts++;
                        }
                    } else {
                        pts[npts].x = c->x; pts[npts].y = c->y + c->h; npts++;
                    }

                    if (has_xform && m) {
                        for (int pi = 0; pi < npts; pi++) {
                            mat4_transform_pt(m, &pts[pi].x, &pts[pi].y);
                        }
                    }

                    /* Emit anti-aliased line segments around contour */
                    for (int pi = 0; pi < npts; pi++) {
                        if (local_v_count + 18 > w->max_vertices) break;
                        int next_pi = (pi + 1) % npts;
                        float p0x = pts[pi].x, p0y = pts[pi].y;
                        float p1x = pts[next_pi].x, p1y = pts[next_pi].y;
                        float dx = p1x - p0x, dy = p1y - p0y;
                        float len = sqrtf(dx * dx + dy * dy);
                        if (len > 0.001f) {
                            float unx = -dy / len;
                            float uny =  dx / len;
                            float hw = lw * 0.5f;
                            float feather = 0.6f;
                            float w_in = hw > feather ? (hw - feather * 0.5f) : (hw * 0.5f);
                            float w_out = hw + feather * 0.5f;

                            /* Core quad */
                            emit_quad_fade(v_base, &local_v_count,
                                           p0x - unx * w_in, p0y - uny * w_in, c->a,
                                           p1x - unx * w_in, p1y - uny * w_in, c->a,
                                           p1x + unx * w_in, p1y + uny * w_in, c->a,
                                           p0x + unx * w_in, p0y + uny * w_in, c->a,
                                           c->r, c->g, c->b);
                            /* Outer fringe */
                            emit_quad_fade(v_base, &local_v_count,
                                           p0x + unx * w_in, p0y + uny * w_in, c->a,
                                           p1x + unx * w_in, p1y + uny * w_in, c->a,
                                           p1x + unx * w_out, p1y + uny * w_out, 0.0f,
                                           p0x + unx * w_out, p0y + uny * w_out, 0.0f,
                                           c->r, c->g, c->b);
                            /* Inner fringe */
                            emit_quad_fade(v_base, &local_v_count,
                                           p0x - unx * w_in, p0y - uny * w_in, c->a,
                                           p0x - unx * w_out, p0y - uny * w_out, 0.0f,
                                           p1x - unx * w_out, p1y - uny * w_out, 0.0f,
                                           p1x - unx * w_in, p1y - uny * w_in, c->a,
                                           c->r, c->g, c->b);
                        }
                    }
                }
                break;
            }


            case MINI_CMD_LINE: {
                float lw = (c->u1 > 0.0f) ? c->u1 : ((c->reserved > 0) ? (float)c->reserved : 1.0f);
                float x0 = c->x, y0 = c->y;
                float x1 = c->w, y1 = c->h;
                if (has_xform && m) {
                    mat4_transform_pt(m, &x0, &y0);
                    mat4_transform_pt(m, &x1, &y1);
                }
                float dx = x1 - x0, dy = y1 - y0;
                float len = sqrtf(dx * dx + dy * dy);
                if (len > 0.0f) {
                    float unx = -dy / len;
                    float uny =  dx / len;
                    float hw = lw * 0.5f;
                    float feather = 1.0f;

                    if (hw >= feather) {
                        float w_in = hw - feather * 0.5f;
                        float w_out = hw + feather * 0.5f;

                        /* 1. Core solid quad */
                        if (local_v_count + 6 <= w->max_vertices) {
                            emit_quad_fade(v_base, &local_v_count,
                                           x0 - unx * w_in, y0 - uny * w_in, c->a,
                                           x1 - unx * w_in, y1 - uny * w_in, c->a,
                                           x1 + unx * w_in, y1 + uny * w_in, c->a,
                                           x0 + unx * w_in, y0 + uny * w_in, c->a,
                                           c->r, c->g, c->b);
                        }
                        /* 2. Side A fringe */
                        if (local_v_count + 6 <= w->max_vertices) {
                            emit_quad_fade(v_base, &local_v_count,
                                           x0 + unx * w_in, y0 + uny * w_in, c->a,
                                           x1 + unx * w_in, y1 + uny * w_in, c->a,
                                           x1 + unx * w_out, y1 + uny * w_out, 0.0f,
                                           x0 + unx * w_out, y0 + uny * w_out, 0.0f,
                                           c->r, c->g, c->b);
                        }
                        /* 3. Side B fringe */
                        if (local_v_count + 6 <= w->max_vertices) {
                            emit_quad_fade(v_base, &local_v_count,
                                           x0 - unx * w_in, y0 - uny * w_in, c->a,
                                           x0 - unx * w_out, y0 - uny * w_out, 0.0f,
                                           x1 - unx * w_out, y1 - uny * w_out, 0.0f,
                                           x1 - unx * w_in, y1 - uny * w_in, c->a,
                                           c->r, c->g, c->b);
                        }
                    } else {
                        /* Subpixel / thin line: central peak fading to 0 */
                        float core_a = c->a * (lw / (feather * 2.0f));
                        float w_out = hw + feather;
                        if (local_v_count + 12 <= w->max_vertices) {
                            emit_quad_fade(v_base, &local_v_count,
                                           x0, y0, core_a,
                                           x1, y1, core_a,
                                           x1 + unx * w_out, y1 + uny * w_out, 0.0f,
                                           x0 + unx * w_out, y0 + uny * w_out, 0.0f,
                                           c->r, c->g, c->b);
                            emit_quad_fade(v_base, &local_v_count,
                                           x0, y0, core_a,
                                           x0 - unx * w_out, y0 - uny * w_out, 0.0f,
                                           x1 - unx * w_out, y1 - uny * w_out, 0.0f,
                                           x1, y1, core_a,
                                           c->r, c->g, c->b);
                        }
                    }
                }
                break;
            }


            case MINI_CMD_TRIANGLE: {
                float p0x = c->x,  p0y = c->y;
                float p1x = c->w,  p1y = c->h;
                float p2x = c->u0, p2y = c->v0;
                if (has_xform && m) {
                    mat4_transform_pt(m, &p0x, &p0y);
                    mat4_transform_pt(m, &p1x, &p1y);
                    mat4_transform_pt(m, &p2x, &p2y);
                }
                if (local_v_count + 3 <= w->max_vertices) {
                    MiniVkVertex *v = v_base + local_v_count;
                    emit_v(&v[0], p0x, p0y, -1, -1, c->r, c->g, c->b, c->a);
                    emit_v(&v[1], p1x, p1y, -1, -1, c->r, c->g, c->b, c->a);
                    emit_v(&v[2], p2x, p2y, -1, -1, c->r, c->g, c->b, c->a);
                    local_v_count += 3;
                }
                break;
            }

            case MINI_CMD_CIRCLE: {
                float rad = c->w;
                if (rad <= 0.0f) break;
                int segs = (rad > 16.0f) ? 48 : (rad > 4.0f ? 24 : 16);
                float cx = c->x, cy = c->y;
                float r_in = rad > 0.5f ? (rad - 0.5f) : 0.0f;
                float r_out = rad + 0.5f;
                float core_a = c->a;

                for (int si = 0; si < segs; si++) {
                    if (local_v_count + 9 > w->max_vertices) break;
                    float a0 = (float)si * (2.0f * PI / (float)segs);
                    float a1 = (float)(si + 1) * (2.0f * PI / (float)segs);
                    float ca0 = cosf(a0), sa0 = sinf(a0);
                    float ca1 = cosf(a1), sa1 = sinf(a1);

                    float p0x = cx, p0y = cy;
                    float in0x = cx + ca0 * r_in,  in0y = cy + sa0 * r_in;
                    float in1x = cx + ca1 * r_in,  in1y = cy + sa1 * r_in;
                    float out0x = cx + ca0 * r_out, out0y = cy + sa0 * r_out;
                    float out1x = cx + ca1 * r_out, out1y = cy + sa1 * r_out;

                    if (has_xform && m) {
                        mat4_transform_pt(m, &p0x, &p0y);
                        mat4_transform_pt(m, &in0x, &in0y);
                        mat4_transform_pt(m, &in1x, &in1y);
                        mat4_transform_pt(m, &out0x, &out0y);
                        mat4_transform_pt(m, &out1x, &out1y);
                    }

                    if (r_in > 0.0f && local_v_count + 3 <= w->max_vertices) {
                        MiniVkVertex *v = v_base + local_v_count;
                        emit_v(&v[0], p0x, p0y, -1, -1, c->r, c->g, c->b, core_a);
                        emit_v(&v[1], in0x, in0y, -1, -1, c->r, c->g, c->b, core_a);
                        emit_v(&v[2], in1x, in1y, -1, -1, c->r, c->g, c->b, core_a);
                        local_v_count += 3;
                    }

                    if (local_v_count + 6 <= w->max_vertices) {
                        emit_quad_fade(v_base, &local_v_count,
                                       in0x, in0y, core_a,
                                       in1x, in1y, core_a,
                                       out1x, out1y, 0.0f,
                                       out0x, out0y, 0.0f,
                                       c->r, c->g, c->b);
                    }
                }
                break;
            }


            case MINI_CMD_RECT_ROUNDED: {
                float r_tl = c->u0, r_tr = c->v0, r_br = c->u1, r_bl = c->v1;
                float max_r = (c->w < c->h ? c->w : c->h) * 0.5f;
                if (r_tr <= 0.0f && r_br <= 0.0f && r_bl <= 0.0f) {
                    r_tr = r_tl; r_br = r_tl; r_bl = r_tl;
                }
                if (r_tl > max_r) r_tl = max_r;
                if (r_tr > max_r) r_tr = max_r;
                if (r_br > max_r) r_br = max_r;
                if (r_bl > max_r) r_bl = max_r;

                if (r_tl <= 0.5f && r_tr <= 0.5f && r_br <= 0.5f && r_bl <= 0.5f) {
                    float x0 = c->x, y0 = c->y;
                    float x1 = c->x + c->w, y1 = c->y + c->h;
                    if (has_xform && m) {
                        emit_xform_quad_aa(v_base, &local_v_count, w->max_vertices, m, x0, y0, x1, y1, c->r, c->g, c->b, c->a);
                    } else {
                        emit_quad(v_base, &local_v_count, x0, y0, -1, -1, x1, y1, -1, -1, c->r, c->g, c->b, c->a);
                    }
                } else {
                    float cx = c->x + c->w * 0.5f, cy = c->y + c->h * 0.5f;
                    struct { float x, y; } pts[128], norms[128];
                    int npts = 0;
                    const int segs = 16;

                    /* Top-Left Corner Arc: PI to 1.5*PI */
                    if (r_tl > 0.5f) {
                        for (int s = 0; s <= segs; s++) {
                            float ang = PI + (s * PI / (2.0f * segs));
                            float ca = cosf(ang), sa = sinf(ang);
                            pts[npts].x = c->x + r_tl + ca * r_tl;
                            pts[npts].y = c->y + r_tl + sa * r_tl;
                            norms[npts].x = ca;
                            norms[npts].y = sa;
                            npts++;
                        }
                    } else {
                        pts[npts].x = c->x; pts[npts].y = c->y;
                        norms[npts].x = -0.7071f; norms[npts].y = -0.7071f;
                        npts++;
                    }

                    /* Top-Right Corner Arc: 1.5*PI to 2.0*PI */
                    if (r_tr > 0.5f) {
                        for (int s = 0; s <= segs; s++) {
                            float ang = 1.5f * PI + (s * PI / (2.0f * segs));
                            float ca = cosf(ang), sa = sinf(ang);
                            pts[npts].x = c->x + c->w - r_tr + ca * r_tr;
                            pts[npts].y = c->y + r_tr + sa * r_tr;
                            norms[npts].x = ca;
                            norms[npts].y = sa;
                            npts++;
                        }
                    } else {
                        pts[npts].x = c->x + c->w; pts[npts].y = c->y;
                        norms[npts].x = 0.7071f; norms[npts].y = -0.7071f;
                        npts++;
                    }

                    /* Bottom-Right Corner Arc: 0 to 0.5*PI */
                    if (r_br > 0.5f) {
                        for (int s = 0; s <= segs; s++) {
                            float ang = 0.0f + (s * PI / (2.0f * segs));
                            float ca = cosf(ang), sa = sinf(ang);
                            pts[npts].x = c->x + c->w - r_br + ca * r_br;
                            pts[npts].y = c->y + c->h - r_br + sa * r_br;
                            norms[npts].x = ca;
                            norms[npts].y = sa;
                            npts++;
                        }
                    } else {
                        pts[npts].x = c->x + c->w; pts[npts].y = c->y + c->h;
                        norms[npts].x = 0.7071f; norms[npts].y = 0.7071f;
                        npts++;
                    }

                    /* Bottom-Left Corner Arc: 0.5*PI to PI */
                    if (r_bl > 0.5f) {
                        for (int s = 0; s <= segs; s++) {
                            float ang = 0.5f * PI + (s * PI / (2.0f * segs));
                            float ca = cosf(ang), sa = sinf(ang);
                            pts[npts].x = c->x + r_bl + ca * r_bl;
                            pts[npts].y = c->y + c->h - r_bl + sa * r_bl;
                            norms[npts].x = ca;
                            norms[npts].y = sa;
                            npts++;
                        }
                    } else {
                        pts[npts].x = c->x; pts[npts].y = c->y + c->h;
                        norms[npts].x = -0.7071f; norms[npts].y = 0.7071f;
                        npts++;
                    }

                    float feather = 1.0f;
                    struct { float x, y; } in_pts[128], out_pts[128];
                    for (int pi = 0; pi < npts; pi++) {
                        float nx = norms[pi].x;
                        float ny = norms[pi].y;
                        in_pts[pi].x = pts[pi].x - nx * (feather * 0.5f);
                        in_pts[pi].y = pts[pi].y - ny * (feather * 0.5f);
                        out_pts[pi].x = pts[pi].x + nx * (feather * 0.5f);
                        out_pts[pi].y = pts[pi].y + ny * (feather * 0.5f);
                    }

                    if (has_xform && m) {
                        mat4_transform_pt(m, &cx, &cy);
                        for (int pi = 0; pi < npts; pi++) {
                            mat4_transform_pt(m, &in_pts[pi].x, &in_pts[pi].y);
                            mat4_transform_pt(m, &out_pts[pi].x, &out_pts[pi].y);
                        }
                    }

                    /* Emit solid core triangle fan */
                    for (int pi = 0; pi < npts; pi++) {
                        if (local_v_count + 3 > w->max_vertices) break;
                        int next_pi = (pi + 1) % npts;
                        MiniVkVertex *v = v_base + local_v_count;
                        emit_v(&v[0], cx, cy, -1, -1, c->r, c->g, c->b, c->a);
                        emit_v(&v[1], in_pts[pi].x, in_pts[pi].y, -1, -1, c->r, c->g, c->b, c->a);
                        emit_v(&v[2], in_pts[next_pi].x, in_pts[next_pi].y, -1, -1, c->r, c->g, c->b, c->a);
                        local_v_count += 3;
                    }

                    /* Emit outer anti-aliasing feather fringe */
                    for (int pi = 0; pi < npts; pi++) {
                        if (local_v_count + 6 > w->max_vertices) break;
                        int next_pi = (pi + 1) % npts;
                        emit_quad_fade(v_base, &local_v_count,
                                       in_pts[pi].x, in_pts[pi].y, c->a,
                                       in_pts[next_pi].x, in_pts[next_pi].y, c->a,
                                       out_pts[next_pi].x, out_pts[next_pi].y, 0.0f,
                                       out_pts[pi].x, out_pts[pi].y, 0.0f,
                                       c->r, c->g, c->b);
                    }
                }
                break;
            }


            case MINI_CMD_GRADIENT: {
                int type = c->reserved;           /* 0=linear, 1=radial, 2=conic */
                float angle = (float)c->font_id;  /* degrees 0..359 */
                float rad = (float)c->texture_id; /* corner radius */
                if (rad > c->w * 0.5f) rad = c->w * 0.5f;
                if (rad > c->h * 0.5f) rad = c->h * 0.5f;

                if (rad > 0.0f) {
                    /* Rounded gradient box using triangle fan with anti-aliasing feather fringe */
                    float cx = c->x + c->w * 0.5f;
                    float cy = c->y + c->h * 0.5f;
                    float cr, cg, cb, ca;
                    calc_grad_color(cx, cy, c->x, c->y, c->w, c->h, type, angle,
                                     c->r, c->g, c->b, c->a, c->u0, c->v0, c->u1, c->v1,
                                     &cr, &cg, &cb, &ca);

                    float cx1 = c->x + rad,        cy1 = c->y + rad;
                    float cx2 = c->x + c->w - rad, cy2 = c->y + rad;
                    float cx3 = c->x + c->w - rad, cy3 = c->y + c->h - rad;
                    float cx4 = c->x + rad,        cy4 = c->y + c->h - rad;

                    int segs = 16;
                    MiniVkPoint pts[128], norms[128];
                    float pr[128], pg[128], pb[128], pa[128];
                    int npts = 0;

                    for (int i = 0; i <= segs; i++) {
                        float a = PI + (i * PI / (2.0f * segs));
                        float ca = cosf(a), sa = sinf(a);
                        float px = cx1 + ca * rad, py = cy1 + sa * rad;
                        pts[npts] = (MiniVkPoint){px, py};
                        norms[npts] = (MiniVkPoint){ca, sa};
                        calc_grad_color(px, py, c->x, c->y, c->w, c->h, type, angle,
                                        c->r, c->g, c->b, c->a, c->u0, c->v0, c->u1, c->v1,
                                        &pr[npts], &pg[npts], &pb[npts], &pa[npts]);
                        npts++;
                    }
                    for (int i = 0; i <= segs; i++) {
                        float a = 1.5f * PI + (i * PI / (2.0f * segs));
                        float ca = cosf(a), sa = sinf(a);
                        float px = cx2 + ca * rad, py = cy2 + sa * rad;
                        pts[npts] = (MiniVkPoint){px, py};
                        norms[npts] = (MiniVkPoint){ca, sa};
                        calc_grad_color(px, py, c->x, c->y, c->w, c->h, type, angle,
                                        c->r, c->g, c->b, c->a, c->u0, c->v0, c->u1, c->v1,
                                        &pr[npts], &pg[npts], &pb[npts], &pa[npts]);
                        npts++;
                    }
                    for (int i = 0; i <= segs; i++) {
                        float a = 0.0f + (i * PI / (2.0f * segs));
                        float ca = cosf(a), sa = sinf(a);
                        float px = cx3 + ca * rad, py = cy3 + sa * rad;
                        pts[npts] = (MiniVkPoint){px, py};
                        norms[npts] = (MiniVkPoint){ca, sa};
                        calc_grad_color(px, py, c->x, c->y, c->w, c->h, type, angle,
                                        c->r, c->g, c->b, c->a, c->u0, c->v0, c->u1, c->v1,
                                        &pr[npts], &pg[npts], &pb[npts], &pa[npts]);
                        npts++;
                    }
                    for (int i = 0; i <= segs; i++) {
                        float a = 0.5f * PI + (i * PI / (2.0f * segs));
                        float ca = cosf(a), sa = sinf(a);
                        float px = cx4 + ca * rad, py = cy4 + sa * rad;
                        pts[npts] = (MiniVkPoint){px, py};
                        norms[npts] = (MiniVkPoint){ca, sa};
                        calc_grad_color(px, py, c->x, c->y, c->w, c->h, type, angle,
                                        c->r, c->g, c->b, c->a, c->u0, c->v0, c->u1, c->v1,
                                        &pr[npts], &pg[npts], &pb[npts], &pa[npts]);
                        npts++;
                    }

                    float feather = 1.0f;
                    MiniVkPoint in_pts[128], out_pts[128];
                    for (int pi = 0; pi < npts; pi++) {
                        float nx = norms[pi].x;
                        float ny = norms[pi].y;
                        in_pts[pi].x = pts[pi].x - nx * (feather * 0.5f);
                        in_pts[pi].y = pts[pi].y - ny * (feather * 0.5f);
                        out_pts[pi].x = pts[pi].x + nx * (feather * 0.5f);
                        out_pts[pi].y = pts[pi].y + ny * (feather * 0.5f);
                    }

                    if (has_xform && m) {
                        mat4_transform_pt(m, &cx, &cy);
                        for (int pi = 0; pi < npts; pi++) {
                            mat4_transform_pt(m, &in_pts[pi].x, &in_pts[pi].y);
                            mat4_transform_pt(m, &out_pts[pi].x, &out_pts[pi].y);
                        }
                    }

                    /* Emit solid core triangle fan */
                    for (int pi = 0; pi < npts; pi++) {
                        if (local_v_count + 3 > w->max_vertices) break;
                        int next_pi = (pi + 1) % npts;
                        MiniVkVertex *v = v_base + local_v_count;
                        emit_v(&v[0], cx, cy, -1, -1, cr, cg, cb, ca);
                        emit_v(&v[1], in_pts[pi].x, in_pts[pi].y, -1, -1, pr[pi], pg[pi], pb[pi], pa[pi]);
                        emit_v(&v[2], in_pts[next_pi].x, in_pts[next_pi].y, -1, -1, pr[next_pi], pg[next_pi], pb[next_pi], pa[next_pi]);
                        local_v_count += 3;
                    }

                    /* Emit outer anti-aliasing feather fringe */
                    for (int pi = 0; pi < npts; pi++) {
                        if (local_v_count + 6 > w->max_vertices) break;
                        int next_pi = (pi + 1) % npts;
                        MiniVkVertex *v = v_base + local_v_count;
                        emit_v(&v[0], in_pts[pi].x, in_pts[pi].y, -1, -1, pr[pi], pg[pi], pb[pi], pa[pi]);
                        emit_v(&v[1], in_pts[next_pi].x, in_pts[next_pi].y, -1, -1, pr[next_pi], pg[next_pi], pb[next_pi], pa[next_pi]);
                        emit_v(&v[2], out_pts[next_pi].x, out_pts[next_pi].y, -1, -1, pr[next_pi], pg[next_pi], pb[next_pi], 0.0f);
                        emit_v(&v[3], in_pts[pi].x, in_pts[pi].y, -1, -1, pr[pi], pg[pi], pb[pi], pa[pi]);
                        emit_v(&v[4], out_pts[next_pi].x, out_pts[next_pi].y, -1, -1, pr[next_pi], pg[next_pi], pb[next_pi], 0.0f);
                        emit_v(&v[5], out_pts[pi].x, out_pts[pi].y, -1, -1, pr[pi], pg[pi], pb[pi], 0.0f);
                        local_v_count += 6;
                    }
                } else {
                    /* Subdivided grid for sharp rectangular gradient (4x4 cells) */
                    const int GW = 4, GH = 4;
                    for (int gy = 0; gy < GH; gy++) {
                        float y0 = c->y + (float)gy / (float)GH * c->h;
                        float y1 = c->y + (float)(gy + 1) / (float)GH * c->h;
                        for (int gx = 0; gx < GW; gx++) {
                            float x0 = c->x + (float)gx / (float)GW * c->w;
                            float x1 = c->x + (float)(gx + 1) / (float)GW * c->w;

                            float p0x = x0, p0y = y0;
                            float p1x = x1, p1y = y0;
                            float p2x = x1, p2y = y1;
                            float p3x = x0, p3y = y1;

                            float c0r, c0g, c0b, c0a;
                            float c1r, c1g, c1b, c1a;
                            float c2r, c2g, c2b, c2a;
                            float c3r, c3g, c3b, c3a;

                            calc_grad_color(x0, y0, c->x, c->y, c->w, c->h, type, angle,
                                            c->r, c->g, c->b, c->a, c->u0, c->v0, c->u1, c->v1,
                                            &c0r, &c0g, &c0b, &c0a);
                            calc_grad_color(x1, y0, c->x, c->y, c->w, c->h, type, angle,
                                            c->r, c->g, c->b, c->a, c->u0, c->v0, c->u1, c->v1,
                                            &c1r, &c1g, &c1b, &c1a);
                            calc_grad_color(x1, y1, c->x, c->y, c->w, c->h, type, angle,
                                            c->r, c->g, c->b, c->a, c->u0, c->v0, c->u1, c->v1,
                                            &c2r, &c2g, &c2b, &c2a);
                            calc_grad_color(x0, y1, c->x, c->y, c->w, c->h, type, angle,
                                            c->r, c->g, c->b, c->a, c->u0, c->v0, c->u1, c->v1,
                                            &c3r, &c3g, &c3b, &c3a);

                            if (has_xform && m) {
                                mat4_transform_pt(m, &p0x, &p0y);
                                mat4_transform_pt(m, &p1x, &p1y);
                                mat4_transform_pt(m, &p2x, &p2y);
                                mat4_transform_pt(m, &p3x, &p3y);
                            }

                            if (local_v_count + 6 > w->max_vertices) break;
                            MiniVkVertex *v = v_base + local_v_count;
                            emit_v(&v[0], p0x, p0y, -1, -1, c0r, c0g, c0b, c0a);
                            emit_v(&v[1], p1x, p1y, -1, -1, c1r, c1g, c1b, c1a);
                            emit_v(&v[2], p2x, p2y, -1, -1, c2r, c2g, c2b, c2a);
                            emit_v(&v[3], p0x, p0y, -1, -1, c0r, c0g, c0b, c0a);
                            emit_v(&v[4], p2x, p2y, -1, -1, c2r, c2g, c2b, c2a);
                            emit_v(&v[5], p3x, p3y, -1, -1, c3r, c3g, c3b, c3a);
                            local_v_count += 6;
                        }
                    }
                }
                break;
            }


            case MINI_CMD_SHADOW: {
                float rad = c->u0;
                float spread = c->v0;
                float blur = c->u1;

                if (blur <= 0.0f) {
                    /* Hard zero-blur drop shadow (W3C CSS spec: offset copy of border-box) */
                    float sx = c->x - spread;
                    float sy = c->y - spread;
                    float sw = c->w + spread * 2.0f;
                    float sh = c->h + spread * 2.0f;
                    if (sw <= 0.0f || sh <= 0.0f) break;
                    float ir = rad;
                    if (ir > sw * 0.5f) ir = sw * 0.5f;
                    if (ir > sh * 0.5f) ir = sh * 0.5f;
                    if (ir < 0.0f) ir = 0.0f;

                    if (ir <= 0.5f) {
                        float x0 = sx, y0 = sy;
                        float x1 = sx + sw, y1 = sy + sh;
                        if (has_xform && m) {
                            emit_xform_quad_aa(v_base, &local_v_count, w->max_vertices, m, x0, y0, x1, y1, c->r, c->g, c->b, c->a);
                        } else {
                            emit_quad(v_base, &local_v_count, x0, y0, -1, -1, x1, y1, -1, -1, c->r, c->g, c->b, c->a);
                        }
                    } else {
                        /* Rounded rectangle hard shadow */
                        float cx = sx + sw * 0.5f, cy = sy + sh * 0.5f;
                        struct { float x, y; } pts[128];
                        int npts = 0;
                        const int segs = 16;
                        for (int s = 0; s <= segs; s++) {
                            float ang = PI + (s * PI / (2.0f * segs));
                            pts[npts].x = sx + ir + cosf(ang) * ir;
                            pts[npts++].y = sy + ir + sinf(ang) * ir;
                        }
                        for (int s = 0; s <= segs; s++) {
                            float ang = 1.5f * PI + (s * PI / (2.0f * segs));
                            pts[npts].x = sx + sw - ir + cosf(ang) * ir;
                            pts[npts++].y = sy + ir + sinf(ang) * ir;
                        }
                        for (int s = 0; s <= segs; s++) {
                            float ang = 0.0f + (s * PI / (2.0f * segs));
                            pts[npts].x = sx + sw - ir + cosf(ang) * ir;
                            pts[npts++].y = sy + sh - ir + sinf(ang) * ir;
                        }
                        for (int s = 0; s <= segs; s++) {
                            float ang = 0.5f * PI + (s * PI / (2.0f * segs));
                            pts[npts].x = sx + ir + cosf(ang) * ir;
                            pts[npts++].y = sy + sh - ir + sinf(ang) * ir;
                        }
                        if (has_xform && m) {
                            mat4_transform_pt(m, &cx, &cy);
                            for (int i = 0; i < npts; i++) mat4_transform_pt(m, &pts[i].x, &pts[i].y);
                        }
                        for (int i = 0; i < npts; i++) {
                            if (local_v_count + 3 > w->max_vertices) break;
                            int next_i = (i + 1) % npts;
                            MiniVkVertex *v = v_base + local_v_count;
                            emit_v(&v[0], cx, cy, -1, -1, c->r, c->g, c->b, c->a);
                            emit_v(&v[1], pts[i].x, pts[i].y, -1, -1, c->r, c->g, c->b, c->a);
                            emit_v(&v[2], pts[next_i].x, pts[next_i].y, -1, -1, c->r, c->g, c->b, c->a);
                            local_v_count += 3;
                        }
                    }
                    break;
                }

                if (rad > c->w * 0.5f) rad = c->w * 0.5f;
                if (rad > c->h * 0.5f) rad = c->h * 0.5f;

                float ix = c->x - spread, iy = c->y - spread;
                float iw = c->w + spread * 2.0f, ih = c->h + spread * 2.0f;
                float ir = rad;
                if (ir > iw * 0.5f) ir = iw * 0.5f;
                if (ir > ih * 0.5f) ir = ih * 0.5f;
                if (ir < 0.0f) ir = 0.0f;

                if (iw <= 0.0f || ih <= 0.0f) break;

                float cx1 = ix + ir, cy1 = iy + ir;
                float cx2 = ix + iw - ir, cy2 = iy + ir;
                float cx3 = ix + iw - ir, cy3 = iy + ih - ir;
                float cx4 = ix + ir, cy4 = iy + ih - ir;

                int segments = 8;

                /* 1. Draw solid inner core whenever inner width and height are positive */
                if (iw > 0.0f && ih > 0.0f) {
                    float center_x = ix + iw * 0.5f;
                    float center_y = iy + ih * 0.5f;
                    float core_a = c->a;
                    MiniVkPoint pts[48];
                    int npts = 0;
                    for (int i = 0; i <= segments; i++) {
                        float a = PI + (i * PI / (2.0f * segments));
                        pts[npts++] = (MiniVkPoint){cx1 + cosf(a) * ir, cy1 + sinf(a) * ir};
                    }
                    for (int i = 0; i <= segments; i++) {
                        float a = 1.5f * PI + (i * PI / (2.0f * segments));
                        pts[npts++] = (MiniVkPoint){cx2 + cosf(a) * ir, cy2 + sinf(a) * ir};
                    }
                    for (int i = 0; i <= segments; i++) {
                        float a = 0.0f + (i * PI / (2.0f * segments));
                        pts[npts++] = (MiniVkPoint){cx3 + cosf(a) * ir, cy3 + sinf(a) * ir};
                    }
                    for (int i = 0; i <= segments; i++) {
                        float a = 0.5f * PI + (i * PI / (2.0f * segments));
                        pts[npts++] = (MiniVkPoint){cx4 + cosf(a) * ir, cy4 + sinf(a) * ir};
                    }
                    if (has_xform && m) {
                        mat4_transform_pt(m, &center_x, &center_y);
                        for (int pi = 0; pi < npts; pi++) mat4_transform_pt(m, &pts[pi].x, &pts[pi].y);
                    }
                    for (int pi = 0; pi < npts; pi++) {
                        if (local_v_count + 3 > w->max_vertices) break;
                        int next_pi = (pi + 1) % npts;
                        MiniVkVertex *v = v_base + local_v_count;
                        emit_v(&v[0], center_x, center_y, -1, -1, c->r, c->g, c->b, core_a);
                        emit_v(&v[1], pts[pi].x, pts[pi].y, -1, -1, c->r, c->g, c->b, core_a);
                        emit_v(&v[2], pts[next_pi].x, pts[next_pi].y, -1, -1, c->r, c->g, c->b, core_a);
                        local_v_count += 3;
                    }
                }

                /* 2. Edge Quads & Corner Strips for smooth Gaussian blur falloff */
                if (blur > 0.0f) {
                    const int num_rings = 6;
                    const float ring_radii[7]  = {0.0f,  0.15f, 0.35f, 0.58f, 0.80f, 0.93f, 1.0f};
                    const float ring_alphas[7] = {1.0f,  0.80f, 0.55f, 0.32f, 0.14f, 0.04f, 0.0f};

                    for (int ring = 0; ring < num_rings; ring++) {
                        float r_in = ir + blur * ring_radii[ring];
                        float r_out = ir + blur * ring_radii[ring + 1];
                        float a_in = c->a * ring_alphas[ring];
                        float a_out = c->a * ring_alphas[ring + 1];

                        /* 4 Edge Quads: Top, Right, Bottom, Left */
                        struct { float x0, y0, x1, y1, x2, y2, x3, y3; } edge_quads[4] = {
                            { cx1, cy1 - r_in, cx2, cy2 - r_in, cx2, cy2 - r_out, cx1, cy1 - r_out },
                            { cx2 + r_in, cy2, cx3 + r_in, cy3, cx3 + r_out, cy3, cx2 + r_out, cy2 },
                            { cx3, cy3 + r_in, cx4, cy4 + r_in, cx4, cy4 + r_out, cx3, cy3 + r_out },
                            { cx4 - r_in, cy4, cx1 - r_in, cy1, cx1 - r_out, cy1, cx4 - r_out, cy4 }
                        };

                        for (int eq = 0; eq < 4; eq++) {
                            float p0x = edge_quads[eq].x0, p0y = edge_quads[eq].y0;
                            float p1x = edge_quads[eq].x1, p1y = edge_quads[eq].y1;
                            float p2x = edge_quads[eq].x2, p2y = edge_quads[eq].y2;
                            float p3x = edge_quads[eq].x3, p3y = edge_quads[eq].y3;
                            if (has_xform && m) {
                                mat4_transform_pt(m, &p0x, &p0y);
                                mat4_transform_pt(m, &p1x, &p1y);
                                mat4_transform_pt(m, &p2x, &p2y);
                                mat4_transform_pt(m, &p3x, &p3y);
                            }
                            if (local_v_count + 6 > w->max_vertices) break;
                            MiniVkVertex *v = v_base + local_v_count;
                            emit_v(&v[0], p0x, p0y, -1, -1, c->r, c->g, c->b, a_in);
                            emit_v(&v[1], p1x, p1y, -1, -1, c->r, c->g, c->b, a_in);
                            emit_v(&v[2], p2x, p2y, -1, -1, c->r, c->g, c->b, a_out);
                            emit_v(&v[3], p0x, p0y, -1, -1, c->r, c->g, c->b, a_in);
                            emit_v(&v[4], p2x, p2y, -1, -1, c->r, c->g, c->b, a_out);
                            emit_v(&v[5], p3x, p3y, -1, -1, c->r, c->g, c->b, a_out);
                            local_v_count += 6;
                        }

                        /* 4 Corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left */
                        struct { float ccx, ccy, start_ang; } corners[4] = {
                            { cx1, cy1, PI },
                            { cx2, cy2, 1.5f * PI },
                            { cx3, cy3, 0.0f },
                            { cx4, cy4, 0.5f * PI }
                        };

                        for (int ci = 0; ci < 4; ci++) {
                            float ccx = corners[ci].ccx, ccy = corners[ci].ccy;
                            float start_ang = corners[ci].start_ang;
                            for (int i = 0; i < segments; i++) {
                                float a0 = start_ang + (i * PI / (2.0f * segments));
                                float a1 = start_ang + ((i + 1) * PI / (2.0f * segments));
                                float ca0 = cosf(a0), sa0 = sinf(a0);
                                float ca1 = cosf(a1), sa1 = sinf(a1);

                                float p0x = ccx + ca0 * r_in,  p0y = ccy + sa0 * r_in;
                                float p1x = ccx + ca1 * r_in,  p1y = ccy + sa1 * r_in;
                                float p2x = ccx + ca1 * r_out, p2y = ccy + sa1 * r_out;
                                float p3x = ccx + ca0 * r_out, p3y = ccy + sa0 * r_out;

                                if (has_xform && m) {
                                    mat4_transform_pt(m, &p0x, &p0y);
                                    mat4_transform_pt(m, &p1x, &p1y);
                                    mat4_transform_pt(m, &p2x, &p2y);
                                    mat4_transform_pt(m, &p3x, &p3y);
                                }
                                if (local_v_count + 6 > w->max_vertices) break;
                                MiniVkVertex *v = v_base + local_v_count;
                                emit_v(&v[0], p0x, p0y, -1, -1, c->r, c->g, c->b, a_in);
                                emit_v(&v[1], p1x, p1y, -1, -1, c->r, c->g, c->b, a_in);
                                emit_v(&v[2], p2x, p2y, -1, -1, c->r, c->g, c->b, a_out);
                                emit_v(&v[3], p0x, p0y, -1, -1, c->r, c->g, c->b, a_in);
                                emit_v(&v[4], p2x, p2y, -1, -1, c->r, c->g, c->b, a_out);
                                emit_v(&v[5], p3x, p3y, -1, -1, c->r, c->g, c->b, a_out);
                                local_v_count += 6;
                            }
                        }
                    }
                }
                break;
            }


            case 22: { /* MINI_CMD_TEXTURE_ROUNDED */
                float r_tl = c->u0, r_tr = c->v0, r_br = c->u1, r_bl = c->v1;
                float max_r = (c->w < c->h ? c->w : c->h) * 0.5f;
                if (r_tl > max_r) r_tl = max_r; if (r_tl < 0) r_tl = 0;
                if (r_tr > max_r) r_tr = max_r; if (r_tr < 0) r_tr = 0;
                if (r_br > max_r) r_br = max_r; if (r_br < 0) r_br = 0;
                if (r_bl > max_r) r_bl = max_r; if (r_bl < 0) r_bl = 0;

                float cx = c->x + c->w * 0.5f, cy = c->y + c->h * 0.5f;
                float cx1 = c->x + r_tl, cy1 = c->y + r_tl;
                float cx2 = c->x + c->w - r_tr, cy2 = c->y + r_tr;
                float cx3 = c->x + c->w - r_br, cy3 = c->y + c->h - r_br;
                float cx4 = c->x + r_bl, cy4 = c->y + c->h - r_bl;

                int segs = 16;
                MiniVkPoint pts[128];
                int npts = 0;
                if (r_tl > 0.5f) {
                    for (int s = 0; s <= segs; s++) {
                        float a = PI + (s * PI / (2.0f * segs));
                        pts[npts++] = (MiniVkPoint){cx1 + cosf(a) * r_tl, cy1 + sinf(a) * r_tl};
                    }
                } else { pts[npts++] = (MiniVkPoint){c->x, c->y}; }
                if (r_tr > 0.5f) {
                    for (int s = 0; s <= segs; s++) {
                        float a = 1.5f * PI + (s * PI / (2.0f * segs));
                        pts[npts++] = (MiniVkPoint){cx2 + cosf(a) * r_tr, cy2 + sinf(a) * r_tr};
                    }
                } else { pts[npts++] = (MiniVkPoint){c->x + c->w, c->y}; }
                if (r_br > 0.5f) {
                    for (int s = 0; s <= segs; s++) {
                        float a = 0.0f + (s * PI / (2.0f * segs));
                        pts[npts++] = (MiniVkPoint){cx3 + cosf(a) * r_br, cy3 + sinf(a) * r_br};
                    }
                } else { pts[npts++] = (MiniVkPoint){c->x + c->w, c->y + c->h}; }
                if (r_bl > 0.5f) {
                    for (int s = 0; s <= segs; s++) {
                        float a = 0.5f * PI + (s * PI / (2.0f * segs));
                        pts[npts++] = (MiniVkPoint){cx4 + cosf(a) * r_bl, cy4 + sinf(a) * r_bl};
                    }
                } else { pts[npts++] = (MiniVkPoint){c->x, c->y + c->h}; }

                float center_x = cx, center_y = cy;
                float center_u = (cx - c->x) / c->w, center_v = (cy - c->y) / c->h;
                if (has_xform && m) {
                    mat4_transform_pt(m, &center_x, &center_y);
                    for (int pi = 0; pi < npts; pi++) mat4_transform_pt(m, &pts[pi].x, &pts[pi].y);
                }

                for (int pi = 0; pi < npts; pi++) {
                    if (local_v_count + 3 > w->max_vertices) break;
                    int next_pi = (pi + 1) % npts;
                    float u1 = (pts[pi].x - c->x) / c->w, v1 = (pts[pi].y - c->y) / c->h;
                    float u2 = (pts[next_pi].x - c->x) / c->w, v2 = (pts[next_pi].y - c->y) / c->h;
                    MiniVkVertex *v = v_base + local_v_count;
                    emit_v(&v[0], center_x, center_y, center_u, center_v, 1.0f, 1.0f, 1.0f, c->a);
                    emit_v(&v[1], pts[pi].x, pts[pi].y, u1, v1, 1.0f, 1.0f, 1.0f, c->a);
                    emit_v(&v[2], pts[next_pi].x, pts[next_pi].y, u2, v2, 1.0f, 1.0f, 1.0f, c->a);
                    local_v_count += 3;
                }
                break;
            }


                        case 20: { /* MINI_CMD_BACKDROP_FILTER (Frosted Glass / Backdrop Filter) */
                float blur = c->u0;
                float invert = c->v0;
                if (blur <= 0.0f && invert <= 0.0f)
                    break;
                float r_tl = c->u1, r_tr = c->v1, r_br = c->r, r_bl = c->g;
                float max_r = (c->w < c->h ? c->w : c->h) * 0.5f;
                if (r_tr <= 0.0f && r_br <= 0.0f && r_bl <= 0.0f) {
                    r_tr = r_tl; r_br = r_tl; r_bl = r_tl;
                }
                if (r_tl > max_r) r_tl = max_r;
                if (r_tr > max_r) r_tr = max_r;
                if (r_br > max_r) r_br = max_r;
                if (r_bl > max_r) r_bl = max_r;

                /* Frosted glass ambient color & density */
                float fr = 0.55f, fg = 0.65f, fb = 0.80f, fa = 0.06f;
                if (fabsf(invert - 0.5f) < 0.01f) {
                    fr = 0.20f; fg = 0.35f; fb = 0.65f; fa = 0.55f;
                } else if (invert > 0.0f) {
                    fr = 0.40f; fg = 0.45f; fb = 0.50f; fa = 0.60f;
                }

                if (r_tl <= 0.5f && r_tr <= 0.5f && r_br <= 0.5f && r_bl <= 0.5f) {
                    float x0 = c->x, y0 = c->y;
                    float x1 = c->x + c->w, y1 = c->y + c->h;
                    if (has_xform && m) {
                        float p0x = x0, p0y = y0;
                        float p1x = x1, p1y = y0;
                        float p2x = x1, p2y = y1;
                        float p3x = x0, p3y = y1;
                        mat4_transform_pt(m, &p0x, &p0y);
                        mat4_transform_pt(m, &p1x, &p1y);
                        mat4_transform_pt(m, &p2x, &p2y);
                        mat4_transform_pt(m, &p3x, &p3y);
                        MiniVkVertex *v = v_base + local_v_count;
                        emit_v(&v[0], p0x, p0y, -1, -1, fr, fg, fb, fa);
                        emit_v(&v[1], p1x, p1y, -1, -1, fr, fg, fb, fa);
                        emit_v(&v[2], p2x, p2y, -1, -1, fr, fg, fb, fa);
                        emit_v(&v[3], p0x, p0y, -1, -1, fr, fg, fb, fa);
                        emit_v(&v[4], p2x, p2y, -1, -1, fr, fg, fb, fa);
                        emit_v(&v[5], p3x, p3y, -1, -1, fr, fg, fb, fa);
                        local_v_count += 6;
                    } else {
                        emit_quad(v_base, &local_v_count, x0, y0, -1, -1, x1, y1, -1, -1, fr, fg, fb, fa);
                    }
                } else {
                    float cx = c->x + c->w * 0.5f, cy = c->y + c->h * 0.5f;
                    struct { float x, y; } pts[128];
                    int npts = 0;
                    const int segs = 16;

                    if (r_tl > 0.5f) {
                        for (int s = 0; s <= segs; s++) {
                            float ang = 3.14159265f + (s * 3.14159265f / (2.0f * segs));
                            pts[npts].x = c->x + r_tl + cosf(ang) * r_tl;
                            pts[npts].y = c->y + r_tl + sinf(ang) * r_tl;
                            npts++;
                        }
                    } else {
                        pts[npts].x = c->x; pts[npts].y = c->y; npts++;
                    }

                    if (r_tr > 0.5f) {
                        for (int s = 0; s <= segs; s++) {
                            float ang = 1.5f * 3.14159265f + (s * 3.14159265f / (2.0f * segs));
                            pts[npts].x = c->x + c->w - r_tr + cosf(ang) * r_tr;
                            pts[npts].y = c->y + r_tr + sinf(ang) * r_tr;
                            npts++;
                        }
                    } else {
                        pts[npts].x = c->x + c->w; pts[npts].y = c->y; npts++;
                    }

                    if (r_br > 0.5f) {
                        for (int s = 0; s <= segs; s++) {
                            float ang = 0.0f + (s * 3.14159265f / (2.0f * segs));
                            pts[npts].x = c->x + c->w - r_br + cosf(ang) * r_br;
                            pts[npts].y = c->y + c->h - r_br + sinf(ang) * r_br;
                            npts++;
                        }
                    } else {
                        pts[npts].x = c->x + c->w; pts[npts].y = c->y + c->h; npts++;
                    }

                    if (r_bl > 0.5f) {
                        for (int s = 0; s <= segs; s++) {
                            float ang = 0.5f * 3.14159265f + (s * 3.14159265f / (2.0f * segs));
                            pts[npts].x = c->x + r_bl + cosf(ang) * r_bl;
                            pts[npts].y = c->y + c->h - r_bl + sinf(ang) * r_bl;
                            npts++;
                        }
                    } else {
                        pts[npts].x = c->x; pts[npts].y = c->y + c->h; npts++;
                    }

                    if (has_xform && m) {
                        mat4_transform_pt(m, &cx, &cy);
                        for (int pi = 0; pi < npts; pi++) {
                            mat4_transform_pt(m, &pts[pi].x, &pts[pi].y);
                        }
                    }

                    for (int pi = 0; pi < npts; pi++) {
                        if (local_v_count + 3 > w->max_vertices) break;
                        int next_pi = (pi + 1) % npts;
                        MiniVkVertex *v = v_base + local_v_count;
                        emit_v(&v[0], cx, cy, -1, -1, fr, fg, fb, fa);
                        emit_v(&v[1], pts[pi].x, pts[pi].y, -1, -1, fr, fg, fb, fa);
                        emit_v(&v[2], pts[next_pi].x, pts[next_pi].y, -1, -1, fr, fg, fb, fa);
                        local_v_count += 3;
                    }
                }
                break;
            }

            case MINI_CMD_PUSH_CLIP:
            case MINI_CMD_POP_CLIP:
            case MINI_CMD_PUSH_ROUNDED_CLIP:
            case MINI_CMD_PUSH_XFORM:
            case MINI_CMD_POP_XFORM:
                /* Already pre-resolved in mini_vk_flush pre-pass into cmd_scissors and cmd_matrices */
                break;

            default:
                break;
            }
        }

        /* Final draw of any remaining vertices in this chunk */
        if (local_v_count > batch_start_v) {
            vkCmdDraw(cb, local_v_count - batch_start_v, 1, global_base + batch_start_v, 0);
        }

        w->vertices_written = local_v_count;
        vkEndCommandBuffer(cb);
#ifdef _WIN32
        SetEvent(w->done_event);
#else
        pthread_mutex_lock(&w->mutex);
        w->work_done = 1;
        pthread_cond_signal(&w->done_cond);
        pthread_mutex_unlock(&w->mutex);
#endif
    }

#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

/* ------------------------------------------------------------------ */
/* Pipeline & Render Pass Creation                                    */
/* ------------------------------------------------------------------ */
static int init_render_pass(MiniVkContext *vk) {
    /* 1. Clear Render Pass (for first flush of each frame) */
    VkAttachmentDescription color_clear = {
        .format = vk->swapchain_format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    };

    VkAttachmentReference color_ref = {
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    };

    VkSubpassDescription subpass = {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_ref
    };

    VkSubpassDependency dependency = {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
    };

    VkRenderPassCreateInfo rp_clear_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &color_clear,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency
    };

    if (vkCreateRenderPass(vk->device, &rp_clear_info, NULL, &vk->render_pass_clear) != VK_SUCCESS)
        return -1;

    /* 2. Load Render Pass (for subsequent flushes in same frame) */
    VkAttachmentDescription color_load = {
        .format = vk->swapchain_format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    };

    VkRenderPassCreateInfo rp_load_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &color_load,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency
    };

    if (vkCreateRenderPass(vk->device, &rp_load_info, NULL, &vk->render_pass_load) != VK_SUCCESS)
        return -1;

    vk->render_pass = vk->render_pass_clear;
    return 0;
}

static int init_pipeline(MiniVkContext *vk) {
    /* Shaders from mini_vk_shaders.h */
    VkShaderModuleCreateInfo vert_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = MINI_VK_VERT_SPV_SIZE,
        .pCode = MINI_VK_VERT_SPV
    };
    VkShaderModule vert_mod;
    if (vkCreateShaderModule(vk->device, &vert_info, NULL, &vert_mod) != VK_SUCCESS)
        return -1;

    VkShaderModuleCreateInfo frag_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = MINI_VK_FRAG_SPV_SIZE,
        .pCode = MINI_VK_FRAG_SPV
    };
    VkShaderModule frag_mod;
    if (vkCreateShaderModule(vk->device, &frag_info, NULL, &frag_mod) != VK_SUCCESS) {
        vkDestroyShaderModule(vk->device, vert_mod, NULL);
        return -1;
    }

    VkPipelineShaderStageCreateInfo stages[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vert_mod,
            .pName = "main"
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = frag_mod,
            .pName = "main"
        }
    };

    /* Vertex input layout */
    VkVertexInputBindingDescription binding_desc = {
        .binding = 0,
        .stride = sizeof(MiniVkVertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
    };

    VkVertexInputAttributeDescription attr_descs[3] = {
        { .location = 0, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(MiniVkVertex, x) },
        { .location = 1, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(MiniVkVertex, u) },
        { .location = 2, .binding = 0, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = offsetof(MiniVkVertex, r) }
    };

    VkPipelineVertexInputStateCreateInfo vertex_input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding_desc,
        .vertexAttributeDescriptionCount = 3,
        .pVertexAttributeDescriptions = attr_descs
    };

    VkPipelineInputAssemblyStateCreateInfo input_asm = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE
    };

    VkPipelineViewportStateCreateInfo viewport_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1
    };

    VkPipelineRasterizationStateCreateInfo rasterizer = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .lineWidth = 1.0f,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_CLOCKWISE
    };

    VkPipelineMultisampleStateCreateInfo multisampling = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .sampleShadingEnable = VK_FALSE,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT
    };

    /* Standard Premultiplied / Straight Alpha Blend */
    VkPipelineColorBlendAttachmentState blend_alpha = {
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        .blendEnable = VK_TRUE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .alphaBlendOp = VK_BLEND_OP_ADD
    };

    VkPipelineColorBlendStateCreateInfo color_blending = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .attachmentCount = 1,
        .pAttachments = &blend_alpha
    };

    VkDynamicState dynamic_states[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamic_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamic_states
    };

    /* Push constants: projection & model matrices */
    VkPushConstantRange pc_range = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(MiniVkPushConstants)
    };

    VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &vk->desc_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pc_range
    };

    if (vkCreatePipelineLayout(vk->device, &layout_info, NULL, &vk->pipeline_layout) != VK_SUCCESS) {
        vkDestroyShaderModule(vk->device, vert_mod, NULL);
        vkDestroyShaderModule(vk->device, frag_mod, NULL);
        return -1;
    }

    VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_asm,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &color_blending,
        .pDynamicState = &dynamic_state,
        .layout = vk->pipeline_layout,
        .renderPass = vk->render_pass,
        .subpass = 0
    };

    int res = vkCreateGraphicsPipelines(vk->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &vk->pipeline_alpha);

    vkDestroyShaderModule(vk->device, vert_mod, NULL);
    vkDestroyShaderModule(vk->device, frag_mod, NULL);
    return (res == VK_SUCCESS) ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* Texture Atlas Initialization & Staging Buffer                      */
/* ------------------------------------------------------------------ */
static int init_atlas_texture(MiniVkContext *vk) {
    /* 1. Create Texture Image */
    VkImageCreateInfo img_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .extent = { ATLAS_WIDTH, ATLAS_HEIGHT, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .samples = VK_SAMPLE_COUNT_1_BIT
    };

    if (vkCreateImage(vk->device, &img_info, NULL, &vk->atlas_image) != VK_SUCCESS)
        return -1;

    VkMemoryRequirements mem_reqs;
    vkGetImageMemoryRequirements(vk->device, vk->atlas_image, &mem_reqs);

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = find_memory_type(vk->physical_device, mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    };
    if (vkAllocateMemory(vk->device, &alloc_info, NULL, &vk->atlas_memory) != VK_SUCCESS)
        return -1;

    vkBindImageMemory(vk->device, vk->atlas_image, vk->atlas_memory, 0);

    /* 2. Create Image View */
    VkImageViewCreateInfo view_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = vk->atlas_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };
    if (vkCreateImageView(vk->device, &view_info, NULL, &vk->atlas_view) != VK_SUCCESS)
        return -1;

    /* 3. Create Sampler */
    VkSamplerCreateInfo sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
        .compareEnable = VK_FALSE,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR
    };
    if (vkCreateSampler(vk->device, &sampler_info, NULL, &vk->atlas_sampler) != VK_SUCCESS)
        return -1;

    /* 4. Create Host-visible Staging Buffer */
    VkDeviceSize staging_size = (VkDeviceSize)ATLAS_WIDTH * ATLAS_HEIGHT * 4;
    create_buffer(vk, staging_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  &vk->atlas_staging_buffer, &vk->atlas_staging_memory);
    vkMapMemory(vk->device, vk->atlas_staging_memory, 0, staging_size, 0, (void **)&vk->atlas_staging_mapped);
    memset(vk->atlas_staging_mapped, 0, (size_t)staging_size); /* initialize with transparent black */

    /* 5. Create Descriptor Set Layout & Descriptor Set */
    VkDescriptorSetLayoutBinding sampler_binding = {
        .binding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImmutableSamplers = NULL,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
    };
    VkDescriptorSetLayoutCreateInfo desc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &sampler_binding
    };
    vkCreateDescriptorSetLayout(vk->device, &desc_info, NULL, &vk->desc_set_layout);

    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 10
    };
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
        .maxSets = 10
    };
    vkCreateDescriptorPool(vk->device, &pool_info, NULL, &vk->desc_pool);

    VkDescriptorSetAllocateInfo alloc_desc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = vk->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &vk->desc_set_layout
    };
    vkAllocateDescriptorSets(vk->device, &alloc_desc, &vk->atlas_desc_set);

    VkDescriptorImageInfo image_desc = {
        .sampler = vk->atlas_sampler,
        .imageView = vk->atlas_view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    };
    VkWriteDescriptorSet write_desc = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = vk->atlas_desc_set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .pImageInfo = &image_desc
    };
    vkUpdateDescriptorSets(vk->device, 1, &write_desc, 0, NULL);

    /* Mark full atlas dirty for initial upload */
    vk->atlas_dirty = 1;
    vk->dirty_min_x = 0; vk->dirty_min_y = 0;
    vk->dirty_max_x = ATLAS_WIDTH; vk->dirty_max_y = ATLAS_HEIGHT;

    return 0;
}

/* ------------------------------------------------------------------ */
/* Swapchain Creation & Recreation                                    */
/* ------------------------------------------------------------------ */
static int create_swapchain(MiniVkContext *vk) {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk->physical_device, vk->surface, &caps);

    uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && image_count > caps.maxImageCount)
        image_count = caps.maxImageCount;

    VkExtent2D extent = caps.currentExtent;
    if (extent.width == 0xFFFFFFFF) {
        int fw = 0, fh = 0;
        glfwGetFramebufferSize(vk->window, &fw, &fh);
        uint32_t w = (fw > 0) ? (uint32_t)fw : (uint32_t)(vk->width > 0 ? vk->width : 1280);
        uint32_t h = (fh > 0) ? (uint32_t)fh : (uint32_t)(vk->height > 0 ? vk->height : 720);
        if (w < caps.minImageExtent.width) w = caps.minImageExtent.width;
        if (w > caps.maxImageExtent.width) w = caps.maxImageExtent.width;
        if (h < caps.minImageExtent.height) h = caps.minImageExtent.height;
        if (h > caps.maxImageExtent.height) h = caps.maxImageExtent.height;
        extent.width = w;
        extent.height = h;
    }
    vk->swapchain_extent = extent;

    /* Choose Present Mode: Mailbox for ultra-low latency without tearing, FIFO for vsync */
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    uint32_t pm_count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(vk->physical_device, vk->surface, &pm_count, NULL);
    if (pm_count > 0) {
        if (pm_count > 16) pm_count = 16;
        VkPresentModeKHR modes[16];
        vkGetPhysicalDeviceSurfacePresentModesKHR(vk->physical_device, vk->surface, &pm_count, modes);
        for (uint32_t i = 0; i < pm_count; i++) {
            if (!vk->vsync && modes[i] == VK_PRESENT_MODE_MAILBOX_KHR) {
                present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
                break;
            }
        }
    }

    VkCompositeAlphaFlagBitsKHR comp_alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (!(caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)) {
        if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR)
            comp_alpha = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
        else if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR)
            comp_alpha = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
        else if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR)
            comp_alpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    }

    VkImageUsageFlags img_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        img_usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    VkSwapchainCreateInfoKHR sci = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = vk->surface,
        .minImageCount = image_count,
        .imageFormat = vk->swapchain_format,
        .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = img_usage,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = caps.currentTransform,
        .compositeAlpha = comp_alpha,
        .presentMode = present_mode,
        .clipped = VK_TRUE
    };

    if (vkCreateSwapchainKHR(vk->device, &sci, NULL, &vk->swapchain) != VK_SUCCESS)
        return -1;

    vkGetSwapchainImagesKHR(vk->device, vk->swapchain, &vk->swapchain_image_count, NULL);
    if (vk->swapchain_image_count > 8) vk->swapchain_image_count = 8;
    vkGetSwapchainImagesKHR(vk->device, vk->swapchain, &vk->swapchain_image_count, vk->swapchain_images);

    for (uint32_t i = 0; i < vk->swapchain_image_count; i++) {
        VkImageViewCreateInfo iv_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = vk->swapchain_images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = vk->swapchain_format,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        vkCreateImageView(vk->device, &iv_info, NULL, &vk->swapchain_image_views[i]);

        VkFramebufferCreateInfo fb_info = {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = vk->render_pass,
            .attachmentCount = 1,
            .pAttachments = &vk->swapchain_image_views[i],
            .width = extent.width,
            .height = extent.height,
            .layers = 1
        };
        vkCreateFramebuffer(vk->device, &fb_info, NULL, &vk->framebuffers[i]);
    }

    return 0;
}

static void cleanup_swapchain(MiniVkContext *vk) {
    for (uint32_t i = 0; i < vk->swapchain_image_count; i++) {
        if (vk->framebuffers[i]) vkDestroyFramebuffer(vk->device, vk->framebuffers[i], NULL);
        if (vk->swapchain_image_views[i]) vkDestroyImageView(vk->device, vk->swapchain_image_views[i], NULL);
    }
    if (vk->swapchain) {
        vkDestroySwapchainKHR(vk->device, vk->swapchain, NULL);
        vk->swapchain = VK_NULL_HANDLE;
    }
}

static void recreate_swapchain(MiniVkContext *vk) {
    int w = 0, h = 0;
    glfwGetFramebufferSize(vk->window, &w, &h);
    while (w == 0 || h == 0) {
        glfwGetFramebufferSize(vk->window, &w, &h);
        glfwWaitEvents();
    }
    vkDeviceWaitIdle(vk->device);
    cleanup_swapchain(vk);
    vk->width = w;
    vk->height = h;
    create_swapchain(vk);
    vk->framebuffer_resized = 0;
}

/* ------------------------------------------------------------------ */
/* Public API Implementations                                         */
/* ------------------------------------------------------------------ */

MiniVkContext *mini_vk_create(void *glfw_window, int width, int height, int vsync, int sample_count) {
    MiniVkContext *vk = (MiniVkContext *)calloc(1, sizeof(*vk));
    if (!vk) return NULL;

    vk->window = (GLFWwindow *)glfw_window;
    vk->width = width;
    vk->height = height;
    vk->vsync = vsync;
    vk->sample_count = sample_count;

    /* 1. Create Vulkan Instance */
    if (!glfwVulkanSupported()) {
        fprintf(stderr, "[vulkan] GLFW reports Vulkan is not supported\n");
        free(vk);
        return NULL;
    }

    uint32_t ext_count = 0;
    const char **ext_names = glfwGetRequiredInstanceExtensions(&ext_count);
    if (!ext_names || ext_count == 0) {
        fprintf(stderr, "[vulkan] No required Vulkan instance extensions available\n");
        free(vk);
        return NULL;
    }

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "TinyFramework",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "MiniVulkan",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_1
    };

    VkInstanceCreateInfo inst_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = ext_count,
        .ppEnabledExtensionNames = ext_names
    };

    if (vkCreateInstance(&inst_info, NULL, &vk->instance) != VK_SUCCESS) {
        free(vk);
        return NULL;
    }

    /* 2. Create Surface */
    if (glfwCreateWindowSurface(vk->instance, vk->window, NULL, &vk->surface) != VK_SUCCESS) {
        vkDestroyInstance(vk->instance, NULL);
        free(vk);
        return NULL;
    }

    /* 3. Pick Physical Device (prefer Discrete GPU) */
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(vk->instance, &device_count, NULL);
    if (device_count == 0) {
        vkDestroySurfaceKHR(vk->instance, vk->surface, NULL);
        vkDestroyInstance(vk->instance, NULL);
        free(vk);
        return NULL;
    }

    VkPhysicalDevice *devices = (VkPhysicalDevice *)malloc(device_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(vk->instance, &device_count, devices);

    vk->physical_device = devices[0];
    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            vk->physical_device = devices[i];
            break;
        }
    }
    free(devices);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(vk->physical_device, &props);
    fprintf(stderr, "[vulkan] Multi-threaded GPU selected: %s (API: %d.%d.%d)\n",
            props.deviceName,
            VK_VERSION_MAJOR(props.apiVersion),
            VK_VERSION_MINOR(props.apiVersion),
            VK_VERSION_PATCH(props.apiVersion));
    mini_logf(MINI_LOG_INFO, "vulkan", "Multi-threaded GPU selected: %s (API: %d.%d.%d)",
              props.deviceName,
              VK_VERSION_MAJOR(props.apiVersion),
              VK_VERSION_MINOR(props.apiVersion),
              VK_VERSION_PATCH(props.apiVersion));

    /* 4. Find Queue Family */
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk->physical_device, &qf_count, NULL);
    VkQueueFamilyProperties *qf_props = (VkQueueFamilyProperties *)malloc(qf_count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(vk->physical_device, &qf_count, qf_props);

    int found_qf = 0;
    vk->graphics_queue_family = 0;
    for (uint32_t i = 0; i < qf_count; i++) {
        VkBool32 present_support = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(vk->physical_device, i, vk->surface, &present_support);
        if ((qf_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present_support) {
            vk->graphics_queue_family = i;
            found_qf = 1;
            break;
        }
    }
    free(qf_props);
    if (!found_qf) {
        fprintf(stderr, "[vulkan] No queue family supports both graphics and presentation\n");
        vkDestroySurfaceKHR(vk->instance, vk->surface, NULL);
        vkDestroyInstance(vk->instance, NULL);
        free(vk);
        return NULL;
    }

    /* 5. Create Logical Device */
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = vk->graphics_queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority
    };

    const char *device_extensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_create_info,
        .enabledExtensionCount = 1,
        .ppEnabledExtensionNames = device_extensions
    };

    if (vkCreateDevice(vk->physical_device, &device_info, NULL, &vk->device) != VK_SUCCESS) {
        vkDestroySurfaceKHR(vk->instance, vk->surface, NULL);
        vkDestroyInstance(vk->instance, NULL);
        free(vk);
        return NULL;
    }

    vkGetDeviceQueue(vk->device, vk->graphics_queue_family, 0, &vk->graphics_queue);
    vk->present_queue = vk->graphics_queue;

    /* 6. Swapchain format */
    vk->swapchain_format = VK_FORMAT_B8G8R8A8_UNORM;
    uint32_t format_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(vk->physical_device, vk->surface, &format_count, NULL);
    if (format_count > 0) {
        VkSurfaceFormatKHR formats[16];
        vkGetPhysicalDeviceSurfaceFormatsKHR(vk->physical_device, vk->surface, &format_count, formats);
        for (uint32_t i = 0; i < format_count; i++) {
            if (formats[i].format == VK_FORMAT_B8G8R8A8_UNORM || formats[i].format == VK_FORMAT_R8G8B8A8_UNORM) {
                vk->swapchain_format = formats[i].format;
                break;
            }
        }
    }

    /* 7. Render Pass, Atlas & Pipelines */
    init_render_pass(vk);
    init_atlas_texture(vk);
    init_pipeline(vk);

    /* 8. Swapchain */
    create_swapchain(vk);

    /* 9. Frame Resources (Command Pools, Buffers, Sync) */
    VkDeviceSize vb_size = (VkDeviceSize)MINI_VK_MAX_THREADS * MINI_VK_MAX_VERTICES_PER_WORKER * sizeof(MiniVkVertex);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        create_buffer(vk, vb_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      &vk->vertex_buffer[i], &vk->vertex_memory[i]);
        vkMapMemory(vk->device, vk->vertex_memory[i], 0, vb_size, 0, (void **)&vk->vertex_mapped[i]);

        VkCommandPoolCreateInfo pool_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = vk->graphics_queue_family,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        };
        vkCreateCommandPool(vk->device, &pool_info, NULL, &vk->primary_cmd_pool[i]);

        VkCommandBufferAllocateInfo cb_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = vk->primary_cmd_pool[i],
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };
        vkAllocateCommandBuffers(vk->device, &cb_info, &vk->primary_cb[i]);

        VkSemaphoreCreateInfo sem_info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        VkFenceCreateInfo fence_info = {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT
        };
        vkCreateSemaphore(vk->device, &sem_info, NULL, &vk->image_available_sem[i]);
        vkCreateSemaphore(vk->device, &sem_info, NULL, &vk->render_finished_sem[i]);
        vkCreateFence(vk->device, &fence_info, NULL, &vk->in_flight_fence[i]);
    }

    /* 10. Multi-threaded Worker Pool Initialization */
#ifdef _WIN32
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    vk->num_workers = (int)sys_info.dwNumberOfProcessors;
#else
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    vk->num_workers = (nprocs > 0) ? (int)nprocs : 2;
#endif
    if (vk->num_workers < 2) vk->num_workers = 2;
    if (vk->num_workers > MINI_VK_MAX_THREADS) vk->num_workers = MINI_VK_MAX_THREADS;

    fprintf(stderr, "[vulkan] Initializing multi-threaded worker pool with %d threads...\n", vk->num_workers);
    mini_logf(MINI_LOG_INFO, "vulkan", "Initializing multi-threaded worker pool with %d threads", vk->num_workers);

    for (int t = 0; t < vk->num_workers; t++) {
        MiniVkWorker *w = &vk->workers[t];
        w->thread_id = t;
        w->vk = vk;
        w->should_exit = 0;
#ifdef _WIN32
        w->start_event = CreateEvent(NULL, FALSE, FALSE, NULL);
        w->done_event  = CreateEvent(NULL, FALSE, FALSE, NULL);
#else
        pthread_mutex_init(&w->mutex, NULL);
        pthread_cond_init(&w->start_cond, NULL);
        pthread_cond_init(&w->done_cond, NULL);
        w->work_ready = 0;
        w->work_done = 0;
#endif
        w->max_vertices = MINI_VK_MAX_VERTICES_PER_WORKER;
        w->vertex_offset_in_global_buffer = t * MINI_VK_MAX_VERTICES_PER_WORKER;

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkCommandPoolCreateInfo pool_info = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .queueFamilyIndex = vk->graphics_queue_family,
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
            };
            vkCreateCommandPool(vk->device, &pool_info, NULL, &w->cmd_pool[i]);

            VkCommandBufferAllocateInfo cb_info = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = w->cmd_pool[i],
                .level = VK_COMMAND_BUFFER_LEVEL_SECONDARY,
                .commandBufferCount = MAX_FLUSHES_PER_FRAME
            };
            vkAllocateCommandBuffers(vk->device, &cb_info, w->secondary_cb[i]);
        }

#ifdef _WIN32
        w->thread_handle = (HANDLE)_beginthreadex(NULL, 0, mini_vk_worker_proc, w, 0, NULL);
#else
        pthread_create(&w->thread_handle, NULL, mini_vk_worker_proc, w);
#endif
    }

    return vk;
}

void mini_vk_destroy(MiniVkContext *vk) {
    if (!vk) return;
    vkDeviceWaitIdle(vk->device);

    /* Terminate worker threads */
    for (int t = 0; t < vk->num_workers; t++) {
#ifdef _WIN32
        vk->workers[t].should_exit = 1;
        SetEvent(vk->workers[t].start_event);
        WaitForSingleObject(vk->workers[t].thread_handle, 5000);
        CloseHandle(vk->workers[t].thread_handle);
        CloseHandle(vk->workers[t].start_event);
        CloseHandle(vk->workers[t].done_event);
#else
        pthread_mutex_lock(&vk->workers[t].mutex);
        vk->workers[t].should_exit = 1;
        pthread_cond_signal(&vk->workers[t].start_cond);
        pthread_mutex_unlock(&vk->workers[t].mutex);
        pthread_join(vk->workers[t].thread_handle, NULL);
        pthread_cond_destroy(&vk->workers[t].start_cond);
        pthread_cond_destroy(&vk->workers[t].done_cond);
        pthread_mutex_destroy(&vk->workers[t].mutex);
#endif

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyCommandPool(vk->device, vk->workers[t].cmd_pool[i], NULL);
        }
    }

    cleanup_swapchain(vk);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(vk->device, vk->image_available_sem[i], NULL);
        vkDestroySemaphore(vk->device, vk->render_finished_sem[i], NULL);
        vkDestroyFence(vk->device, vk->in_flight_fence[i], NULL);
        vkDestroyCommandPool(vk->device, vk->primary_cmd_pool[i], NULL);
        vkUnmapMemory(vk->device, vk->vertex_memory[i]);
        vkDestroyBuffer(vk->device, vk->vertex_buffer[i], NULL);
        vkFreeMemory(vk->device, vk->vertex_memory[i], NULL);
    }

    vkUnmapMemory(vk->device, vk->atlas_staging_memory);
    vkDestroyBuffer(vk->device, vk->atlas_staging_buffer, NULL);
    vkFreeMemory(vk->device, vk->atlas_staging_memory, NULL);

    vkDestroySampler(vk->device, vk->atlas_sampler, NULL);
    vkDestroyImageView(vk->device, vk->atlas_view, NULL);
    vkDestroyImage(vk->device, vk->atlas_image, NULL);
    vkFreeMemory(vk->device, vk->atlas_memory, NULL);

    vkDestroyDescriptorPool(vk->device, vk->desc_pool, NULL);
    vkDestroyDescriptorSetLayout(vk->device, vk->desc_set_layout, NULL);
    vkDestroyPipeline(vk->device, vk->pipeline_alpha, NULL);
    vkDestroyPipelineLayout(vk->device, vk->pipeline_layout, NULL);
    vkDestroyRenderPass(vk->device, vk->render_pass_clear, NULL);
    vkDestroyRenderPass(vk->device, vk->render_pass_load, NULL);

    vkDestroyDevice(vk->device, NULL);
    vkDestroySurfaceKHR(vk->instance, vk->surface, NULL);
    vkDestroyInstance(vk->instance, NULL);

    if (vk->cmd_matrices) free(vk->cmd_matrices);
    if (vk->cmd_scissors) free(vk->cmd_scissors);
    if (vk->cmd_has_xform) free(vk->cmd_has_xform);

    free(vk);
}

void mini_vk_begin_frame(MiniVkContext *vk, int width, int height) {
    if (!vk) return;
    if (width != vk->width || height != vk->height) {
        vk->width = width;
        vk->height = height;
        vk->framebuffer_resized = 1;
    }
}

void mini_vk_update_atlas(MiniVkContext *vk, int x, int y, int w, int h,
                          const uint32_t *src_rgba, int atlas_w, int atlas_h) {
    if (!vk || !vk->atlas_staging_mapped || !src_rgba) return;

    /* Write directly to persistently mapped host-coherent staging buffer */
    for (int row = 0; row < h; row++) {
        int dst_idx = (y + row) * ATLAS_WIDTH + x;
        int src_idx = row * atlas_w;
        if (dst_idx + w <= ATLAS_WIDTH * ATLAS_HEIGHT) {
            memcpy(vk->atlas_staging_mapped + dst_idx, src_rgba + src_idx, w * sizeof(uint32_t));
        }
    }

    /* Expand dirty bounding box */
    if (!vk->atlas_dirty) {
        vk->atlas_dirty = 1;
        vk->dirty_min_x = x;
        vk->dirty_min_y = y;
        vk->dirty_max_x = x + w;
        vk->dirty_max_y = y + h;
    } else {
        if (x < vk->dirty_min_x) vk->dirty_min_x = x;
        if (y < vk->dirty_min_y) vk->dirty_min_y = y;
        if (x + w > vk->dirty_max_x) vk->dirty_max_x = x + w;
        if (y + h > vk->dirty_max_y) vk->dirty_max_y = y + h;
    }
}

void mini_vk_flush(MiniVkContext *vk, struct MiniRenderer *r) {
    if (!vk || !r) return;
    if (r->vbuf.count == 0 && !vk->atlas_dirty) return;

    uint32_t f_idx = vk->current_frame;

    /* Start frame if not already in progress */
    if (!vk->in_frame) {
        vkWaitForFences(vk->device, 1, &vk->in_flight_fence[f_idx], VK_TRUE, UINT64_MAX);

        /* Acquire Next Swapchain Image */
        uint32_t image_index = 0;
        VkResult res = vkAcquireNextImageKHR(vk->device, vk->swapchain, UINT64_MAX,
                                             vk->image_available_sem[f_idx], VK_NULL_HANDLE, &image_index);
        if (res == VK_ERROR_OUT_OF_DATE_KHR || vk->framebuffer_resized) {
            recreate_swapchain(vk);
            return;
        }
        vk->current_image_index = image_index;
        vkResetFences(vk->device, 1, &vk->in_flight_fence[f_idx]);

        /* Reset primary command pool & command buffer */
        VkCommandBuffer primary_cb = vk->primary_cb[f_idx];
        vkResetCommandPool(vk->device, vk->primary_cmd_pool[f_idx], 0);

        for (int t = 0; t < vk->num_workers; t++) {
            vkResetCommandPool(vk->device, vk->workers[t].cmd_pool[f_idx], 0);
            vk->worker_vert_used[t] = 0;
        }

        VkCommandBufferBeginInfo begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        };
        vkBeginCommandBuffer(primary_cb, &begin_info);

        vk->in_frame = 1;
        vk->flush_count_in_frame = 0;
    }

    VkCommandBuffer primary_cb = vk->primary_cb[f_idx];
    uint32_t image_index = vk->current_image_index;

    /* Upload Atlas Dirty Region if any */
    if (vk->atlas_dirty) {
        static int s_atlas_ready = 0;
        int min_x = s_atlas_ready ? vk->dirty_min_x : 0;
        int min_y = s_atlas_ready ? vk->dirty_min_y : 0;
        int max_x = s_atlas_ready ? vk->dirty_max_x : ATLAS_WIDTH;
        int max_y = s_atlas_ready ? vk->dirty_max_y : ATLAS_HEIGHT;
        int dw = max_x - min_x;
        int dh = max_y - min_y;
        if (dw > 0 && dh > 0) {
            VkImageMemoryBarrier barrier_to_dst = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .oldLayout = s_atlas_ready ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = vk->atlas_image,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
                .srcAccessMask = s_atlas_ready ? VK_ACCESS_SHADER_READ_BIT : 0,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT
            };
            vkCmdPipelineBarrier(primary_cb,
                                 s_atlas_ready ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, NULL, 0, NULL, 1, &barrier_to_dst);

            VkBufferImageCopy copy_region = {
                .bufferOffset = (VkDeviceSize)(min_y * ATLAS_WIDTH + min_x) * 4,
                .bufferRowLength = ATLAS_WIDTH,
                .bufferImageHeight = ATLAS_HEIGHT,
                .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                .imageOffset = { min_x, min_y, 0 },
                .imageExtent = { (uint32_t)dw, (uint32_t)dh, 1 }
            };
            vkCmdCopyBufferToImage(primary_cb, vk->atlas_staging_buffer, vk->atlas_image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

            VkImageMemoryBarrier barrier_to_read = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = vk->atlas_image,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
            };
            vkCmdPipelineBarrier(primary_cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, NULL, 0, NULL, 1, &barrier_to_read);
            vk->atlas_dirty = 0;
            s_atlas_ready = 1;
        }
    }

    if (r->vbuf.count == 0) return;

    /* Use render_pass_clear on first flush of frame; render_pass_load on subsequent flushes */
    VkRenderPass cur_rp = (vk->flush_count_in_frame == 0) ? vk->render_pass_clear : vk->render_pass_load;

    /* Extract clear color from first MINI_CMD_CLEAR if present */
    VkClearValue clear_value = { .color = { { 0.12f, 0.12f, 0.12f, 1.0f } } };
    for (uint32_t i = 0; i < r->vbuf.count; i++) {
        if (r->vbuf.cmds[i].type == MINI_CMD_CLEAR) {
            clear_value.color.float32[0] = r->vbuf.cmds[i].r;
            clear_value.color.float32[1] = r->vbuf.cmds[i].g;
            clear_value.color.float32[2] = r->vbuf.cmds[i].b;
            clear_value.color.float32[3] = r->vbuf.cmds[i].a;
            break;
        }
    }

    VkRenderPassBeginInfo rp_begin = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = cur_rp,
        .framebuffer = vk->framebuffers[image_index],
        .renderArea = { .offset = {0, 0}, .extent = vk->swapchain_extent },
        .clearValueCount = 1,
        .pClearValues = &clear_value
    };

    /* Multi-threaded Parallel Command Processing */
    uint32_t total_cmds = r->vbuf.count;

    /* Pre-pass: Resolve cumulative 4x4 transform matrices and scissor clips */
    if (total_cmds > vk->cmd_states_cap) {
        uint32_t new_cap = total_cmds < 1024 ? 1024 : total_cmds * 2;
        vk->cmd_matrices = (Mat4 *)realloc(vk->cmd_matrices, new_cap * sizeof(Mat4));
        vk->cmd_scissors = (VkRect2D *)realloc(vk->cmd_scissors, new_cap * sizeof(VkRect2D));
        vk->cmd_has_xform = (uint8_t *)realloc(vk->cmd_has_xform, new_cap * sizeof(uint8_t));
        vk->cmd_states_cap = new_cap;
    }

    Mat4 mat_stack[32];
    int mat_top = 0;
    mat_stack[0] = mat4_ident();

    VkRect2D clip_stack[32];
    int clip_top = 0;
    VkRect2D full_viewport_rect = {
        .offset = {0, 0},
        .extent = {vk->swapchain_extent.width, vk->swapchain_extent.height}
    };

    for (uint32_t i = 0; i < total_cmds; i++) {
        const MiniCmd *c = &r->vbuf.cmds[i];

        if (c->type == MINI_CMD_PUSH_XFORM) {
            float tx = c->w;
            float ty = c->h;
            float tz = c->a;
            float sx = (c->u0 != 0.0f) ? c->u0 : 1.0f;
            float sy = (c->v0 != 0.0f) ? c->v0 : 1.0f;
            float skx = c->u1;
            float sky = c->v1;

            Mat4 local = mat4_ident();

            if (c->persp <= 0.0f && c->r == 0.0f && c->g == 0.0f && c->b == 0.0f &&
                sx == 1.0f && sy == 1.0f && skx == 0.0f && sky == 0.0f && tz == 0.0f) {
                local = mat4_make_translation(tx, ty, 0.0f);
            } else {
                Mat4 t1 = mat4_make_translation(c->x + tx, c->y + ty, tz);
                local = mat4_mul(local, t1);

                if (c->persp > 0.0f) {
                    Mat4 pmat = mat4_ident();
                    pmat.m[11] = -1.0f / c->persp;
                    local = mat4_mul(local, pmat);
                }
                if (c->r != 0.0f) {
                    Mat4 rx = mat4_ident();
                    float ang = c->r;
                    rx.m[5] = cosf(ang);  rx.m[9] = -sinf(ang);
                    rx.m[6] = sinf(ang);  rx.m[10] = cosf(ang);
                    local = mat4_mul(local, rx);
                }
                if (c->g != 0.0f) {
                    Mat4 ry = mat4_ident();
                    float ang = c->g;
                    ry.m[0] = cosf(ang);   ry.m[8] = sinf(ang);
                    ry.m[2] = -sinf(ang);  ry.m[10] = cosf(ang);
                    local = mat4_mul(local, ry);
                }
                if (c->b != 0.0f) {
                    Mat4 rz = mat4_make_rot_z(c->b);
                    local = mat4_mul(local, rz);
                }
                if (skx != 0.0f || sky != 0.0f) {
                    Mat4 sk = mat4_ident();
                    sk.m[4] = tanf(skx);
                    sk.m[1] = tanf(sky);
                    local = mat4_mul(local, sk);
                }
                if (sx != 1.0f || sy != 1.0f) {
                    Mat4 sc = mat4_make_scale(sx, sy, 1.0f);
                    local = mat4_mul(local, sc);
                }
                Mat4 t2 = mat4_make_translation(-c->x, -c->y, 0.0f);
                local = mat4_mul(local, t2);
            }

            if (mat_top < 31) {
                mat_stack[mat_top + 1] = mat4_mul(mat_stack[mat_top], local);
                mat_top++;
            }
        } else if (c->type == MINI_CMD_POP_XFORM) {
            if (mat_top > 0)
                mat_top--;
        } else if (c->type == MINI_CMD_PUSH_CLIP || c->type == MINI_CMD_PUSH_ROUNDED_CLIP) {
            float px[4] = { c->x, c->x + c->w, c->x + c->w, c->x };
            float py[4] = { c->y, c->y, c->y + c->h, c->y + c->h };
            float min_x = 1e9f, max_x = -1e9f;
            float min_y = 1e9f, max_y = -1e9f;
            for (int k = 0; k < 4; k++) {
                float tx = px[k], ty = py[k];
                mat4_transform_pt(&mat_stack[mat_top], &tx, &ty);
                if (tx < min_x) min_x = tx;
                if (tx > max_x) max_x = tx;
                if (ty < min_y) min_y = ty;
                if (ty > max_y) max_y = ty;
            }

            float vw = (float)vk->swapchain_extent.width;
            float vh = (float)vk->swapchain_extent.height;
            if (min_x < 0.0f) min_x = 0.0f;
            if (min_y < 0.0f) min_y = 0.0f;
            if (max_x > vw) max_x = vw;
            if (max_y > vh) max_y = vh;

            if (clip_top > 0) {
                VkRect2D p = clip_stack[clip_top - 1];
                float px0 = (float)p.offset.x;
                float py0 = (float)p.offset.y;
                float px1 = px0 + (float)p.extent.width;
                float py1 = py0 + (float)p.extent.height;
                if (px0 > min_x) min_x = px0;
                if (py0 > min_y) min_y = py0;
                if (px1 < max_x) max_x = px1;
                if (py1 < max_y) max_y = py1;
            }

            VkRect2D new_clip;
            new_clip.offset.x = (int32_t)min_x;
            new_clip.offset.y = (int32_t)min_y;
            new_clip.extent.width  = (uint32_t)(max_x > min_x ? (max_x - min_x) : 0);
            new_clip.extent.height = (uint32_t)(max_y > min_y ? (max_y - min_y) : 0);

            if (clip_top < 31) {
                clip_stack[clip_top] = new_clip;
                clip_top++;
            }
        } else if (c->type == MINI_CMD_POP_CLIP) {
            if (clip_top > 0)
                clip_top--;
        }

        vk->cmd_matrices[i] = mat_stack[mat_top];
        vk->cmd_scissors[i] = (clip_top > 0) ? clip_stack[clip_top - 1] : full_viewport_rect;
        vk->cmd_has_xform[i] = (mat_top > 0) ? 1 : 0;
    }

    int active_workers = vk->num_workers;
    if (total_cmds < (uint32_t)active_workers)
        active_workers = (total_cmds > 0) ? (int)total_cmds : 1;

    uint32_t chunk_size = (total_cmds + active_workers - 1) / active_workers;
#ifdef _WIN32
    HANDLE done_events[MINI_VK_MAX_THREADS];
#endif
    VkCommandBuffer secondary_cbs[MINI_VK_MAX_THREADS];

    uint32_t flush_idx = vk->flush_count_in_frame;
    if (flush_idx >= MAX_FLUSHES_PER_FRAME)
        flush_idx = MAX_FLUSHES_PER_FRAME - 1;

    for (int t = 0; t < active_workers; t++) {
        MiniVkWorker *w = &vk->workers[t];
        w->r = r;
        w->cmd_start = t * chunk_size;
        uint32_t rem = (w->cmd_start < total_cmds) ? (total_cmds - w->cmd_start) : 0;
        w->cmd_count = (rem < chunk_size) ? rem : chunk_size;
        w->frame_index = f_idx;
        w->image_index = image_index;
        w->viewport_w = (int)vk->swapchain_extent.width;
        w->viewport_h = (int)vk->swapchain_extent.height;
        w->render_pass = cur_rp;
        w->secondary_cb_current = w->secondary_cb[f_idx][flush_idx];

        uint32_t offset = (t * MINI_VK_MAX_VERTICES_PER_WORKER) + vk->worker_vert_used[t];
        w->vertex_offset_in_global_buffer = offset;
        w->vertex_out = vk->vertex_mapped[f_idx] + offset;
        w->max_vertices = (vk->worker_vert_used[t] < MINI_VK_MAX_VERTICES_PER_WORKER)
                          ? (MINI_VK_MAX_VERTICES_PER_WORKER - vk->worker_vert_used[t]) : 0;

        secondary_cbs[t] = w->secondary_cb_current;

#ifdef _WIN32
        done_events[t] = w->done_event;
        SetEvent(w->start_event);
#else
        pthread_mutex_lock(&w->mutex);
        w->work_done = 0;
        w->work_ready = 1;
        pthread_cond_signal(&w->start_cond);
        pthread_mutex_unlock(&w->mutex);
#endif
    }

    /* Wait for all multi-threaded workers to complete */
#ifdef _WIN32
    WaitForMultipleObjects(active_workers, done_events, TRUE, INFINITE);
#else
    for (int t = 0; t < active_workers; t++) {
        MiniVkWorker *w = &vk->workers[t];
        pthread_mutex_lock(&w->mutex);
        while (!w->work_done)
            pthread_cond_wait(&w->done_cond, &w->mutex);
        pthread_mutex_unlock(&w->mutex);
    }
#endif

    for (int t = 0; t < active_workers; t++) {
        vk->worker_vert_used[t] += vk->workers[t].vertices_written;
    }

    /* Begin Render Pass with Secondary Command Buffer support */
    vkCmdBeginRenderPass(primary_cb, &rp_begin, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);

    /* Execute recorded secondary command buffers in sequential order */
    vkCmdExecuteCommands(primary_cb, active_workers, secondary_cbs);

    vkCmdEndRenderPass(primary_cb);

    vk->flush_count_in_frame++;
}

void mini_vk_end_frame(MiniVkContext *vk) {
    if (!vk) return;
    if (!vk->in_frame) return;

    uint32_t f_idx = vk->current_frame;
    uint32_t image_index = vk->current_image_index;
    VkCommandBuffer primary_cb = vk->primary_cb[f_idx];

    if (vk->flush_count_in_frame > 0) {
        /* Transition swapchain image from COLOR_ATTACHMENT_OPTIMAL to PRESENT_SRC_KHR */
        VkImageMemoryBarrier to_present = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = vk->swapchain_images[image_index],
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = 0
        };
        vkCmdPipelineBarrier(primary_cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &to_present);

        vkEndCommandBuffer(primary_cb);

        /* Submit to Graphics Queue */
        VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        VkSubmitInfo submit_info = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &vk->image_available_sem[f_idx],
            .pWaitDstStageMask = wait_stages,
            .commandBufferCount = 1,
            .pCommandBuffers = &primary_cb,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &vk->render_finished_sem[f_idx]
        };

        vkQueueSubmit(vk->graphics_queue, 1, &submit_info, vk->in_flight_fence[f_idx]);

        VkPresentInfoKHR present_info = {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &vk->render_finished_sem[f_idx],
            .swapchainCount = 1,
            .pSwapchains = &vk->swapchain,
            .pImageIndices = &image_index
        };

        VkResult res = vkQueuePresentKHR(vk->present_queue, &present_info);
        if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR || vk->framebuffer_resized) {
            recreate_swapchain(vk);
        }

        vk->current_frame = (vk->current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    } else {
        vkEndCommandBuffer(primary_cb);
    }

    vk->in_frame = 0;
}

int mini_vk_screenshot_png(MiniVkContext *vk, uint8_t **out_png, size_t *out_len) {
    if (!vk || !out_png || !out_len) return -1;

    vkDeviceWaitIdle(vk->device);

    uint32_t w = vk->swapchain_extent.width;
    uint32_t h = vk->swapchain_extent.height;
    VkDeviceSize img_size = (VkDeviceSize)w * h * 4;

    VkBuffer staging_buf;
    VkDeviceMemory staging_mem;
    if (create_buffer(vk, img_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      &staging_buf, &staging_mem) != 0) {
        return -1;
    }

    VkCommandBufferAllocateInfo cb_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vk->primary_cmd_pool[0],
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(vk->device, &cb_info, &cb);

    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    vkBeginCommandBuffer(cb, &bi);

    VkImage current_img = vk->swapchain_images[vk->current_image_index];

    VkImageMemoryBarrier to_src = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = current_img,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, NULL, 0, NULL, 1, &to_src);

    VkBufferImageCopy copy_region = {
        .bufferOffset = 0,
        .bufferRowLength = w,
        .bufferImageHeight = h,
        .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .imageOffset = { 0, 0, 0 },
        .imageExtent = { w, h, 1 }
    };
    vkCmdCopyImageToBuffer(cb, current_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging_buf, 1, &copy_region);

    VkImageMemoryBarrier to_present = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = current_img,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, NULL, 0, NULL, 1, &to_present);

    vkEndCommandBuffer(cb);

    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb
    };
    vkQueueSubmit(vk->graphics_queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vk->graphics_queue);

    vkFreeCommandBuffers(vk->device, vk->primary_cmd_pool[0], 1, &cb);

    /* Read pixel data and flip vertically */
    uint8_t *mapped = NULL;
    vkMapMemory(vk->device, staging_mem, 0, img_size, 0, (void **)&mapped);

    uint8_t *flipped = (uint8_t *)malloc(img_size);
    if (!flipped) {
        vkUnmapMemory(vk->device, staging_mem);
        vkDestroyBuffer(vk->device, staging_buf, NULL);
        vkFreeMemory(vk->device, staging_mem, NULL);
        return -1;
    }

    /* Handle BGRA to RGBA conversion and vertical flip */
    int is_bgra = (vk->swapchain_format == VK_FORMAT_B8G8R8A8_UNORM || vk->swapchain_format == VK_FORMAT_B8G8R8A8_SRGB);
    for (uint32_t y = 0; y < h; y++) {
        const uint8_t *src_row = mapped + y * w * 4;
        uint8_t *dst_row = flipped + y * w * 4;
        for (uint32_t x = 0; x < w; x++) {
            if (is_bgra) {
                dst_row[x * 4 + 0] = src_row[x * 4 + 2];
                dst_row[x * 4 + 1] = src_row[x * 4 + 1];
                dst_row[x * 4 + 2] = src_row[x * 4 + 0];
                dst_row[x * 4 + 3] = src_row[x * 4 + 3];
            } else {
                memcpy(dst_row + x * 4, src_row + x * 4, 4);
            }
        }
    }

    vkUnmapMemory(vk->device, staging_mem);
    vkDestroyBuffer(vk->device, staging_buf, NULL);
    vkFreeMemory(vk->device, staging_mem, NULL);

    int ret = mini_png_encode_rgba(flipped, (int)w, (int)h, out_png, out_len);
    free(flipped);
    return ret;
}

void mini_vk_resize(MiniVkContext *vk, int width, int height) {
    if (!vk) return;
    vk->width = width;
    vk->height = height;
    vk->framebuffer_resized = 1;
}
