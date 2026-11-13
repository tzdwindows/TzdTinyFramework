#include "mini_audio.h"

#define MA_NO_ENCODING
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <string.h>

#define AUDIO_RING_FRAMES 65536

static ma_device g_device;
static ma_pcm_rb g_rb;
static int g_initialized = 0;
static int g_channels = 1;
static float g_last_sample[2] = {0.0f, 0.0f};

static void ma_data_callback(ma_device *pDevice, void *pOutput, const void *pInput, ma_uint32 frameCount)
{
    (void)pDevice;
    (void)pInput;

    float *out = (float *)pOutput;
    ma_uint32 ch = (ma_uint32)g_channels;
    ma_uint32 frames_read = 0;

    while (frames_read < frameCount)
    {
        ma_uint32 avail = frameCount - frames_read;
        void *pRead = NULL;
        ma_pcm_rb_acquire_read(&g_rb, &avail, &pRead);
        if (avail == 0 || !pRead)
            break;

        const float *src = (const float *)pRead;
        size_t n_floats = (size_t)avail * ch;
        float *dst = out + (size_t)frames_read * ch;
        for (size_t i = 0; i < n_floats; i++)
        {
            float s = src[i];
            if (s > 1.0f) s = 1.0f;
            else if (s < -1.0f) s = -1.0f;
            dst[i] = s;
            g_last_sample[i % ch] = s;
        }

        ma_pcm_rb_commit_read(&g_rb, avail);
        frames_read += avail;
    }

    /* Smoothly decay to zero on underrun so there is no step click */
    for (ma_uint32 f = frames_read; f < frameCount; f++)
    {
        for (ma_uint32 c = 0; c < ch; c++)
        {
            g_last_sample[c] *= 0.92f;
            if (g_last_sample[c] > -1e-5f && g_last_sample[c] < 1e-5f)
                g_last_sample[c] = 0.0f;
            out[(size_t)f * ch + c] = g_last_sample[c];
        }
    }
}

int mini_audio_init(int sample_rate, int channels)
{
    if (g_initialized)
        return 0;

    ma_uint32 sr = (sample_rate > 0) ? (ma_uint32)sample_rate : 44100;
    g_channels = (channels == 2) ? 2 : 1;
    g_last_sample[0] = 0.0f;
    g_last_sample[1] = 0.0f;

    if (ma_pcm_rb_init(ma_format_f32, (ma_uint32)g_channels, AUDIO_RING_FRAMES, NULL, NULL, &g_rb) != MA_SUCCESS)
        return -1;

    ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
    cfg.playback.format   = ma_format_f32;
    cfg.playback.channels = (ma_uint32)g_channels;
    cfg.sampleRate        = sr;
    cfg.dataCallback      = ma_data_callback;
    cfg.periodSizeInMilliseconds = 10;

    if (ma_device_init(NULL, &cfg, &g_device) != MA_SUCCESS)
    {
        ma_pcm_rb_uninit(&g_rb);
        return -1;
    }

    if (ma_device_start(&g_device) != MA_SUCCESS)
    {
        ma_device_uninit(&g_device);
        ma_pcm_rb_uninit(&g_rb);
        return -1;
    }

    g_initialized = 1;
    return 0;
}

void mini_audio_shutdown(void)
{
    if (!g_initialized)
        return;

    ma_device_uninit(&g_device);
    ma_pcm_rb_uninit(&g_rb);
    g_initialized = 0;
}

void mini_audio_queue_pcm(const float *samples, size_t count)
{
    if (!g_initialized || !samples || count == 0)
        return;

    ma_uint32 ch = (ma_uint32)g_channels;
    ma_uint32 frames_total = (ma_uint32)(count / ch);
    ma_uint32 frames_written = 0;

    while (frames_written < frames_total)
    {
        ma_uint32 space = frames_total - frames_written;
        void *pWrite = NULL;
        ma_pcm_rb_acquire_write(&g_rb, &space, &pWrite);
        if (space == 0 || !pWrite)
            break;

        const float *src = samples + (size_t)frames_written * ch;
        float *dst = (float *)pWrite;
        size_t n_floats = (size_t)space * ch;
        for (size_t i = 0; i < n_floats; i++)
        {
            float s = src[i];
            if (s > 1.0f) s = 1.0f;
            else if (s < -1.0f) s = -1.0f;
            dst[i] = s;
        }

        ma_pcm_rb_commit_write(&g_rb, space);
        frames_written += space;
    }
}
