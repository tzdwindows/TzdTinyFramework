/*
 * mini_crash.c — system-level crash interception powered by colunwind.
 *
 * Uses tzdwindows/colunwind for cross-platform crash capture (Win32 VEH/SEH
 * and POSIX sigaction + sigaltstack), minidump/crash dump generation, and
 * precise line-column symbolization, while preserving mini_crash.h APIs
 * and flushing the mini_log ring buffer on crash.
 */
#include "mini_crash.h"
#include "mini_log.h"

#include <colunwind/colunwind.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#if defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <unistd.h>
  #include <sys/stat.h>
#endif

static char g_app[64]           = "tiny_app";
static char g_dir[512]          = ".";
static char g_dir_resolved[512] = ".";
static char g_ver[64]           = "unknown";
static int  g_installed         = 0;

void mini_crash_set_version(const char *v)
{
    if (!v) return;
    snprintf(g_ver, sizeof g_ver, "%s", v);
}

static const char *platform_str(void)
{
#if defined(_WIN32)
    return "Windows";
#elif defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#elif defined(__unix__)
    return "POSIX";
#else
    return "Unknown";
#endif
}

static void stamp_now(char *out, size_t cap)
{
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    if (tm)
        snprintf(out, cap, "%04d-%02d-%02d %02d:%02d:%02d",
                 tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
                 tm->tm_hour, tm->tm_min, tm->tm_sec);
    else
        snprintf(out, cap, "?");
}

static void stamp_fname(char *out, size_t cap)
{
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    if (tm)
        snprintf(out, cap, "%04d%02d%02d-%02d%02d%02d",
                 tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
                 tm->tm_hour, tm->tm_min, tm->tm_sec);
    else
        snprintf(out, cap, "crash");
}

static void resolve_log_dir(void)
{
#if defined(_WIN32)
    if (g_dir[0] && (g_dir[0] == '/' || g_dir[0] == '\\' ||
                     (g_dir[1] == ':' && (g_dir[2] == '\\' || g_dir[2] == '/'))))
    {
        snprintf(g_dir_resolved, sizeof g_dir_resolved, "%s", g_dir);
    }
    else
    {
        char exe_dir[512] = {0};
        if (GetModuleFileNameA(NULL, exe_dir, sizeof exe_dir - 1))
        {
            char *last_slash = strrchr(exe_dir, '\\');
            if (!last_slash) last_slash = strrchr(exe_dir, '/');
            if (last_slash) *last_slash = 0;
            snprintf(g_dir_resolved, sizeof g_dir_resolved, "%s", exe_dir);
        }
        else
        {
            snprintf(g_dir_resolved, sizeof g_dir_resolved, "%s", g_dir[0] ? g_dir : ".");
        }
    }
    CreateDirectoryA(g_dir_resolved, NULL);
#else
    snprintf(g_dir_resolved, sizeof g_dir_resolved, "%s", g_dir[0] ? g_dir : ".");
    mkdir(g_dir_resolved, 0755);
#endif
}

static void write_backtrace_frames(FILE *f, const colunwind_backtrace_t *bt)
{
    if (!bt || bt->frame_count == 0)
    {
        fprintf(f, "   (no frames captured)\n");
        return;
    }
    for (uint32_t i = 0; i < bt->frame_count; i++)
    {
        const colunwind_frame_t *fr = &bt->frames[i];
        const char *mod = fr->module_name[0] ? fr->module_name : "?";
        const char *sym = (fr->function && fr->function[0]) ? fr->function :
                          (fr->symbol_name[0] ? fr->symbol_name : "?");
        const char *file = (fr->file && fr->file[0]) ? fr->file :
                           (fr->file_path[0] ? fr->file_path : NULL);
        if (file && fr->line > 0)
        {
            fprintf(f, "   #%-2u 0x%llX  %s!%s +0x%llX (%s:%u:%u)\n",
                    (unsigned)i,
                    (unsigned long long)fr->instruction_pointer,
                    mod, sym, (unsigned long long)fr->offset,
                    file, (unsigned)fr->line, (unsigned)fr->column);
        }
        else
        {
            fprintf(f, "   #%-2u 0x%llX  %s!%s +0x%llX\n",
                    (unsigned)i,
                    (unsigned long long)fr->instruction_pointer,
                    mod, sym, (unsigned long long)fr->offset);
        }
    }
}

static void write_crash_report_file(const char *kind, const char *name,
                                    uintptr_t fault_addr, uint64_t thread_id,
                                    const char *reason,
                                    const colunwind_backtrace_t *bt)
{
    char ts[64], fname[32];
    stamp_now(ts, sizeof ts);
    stamp_fname(fname, sizeof fname);

    char crash_path[1024];
    snprintf(crash_path, sizeof crash_path, "%s/%s-crash-%s.txt",
             g_dir_resolved, g_app, fname);
    char lastlog[1024];
    snprintf(lastlog, sizeof lastlog, "%s/%s-lastlog.txt",
             g_dir_resolved, g_app);

    FILE *f = fopen(crash_path, "wb");
    if (!f)
        f = stderr;

    fprintf(f, "================ %s CRASH REPORT ================\n", g_app);
    fprintf(f, " app      : %s\n", g_app);
    fprintf(f, " version  : %s\n", g_ver);
    fprintf(f, " host     : %s (%s)\n", platform_str(),
#if defined(__SIZEOF_POINTER__)
            (__SIZEOF_POINTER__ == 8 ? "64-bit" : "32-bit")
#elif defined(_WIN64)
            "64-bit"
#else
            "?"
#endif
            );
    fprintf(f, " time     : %s\n", ts);
    fprintf(f, " kind     : %s\n", kind);
    fprintf(f, " reason   : %s%s%s\n", name ? name : "UNKNOWN",
            reason ? " — " : "", reason ? reason : "");
    fprintf(f, " thread   : %llu\n", (unsigned long long)thread_id);
    fprintf(f, " fault    : 0x%llX\n", (unsigned long long)fault_addr);
    fprintf(f, " backtrace (colunwind):\n");
    write_backtrace_frames(f, bt);
    fprintf(f, "\n-- structured log ring dumped to: %s --\n", lastlog);
    fprintf(f, "==================================================\n");

    if (f != stderr)
    {
        fflush(f);
        fclose(f);
    }

    mini_log_dump_to_file(lastlog);
    mini_log_flush();
}

static void on_colunwind_crash(const colunwind_crash_context_t *ctx, void *user_data)
{
    (void)user_data;
    if (!ctx) return;

    const char *cname = colunwind_crash_type_to_string(ctx->crash_type);
    write_crash_report_file("colunwind", cname, ctx->fault_address,
                            ctx->thread_id, NULL, &ctx->backtrace);

    /* Print detailed line-column backtrace with source context to stderr */
    colunwind_print_backtrace_detailed(&ctx->backtrace, 2);
}

void mini_crash_init(const char *app_name, const char *log_dir)
{
    if (app_name && app_name[0])
        snprintf(g_app, sizeof g_app, "%s", app_name);
    if (log_dir && log_dir[0])
        snprintf(g_dir, sizeof g_dir, "%s", log_dir);
    if (g_installed)
        return;
    g_installed = 1;

    resolve_log_dir();

    colunwind_config_t cfg;
    colunwind_config_init(&cfg);
    cfg.on_crash = on_colunwind_crash;
    cfg.user_data = NULL;
    cfg.dump_directory = g_dir_resolved;
    cfg.enable_auto_dump = true;
    cfg.enable_altstack = true;
    cfg.altstack_size = 128 * 1024;

    colunwind_status_t st = colunwind_init(&cfg);
    if (st == COLUNWIND_SUCCESS)
    {
        colunwind_install_crash_handler();
    }
    else
    {
        fprintf(stderr, "[crash] colunwind_init failed: %s\n",
                colunwind_status_to_string(st));
    }
}

void mini_crash_report(const char *reason, int code)
{
    colunwind_backtrace_t trace;
    memset(&trace, 0, sizeof trace);
    if (colunwind_backtrace_capture(&trace, 1) == COLUNWIND_SUCCESS)
        colunwind_symbolize(&trace);

    write_crash_report_file("manual", "FATAL_ASSERT", 0, 0, reason, &trace);
    colunwind_write_dump(NULL, NULL,
                         COLUNWIND_DUMP_NORMAL | COLUNWIND_DUMP_WITH_UNWIND_TRACE);
    _exit(code ? code : 1);
}

__attribute__((optimize("O0"))) void mini_crash_self_test(void)
{
    volatile int *ptr = (volatile int *)0;
    *ptr = 0xDEADBEEF;
}
