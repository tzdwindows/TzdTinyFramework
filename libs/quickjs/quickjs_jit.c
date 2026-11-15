/*
 * quickjs_jit.c - Baseline JIT for QuickJS (SLJIT).
 *
 * This file is #included at the very end of quickjs.c when JS_JIT is
 * defined, so it shares quickjs.c's translation unit and has access to
 * the interpreter's private internals (JSStackFrame, JSFunctionBytecode,
 * opcode_info, js_poll_interrupts, JS_AtomGetStrRT, the js_*_rt
 * allocators, ...). It is never compiled as a standalone translation
 * unit and must not be listed in the build sources.
 *
 * Scope (honest): this is a *baseline* JIT with guard + deopt, not a
 * V8-tier optimizing JIT. Native code is emitted only for the integer
 * fast-path spine of bytecode; any failed assumption bails back to the
 * interpreter at the exact opcode boundary (sp/pc are mutated only
 * after all guards pass), so semantics are entirely the interpreter's.
 *
 * Two build switches:
 *   JS_JIT            - master switch. Enables the plumbing (registry,
 *                       trace, release, entry gate, free hook). When
 *                       off, the engine is the stock interpreter.
 *   JS_JIT_CODEGEN    - additionally enables real SLJIT native codegen
 *                       (and links sljitLir.c). Without it, compile()
 *                       only records a tracked stub entry and js_jit_run
 *                       is dormant (status never reaches COMPILED), so a
 *                       JS_JIT-only build behaves like the interpreter.
 */
#ifdef JS_JIT

#include "quickjs_jit.h"
#include <stdlib.h>
#include <stddef.h>

#ifdef JS_JIT_CODEGEN
#include "sljitLir.h"

/* ------------------------------------------------------------------ */
/* JitCtx: the single pointer argument to every generated function.  */
/* ------------------------------------------------------------------ */
/* The generated entry function takes ONE argument (struct JitCtx *),
   placed by SLJIT into SLJIT_S0 (first saved arg, see SLJIT_ARGS1(W,P)).
   All interpreter state is funneled through it. The hot registers are
   cached in saved registers at entry and written back on exit. */

struct JitCtx {
    JSValue *sp;             /*  0  current operand stack pointer       */
    uint8_t *pc;            /*  8  current program counter (resume pc) */
    JSContext *ctx;         /* 16                                       */
    JSFunctionBytecode *b;  /* 24  (for cpool / atom lookups)          */
    JSStackFrame *sf;       /* 32                                       */
    JSValue *pret_val;      /* 40  function return value slot          */
    JSValue *var_buf;       /* 48  local variable base (= sf->var_buf) */
    JSValue *arg_buf;       /* 56  argument base (= sf->arg_buf)        */
    JSValue *sp_base;       /* 64  operand stack base at JIT entry      */
    JSValue this_obj;       /* 72  this_obj at JIT entry                */
};

#define JCTX_SP      0
#define JCTX_PC      8
#define JCTX_CTX     16
#define JCTX_B       24
#define JCTX_SF      32
#define JCTX_PRET    40
#define JCTX_VARBUF  48
#define JCTX_ARGBUF  56

/* Saved-register allocation (requires SLJIT_NUMBER_OF_SAVED_REGISTERS>=5). */
#define JREG_JCTX  SLJIT_S0   /* JitCtx * (the single arg)             */
#define JREG_SP    SLJIT_S1   /* JSValue *sp                            */
#define JREG_PC    SLJIT_S2   /* uint8_t *pc (resume pc, per-opcode)   */
#define JREG_VAR   SLJIT_S3   /* JSValue *var_buf                      */
#define JREG_ARG   SLJIT_S4   /* JSValue *arg_buf                      */

/* JSValue layout (non-NaN-boxed, 16 bytes): u (8B) at +0, tag (8B) at +8. */
#define JSV_U    0
#define JSV_TAG  8
#define JSV_SIZE 16
#endif /* JS_JIT_CODEGEN */

/* ------------------------------------------------------------------ */
/* Trace                                                              */
/* ------------------------------------------------------------------ */

static int jit_trace_level = 0;
static int jit_trace_inited = 0;

static void jit_trace_seed(void)
{
    if (jit_trace_inited)
        return;
    jit_trace_inited = 1;
    const char *s = getenv("JS_JIT_TRACE");
    if (s) {
        int v = atoi(s);
        jit_trace_level = (v < 0) ? 0 : v;
    }
}

void js_jit_set_trace(int level)
{
    jit_trace_level = (level < 0) ? 0 : level;
}

static const char *jit_atom_name(JSRuntime *rt, JSAtom a,
                                 char *buf, size_t n)
{
    if (!rt)
        return "?";
    return JS_AtomGetStrRT(rt, buf, (int)n, a);
}

/* ------------------------------------------------------------------ */
/* Registry                                                           */
/* ------------------------------------------------------------------ */

static struct JSJITEntry *jit_entry_new(JSRuntime *rt, JSFunctionBytecode *b)
{
    struct JSJITState *st = rt->jit_state;
    struct JSJITEntry *e;

    if (!st)
        return NULL;
    e = js_mallocz_rt(rt, sizeof(*e));
    if (!e)
        return NULL;
    e->id = st->next_id++;
    e->bc = b;
    e->code = NULL;
    e->code_size = 0;
    e->status = JS_JIT_STATUS_STUB;   /* Phase 1: stub (no native code) */
    e->name = b->func_name;

    if (st->count == st->capacity) {
        size_t ncap = st->capacity ? st->capacity * 2 : 64;
        struct JSJITEntry **na;
        na = js_realloc_rt(rt, st->entries, ncap * sizeof(*na));
        if (!na) {
            js_free_rt(rt, e);
            st->next_id--;
            return NULL;
        }
        st->entries = na;
        st->capacity = ncap;
    }
    st->entries[st->count++] = e;
    return e;
}

/* ------------------------------------------------------------------ */
/* Compile / pre-scan                                                 */
/* ------------------------------------------------------------------ */

#ifdef JS_JIT_CODEGEN
/* Phase 2: real SLJIT codegen for an integer fast-path opcode set.
   Declared here, defined further below. Returns 1 on success and
   fills e->code / e->code_size; returns 0 if the function is not
   JIT-able (unsupported opcode present, ...). */
static int jit_codegen(JSContext *ctx, JSFunctionBytecode *b,
                       struct JSJITEntry *e);
static int jit_native_codegen(JSContext *ctx, JSFunctionBytecode *b,
                              struct JSJITEntry *e);
#endif

int js_jit_compile(JSContext *ctx, JSFunctionBytecode *b)
{
    JSRuntime *rt = ctx->rt;
    struct JSJITState *st = rt->jit_state;
    struct JSJITEntry *e;
    char nbuf[64];

    if (!st)
        return 0;
    if (b->jit_compiled)
        return (b->jit_entry != NULL);
    b->jit_compiled = 1;

    if (st->jit_disabled)
        return 0;

    /* Phase 1 gate: only ordinary (non-generator, non-async, non-ctor)
       functions. Phase 2 additionally pre-scans the opcode stream and
       refuses functions containing any opcode outside the supported set
       (which also excludes try/catch, calls, property access, ...). */
    if (b->func_kind != JS_FUNC_NORMAL)
        return 0;

    e = jit_entry_new(rt, b);
    if (!e)
        return 0;
    b->jit_entry = e;

#ifdef JS_JIT_CODEGEN
    if (jit_native_codegen(ctx, b, e)) {
        e->is_native = 1;
        e->status = JS_JIT_STATUS_COMPILED;
    } else if (jit_codegen(ctx, b, e)) {
        e->is_native = 0;
        e->status = JS_JIT_STATUS_COMPILED;
    } else {
        /* Not JIT-able: keep the stub entry (tracked, but never run). */
        e->status = JS_JIT_STATUS_STUB;
    }
#endif

    if (jit_trace_level >= 1) {
        fprintf(stderr, "[jit #%04llu] compile fn=%s len=%d status=%s\n",
                (unsigned long long)e->id,
                jit_atom_name(rt, b->func_name, nbuf, sizeof nbuf),
                b->byte_code_len,
                e->status == JS_JIT_STATUS_COMPILED ? "compiled" : "stub");
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* Run                                                                */
/* ------------------------------------------------------------------ */

int js_jit_run(JSContext *ctx, JSFunctionBytecode *b, JSStackFrame *sf,
               JSValue this_obj, JSValue **psp, uint8_t **ppc, JSValue *pret_val)
{
    struct JSJITEntry *e = b->jit_entry;

    if (!e || e->status != JS_JIT_STATUS_COMPILED) {
        /* No native code: bail to the interpreter at the current pc. */
        return JS_JIT_BAIL;
    }

#ifdef JS_JIT_CODEGEN
    if (e->is_native) {
        int argc = b->arg_count;
        sljit_sw a0 = 0, a1 = 0, a2 = 0;
        if (argc > 3)
            return JS_JIT_BAIL;
        for (int k = 0; k < argc; k++) {
            if (JS_VALUE_GET_TAG(sf->arg_buf[k]) != JS_TAG_INT)
                return JS_JIT_BAIL;
        }
        if (e->self_recursive) {
            JSValue gval = JS_GetProperty(ctx, ctx->global_obj, b->func_name);
            int same = (JS_VALUE_GET_TAG(gval) == JS_TAG_OBJECT &&
                        JS_VALUE_GET_OBJ(gval) == JS_VALUE_GET_OBJ(sf->cur_func));
            JS_FreeValue(ctx, gval);
            if (!same)
                return JS_JIT_BAIL;
        }
        if (argc > 0) a0 = (sljit_sw)JS_VALUE_GET_INT(sf->arg_buf[0]);
        if (argc > 1) a1 = (sljit_sw)JS_VALUE_GET_INT(sf->arg_buf[1]);
        if (argc > 2) a2 = (sljit_sw)JS_VALUE_GET_INT(sf->arg_buf[2]);

        typedef uint64_t (*JitNativeFn)(JSContext *, sljit_sw, sljit_sw, sljit_sw);
        JitNativeFn nfn = (JitNativeFn)(void *)e->code;
        uint64_t nres = nfn(ctx, a0, a1, a2);
        if ((nres >> 32) != 0) {
            if (jit_trace_level >= 2) {
                fprintf(stderr, "[jit #%04llu] native bail\n",
                        (unsigned long long)e->id);
            }
            return JS_JIT_BAIL;
        }
        *pret_val = JS_NewInt32(ctx, (int32_t)(uint32_t)nres);
        if (jit_trace_level >= 2) {
            fprintf(stderr, "[jit #%04llu] native done\n",
                    (unsigned long long)e->id);
        }
        return JS_JIT_DONE;
    }

    {
        /* Phase 2: invoke the generated native code, which takes a single
           JitCtx pointer (SLJIT_S0). It mutates jctx.sp/pc and writes the
           return value (if JS_JIT_DONE) to *jctx.pret_val. On any failed
           guard it leaves sp/pc at a valid opcode boundary and returns
           JS_JIT_BAIL; the interpreter then resumes from there. */
        struct JitCtx jctx;
        int (*fn)(struct JitCtx *) = (int (*)(struct JitCtx *))(void *)e->code;
        int r;
        jctx.sp = *psp;
        jctx.pc = *ppc;
        jctx.ctx = ctx;
        jctx.b = b;
        jctx.sf = sf;
        jctx.pret_val = pret_val;
        jctx.var_buf = sf->var_buf;
        jctx.arg_buf = sf->arg_buf;
        jctx.sp_base = *psp;
        jctx.this_obj = this_obj;
        r = fn(&jctx);
        *psp = jctx.sp;
        *ppc = jctx.pc;
        if (r != JS_JIT_DONE)
            r = JS_JIT_BAIL;
        if (jit_trace_level >= 2) {
            fprintf(stderr, "[jit #%04llu] %s pc=%ld\n",
                    (unsigned long long)e->id,
                    r == JS_JIT_DONE ? "done" : "bail",
                    (long)(*ppc - b->byte_code_buf));
        }
        return r;
    }
#else
    /* Phase 1 (JS_JIT only): no native code is ever produced, so this
       path is unreachable. Kept for completeness/safety. */
    (void)ctx;
    (void)sf;
    (void)psp;
    (void)ppc;
    (void)pret_val;
    return JS_JIT_BAIL;
#endif
}

/* ------------------------------------------------------------------ */
/* Release                                                            */
/* ------------------------------------------------------------------ */

typedef struct JitFieldIC {
    JSShape *shape;
    JSShape *next_shape;
    JSShape *holder_shape;
    uint32_t offset;
    uint8_t is_writable;
    uint8_t is_proto;
} JitFieldIC;

static void jit_entry_release_code(JSRuntime *rt, struct JSJITEntry *e)
{
    if (e->code) {
#ifdef JS_JIT_CODEGEN
        sljit_free_code(e->code, NULL);
#endif
        e->code = NULL;
        e->code_size = 0;
    }
    if (e->ic_data) {
        JitFieldIC *ics = (JitFieldIC *)e->ic_data;
        for (uint32_t i = 0; i < e->ic_count; i++) {
            js_free_shape_null(rt, ics[i].shape);
            js_free_shape_null(rt, ics[i].next_shape);
            js_free_shape_null(rt, ics[i].holder_shape);
        }
        js_free_rt(rt, e->ic_data);
        e->ic_data = NULL;
        e->ic_count = 0;
    }
}

void js_jit_invalidate(JSRuntime *rt, JSFunctionBytecode *b)
{
    struct JSJITEntry *e = b->jit_entry;

    if (!e)
        return;
    b->jit_entry = NULL;
    if (e->status == JS_JIT_STATUS_FREED)
        return;
    jit_entry_release_code(rt, e);
    e->status = JS_JIT_STATUS_FREED;
    e->bc = NULL;
    if (jit_trace_level >= 1) {
        fprintf(stderr, "[jit #%04llu] free\n",
                (unsigned long long)e->id);
    }
}

void js_jit_flush_all(JSRuntime *rt)
{
    struct JSJITState *st = rt->jit_state;
    size_t i;

    if (!st)
        return;
    for (i = 0; i < st->count; i++) {
        struct JSJITEntry *e = st->entries[i];
        if (!e || e->status == JS_JIT_STATUS_FREED)
            continue;
        if (e->bc) {
            e->bc->jit_entry = NULL;
            e->bc->jit_compiled = 0;  /* allow recompile on next call */
        }
        jit_entry_release_code(rt, e);
        e->status = JS_JIT_STATUS_FREED;
        e->bc = NULL;
    }
}

/* ------------------------------------------------------------------ */
/* Per-runtime state                                                  */
/* ------------------------------------------------------------------ */

void js_jit_state_init(JSRuntime *rt)
{
    struct JSJITState *st;

    jit_trace_seed();
    st = js_mallocz_rt(rt, sizeof(*st));
    rt->jit_state = st;
    if (!st)
        return;  /* OOM: JIT disabled silently */
    /* Seed runtime disable from env (also settable via JS_EnableJIT). */
    {
        const char *d = getenv("JS_JIT_DISABLE");
        if (d && (d[0] == '1' || d[0] == 't' || d[0] == 'T'))
            st->jit_disabled = 1;
    }
}

void js_jit_state_free(JSRuntime *rt)
{
    struct JSJITState *st = rt->jit_state;
    size_t i;

    if (!st)
        return;
    for (i = 0; i < st->count; i++) {
        struct JSJITEntry *e = st->entries[i];
        if (!e)
            continue;
        if (e->status != JS_JIT_STATUS_FREED) {
            jit_entry_release_code(rt, e);
            if (e->bc)
                e->bc->jit_entry = NULL;
        }
        js_free_rt(rt, e);
    }
    js_free_rt(rt, st->entries);
    js_free_rt(rt, st);
    rt->jit_state = NULL;
}

/* Public runtime toggle (declared in quickjs.h when JS_JIT is on). */
void JS_EnableJIT(JSRuntime *rt, int enable)
{
    if (!rt->jit_state)
        return;
    rt->jit_state->jit_disabled = enable ? 0 : 1;
    if (!enable) {
        /* Dropping to interpreter: release compiled code and reset the
           compiled flag so the gate no longer calls js_jit_run. */
        js_jit_flush_all(rt);
    }
}

#ifdef JS_JIT_CODEGEN
/* ================================================================== */
/* Phase 2: SLJIT native codegen                                      */
/* ================================================================== */

/*
 * Calling convention of the generated entry function:
 *
 *   int jit_entry_fn(JSContext *ctx, JSFunctionBytecode *b,
 *                    JSStackFrame *sf, JSValue **psp, uint8_t **ppc,
 *                    JSValue *pret_val);
 *
 * SLJIT register allocation:
 *   SLJIT_S0  = sp   (JSValue *stack pointer)
 *   SLJIT_S1  = pc   (uint8_t *program counter)
 *   SLJIT_S2  = ctx  (JSContext *)
 *   SLJIT_S3  = sf   (JSStackFrame *)
 *   SLJIT_S4  = b    (JSFunctionBytecode *) — for cpool/var offsets
 *   (S5.. free for scratch carry)
 *
 * Invariants mirrored from the interpreter (quickjs.c CASE bodies):
 *   JSValue = { union u (8 bytes), int64_t tag (8 bytes) } = 16 bytes.
 *   JS_TAG_INT=0, JS_TAG_BOOL=1, JS_TAG_NULL=2, JS_TAG_UNDEFINED=3.
 *   stack values grow upward; sp points at the next free slot.
 *   var_buf = sf->var_buf; arg_buf = sf->arg_buf; cpool = b->cpool.
 *
 * Deopt invariant: every opcode evaluates ALL guards BEFORE mutating
 * sp or pc, so on bail sp/pc are already at a valid opcode boundary.
 * A single shared "bail" epilogue writes S0->*psp, S1->*ppc, returns
 * JS_JIT_BAIL. A "done" epilogue pops the return value into *pret_val
 * and returns JS_JIT_DONE.
 */

/* ---- emit helpers (small, local to this TU) -------------------- */

static sljit_s32 jit_poll_helper(JSContext *ctx)
{
    /* nonzero => interrupt/exception pending => the JIT must bail so the
       interpreter re-runs the goto and throws. */
    return js_poll_interrupts(ctx) != 0 ? 1 : 0;
}

/* Slow path for the inlined interrupt poll: only hit when the counter
   reaches zero. Resets the counter and runs the handler. Returns 1 if the
   runtime wants an interrupt (=> JIT must bail). */
static sljit_s32 jit_poll_slow(JSContext *ctx)
{
    return __js_poll_interrupts(ctx) != 0 ? 1 : 0;
}

/* Record a deferred jump that must land on `target_off` (a byte offset into
   b->byte_code_buf). Resolved against per-opcode labels after the emit pass. */
struct jit_fixup {
    struct sljit_jump *jmp;
    int target_off;   /* branch target byte offset (only for kind==BRANCH) */
    int kind;         /* 0=branch, 1=bail, 2=done */
};
#define JIT_FX_BRANCH 0
#define JIT_FX_BAIL   1
#define JIT_FX_DONE   2
#define JIT_FX_EXC    3

#define JIT_RC_OFF ((sljit_sw)offsetof(JSMallocBlockHeader, ref_count) - \
                    (sljit_sw)offsetof(JSMallocBlockHeader, user_data))

/* JitFieldIC defined above jit_entry_release_code */

/* ---- Tier 2 C helpers (stay in JIT, never bail to interpreter) ---- */

static void jit_helper_free_val(JSContext *ctx, JSValue *pv)
{
    JS_FreeValue(ctx, *pv);
}

static void jit_helper_put_slot(JSContext *ctx, JSValue *dst, JSValue *src)
{
    JS_FreeValue(ctx, *dst);
    *dst = *src;
}

static sljit_s32 jit_helper_throw_uninit(struct JitCtx *jc)
{
    JS_ThrowReferenceError(jc->ctx, "cannot access uninitialized variable");
    return -1;
}

static void jit_helper_unwind(struct JitCtx *jc)
{
    while (jc->sp > jc->sp_base) {
        jc->sp--;
        JS_FreeValue(jc->ctx, *jc->sp);
    }
    *jc->pret_val = JS_EXCEPTION;
}

static void jit_helper_push_const(struct JitCtx *jc, sljit_sw idx)
{
    *jc->sp++ = JS_DupValue(jc->ctx, jc->b->cpool[idx]);
}

static sljit_s32 jit_helper_push_atom_val(struct JitCtx *jc, sljit_sw atom)
{
    JSValue v = JS_AtomToValue(jc->ctx, (JSAtom)atom);
    if (JS_IsException(v))
        return -1;
    *jc->sp++ = v;
    return 0;
}

static sljit_s32 jit_helper_get_var(struct JitCtx *jc, sljit_sw atom, sljit_sw throw_ref)
{
    jc->sf->cur_sp = jc->sp;
    JSValue v = JS_GetGlobalVar(jc->ctx, (JSAtom)atom, throw_ref != 0);
    if (JS_IsException(v))
        return -1;
    *jc->sp++ = v;
    return 0;
}

static sljit_s32 jit_helper_put_var(struct JitCtx *jc, sljit_sw atom, sljit_sw flag)
{
    jc->sf->cur_sp = jc->sp;
    JSValue v = *--jc->sp;
    return JS_SetGlobalVar(jc->ctx, (JSAtom)atom, v, (int)flag);
}

static sljit_s32 jit_helper_object(struct JitCtx *jc)
{
    jc->sf->cur_sp = jc->sp;
    JSValue v = JS_NewObject(jc->ctx);
    if (JS_IsException(v))
        return -1;
    *jc->sp++ = v;
    return 0;
}

static sljit_s32 jit_helper_array_from(struct JitCtx *jc, sljit_sw argc)
{
    JSContext *ctx = jc->ctx;
    int n = (int)argc;
    jc->sf->cur_sp = jc->sp;
    JSValue arr = JS_NewArray(ctx);
    if (JS_IsException(arr))
        return -1;
    for (int i = 0; i < n; i++) {
        if (JS_DefinePropertyValueUint32(ctx, arr, (uint32_t)i,
                                         jc->sp[i - n], JS_PROP_C_W_E) < 0) {
            for (int k = i + 1; k < n; k++)
                JS_FreeValue(ctx, jc->sp[k - n]);
            jc->sp -= n;
            JS_FreeValue(ctx, arr);
            return -1;
        }
    }
    jc->sp -= n;
    *jc->sp++ = arr;
    return 0;
}

static sljit_s32 jit_helper_define_field(struct JitCtx *jc, sljit_sw atom, JitFieldIC *ic)
{
    JSContext *ctx = jc->ctx;
    JSValue val = jc->sp[-1];
    JSValue obj = jc->sp[-2];
    if (likely(ic && JS_VALUE_GET_TAG(obj) == JS_TAG_OBJECT)) {
        JSObject *p = JS_VALUE_GET_OBJ(obj);
        JSShape *sh = p->shape;
        if (likely(sh == ic->shape && ic->next_shape != NULL)) {
            JSShape *nsh = ic->next_shape;
            if (nsh->prop_size != sh->prop_size) {
                JSProperty *np = js_realloc(ctx, p->prop, sizeof(JSProperty) * nsh->prop_size);
                if (unlikely(!np)) {
                    jc->sf->cur_sp = jc->sp;
                    return -1;
                }
                p->prop = np;
            }
            p->prop[ic->offset].u.value = val;
            p->shape = js_dup_shape(nsh);
            js_free_shape_null(ctx->rt, sh);
            jc->sp--;
            return 0;
        }
    }
    JSShape *old_sh = (ic && !ic->next_shape && JS_VALUE_GET_TAG(obj) == JS_TAG_OBJECT)
                      ? JS_VALUE_GET_OBJ(obj)->shape : NULL;
    uint32_t old_cnt = old_sh ? (uint32_t)old_sh->prop_count : 0;
    if (old_sh) js_dup_shape(old_sh);
    jc->sf->cur_sp = jc->sp;
    int ret = JS_DefinePropertyValue(ctx, obj, (JSAtom)atom, val,
                                     JS_PROP_C_W_E | JS_PROP_THROW);
    jc->sp--;
    if (ret < 0) {
        if (old_sh) js_free_shape_null(ctx->rt, old_sh);
        return -1;
    }
    if (old_sh) {
        JSObject *p = JS_VALUE_GET_OBJ(obj);
        JSShape *nsh = p->shape;
        if (nsh != old_sh && (uint32_t)nsh->prop_count == old_cnt + 1) {
            JSShapeProperty *prs = get_shape_prop(nsh);
            if (prs[old_cnt].atom == (JSAtom)atom &&
                (prs[old_cnt].flags & JS_PROP_TMASK) == JS_PROP_NORMAL) {
                ic->shape = old_sh; /* already dup'd above */
                ic->next_shape = js_dup_shape(nsh);
                ic->offset = old_cnt;
                return 0;
            }
        }
        js_free_shape_null(ctx->rt, old_sh);
    }
    return 0;
}

static sljit_s32 jit_helper_fclosure(struct JitCtx *jc, sljit_sw idx)
{
    JSObject *fn_obj = JS_VALUE_GET_OBJ(jc->sf->cur_func);
    JSVarRef **cur_var_refs = fn_obj ? fn_obj->u.func.var_refs : NULL;
    jc->sf->cur_pc = jc->pc;
    jc->sf->cur_sp = jc->sp;
    JSValue fn = js_closure(jc->ctx, JS_DupValue(jc->ctx, jc->b->cpool[idx]),
                            cur_var_refs, jc->sf);
    if (JS_IsException(fn))
        return -1;
    *jc->sp++ = fn;
    return 0;
}

static sljit_s32 jit_helper_call(struct JitCtx *jc, sljit_sw argc_w, sljit_sw kind)
{
    JSContext *ctx = jc->ctx;
    int argc = (int)argc_w;
    JSValue *sp = jc->sp;
    JSValue ret;
    jc->sf->cur_sp = sp;
    if (kind == 0) {
        /* OP_call / OP_call0..3: [func, args...] */
        JSValue func = sp[-1 - argc];
        ret = JS_Call(ctx, func, JS_UNDEFINED, argc, sp - argc);
        for (int i = 0; i < argc; i++)
            JS_FreeValue(ctx, sp[-argc + i]);
        JS_FreeValue(ctx, func);
        sp -= argc + 1;
    } else if (kind == 1) {
        /* OP_call_method: [this, func, args...] */
        JSValue this_val = sp[-2 - argc];
        JSValue func = sp[-1 - argc];
        ret = JS_Call(ctx, func, this_val, argc, sp - argc);
        for (int i = 0; i < argc; i++)
            JS_FreeValue(ctx, sp[-argc + i]);
        JS_FreeValue(ctx, func);
        JS_FreeValue(ctx, this_val);
        sp -= argc + 2;
    } else {
        /* OP_call_constructor: [func, new_target, args...] */
        JSValue func = sp[-2 - argc];
        JSValue new_target = sp[-1 - argc];
        ret = JS_CallConstructor2(ctx, func, new_target, argc, sp - argc);
        for (int i = 0; i < argc; i++)
            JS_FreeValue(ctx, sp[-argc + i]);
        JS_FreeValue(ctx, new_target);
        JS_FreeValue(ctx, func);
        sp -= argc + 2;
    }
    jc->sp = sp;
    if (JS_IsException(ret))
        return -1;
    *jc->sp++ = ret;
    return 0;
}

static sljit_s32 jit_helper_get_field_ic(struct JitCtx *jc, sljit_sw atom_w,
                                         JitFieldIC *ic, sljit_sw keep_obj)
{
    JSContext *ctx = jc->ctx;
    JSAtom atom = (JSAtom)atom_w;
    JSValue obj = jc->sp[-1];
    jc->sf->cur_sp = jc->sp;
    if (ic && JS_VALUE_GET_TAG(obj) == JS_TAG_OBJECT) {
        JSObject *p = JS_VALUE_GET_OBJ(obj);
        if (!p->is_exotic) {
            JSProperty *pr = NULL;
            JSShapeProperty *prs = find_own_property(&pr, p, atom);
            if (prs && (prs->flags & JS_PROP_TMASK) == JS_PROP_NORMAL) {
                js_free_shape_null(ctx->rt, ic->shape);
                js_free_shape_null(ctx->rt, ic->holder_shape);
                ic->shape = p->shape ? js_dup_shape(p->shape) : NULL;
                ic->holder_shape = NULL;
                ic->offset = (uint32_t)(pr - p->prop);
                ic->is_writable = (prs->flags & JS_PROP_WRITABLE) != 0;
                ic->is_proto = 0;
            } else if (!prs && p->shape && p->shape->proto && !p->shape->proto->is_exotic) {
                JSObject *proto = p->shape->proto;
                JSShapeProperty *pprs = find_own_property(&pr, proto, atom);
                if (pprs && (pprs->flags & JS_PROP_TMASK) == JS_PROP_NORMAL) {
                    js_free_shape_null(ctx->rt, ic->shape);
                    js_free_shape_null(ctx->rt, ic->holder_shape);
                    ic->shape = js_dup_shape(p->shape);
                    ic->holder_shape = proto->shape ? js_dup_shape(proto->shape) : NULL;
                    ic->offset = (uint32_t)(pr - proto->prop);
                    ic->is_writable = 0;
                    ic->is_proto = 1;
                }
            }
        }
    }
    JSValue res = JS_GetProperty(ctx, obj, atom);
    if (JS_IsException(res))
        return -1;
    if (keep_obj) {
        *jc->sp++ = res;
    } else {
        JS_FreeValue(ctx, obj);
        jc->sp[-1] = res;
    }
    return 0;
}

static sljit_s32 jit_helper_put_field_ic(struct JitCtx *jc, sljit_sw atom_w,
                                         JitFieldIC *ic)
{
    JSContext *ctx = jc->ctx;
    JSAtom atom = (JSAtom)atom_w;
    JSValue obj = jc->sp[-2];
    JSValue val = jc->sp[-1];
    jc->sf->cur_sp = jc->sp;
    int ret = JS_SetProperty(ctx, obj, atom, val);
    if (ret >= 0 && ic && JS_VALUE_GET_TAG(obj) == JS_TAG_OBJECT) {
        JSObject *p = JS_VALUE_GET_OBJ(obj);
        if (!p->is_exotic) {
            JSProperty *pr = NULL;
            JSShapeProperty *prs = find_own_property(&pr, p, atom);
            if (prs && (prs->flags & JS_PROP_TMASK) == JS_PROP_NORMAL &&
                (prs->flags & JS_PROP_WRITABLE)) {
                js_free_shape_null(ctx->rt, ic->shape);
                ic->shape = p->shape ? js_dup_shape(p->shape) : NULL;
                ic->offset = (uint32_t)(pr - p->prop);
                ic->is_writable = 1;
            }
        }
    }
    JS_FreeValue(ctx, obj);
    jc->sp -= 2;
    return ret < 0 ? -1 : 0;
}

static sljit_s32 jit_helper_get_array_el(struct JitCtx *jc, sljit_sw keep_obj)
{
    JSContext *ctx = jc->ctx;
    jc->sf->cur_sp = jc->sp;
    JSValue val = JS_GetPropertyValue(ctx, jc->sp[-2], jc->sp[-1]);
    if (JS_IsException(val)) {
        if (!keep_obj) {
            JS_FreeValue(ctx, jc->sp[-2]);
            jc->sp -= 2;
        } else {
            jc->sp -= 1;
        }
        return -1;
    }
    if (keep_obj) {
        jc->sp[-1] = val;
    } else {
        JS_FreeValue(ctx, jc->sp[-2]);
        jc->sp[-2] = val;
        jc->sp--;
    }
    return 0;
}

static sljit_s32 jit_helper_put_array_el(struct JitCtx *jc)
{
    JSContext *ctx = jc->ctx;
    jc->sf->cur_sp = jc->sp;
    int ret = JS_SetPropertyValue(ctx, jc->sp[-3], jc->sp[-2], jc->sp[-1],
                                  JS_PROP_THROW);
    JS_FreeValue(ctx, jc->sp[-3]);
    jc->sp -= 3;
    return ret < 0 ? -1 : 0;
}

static sljit_s32 jit_helper_get_length(struct JitCtx *jc)
{
    JSContext *ctx = jc->ctx;
    jc->sf->cur_sp = jc->sp;
    JSValue obj = jc->sp[-1];
    JSValue res = JS_GetProperty(ctx, obj, JS_ATOM_length);
    JS_FreeValue(ctx, obj);
    if (JS_IsException(res)) {
        jc->sp--;
        return -1;
    }
    jc->sp[-1] = res;
    return 0;
}

static sljit_s32 jit_helper_binop(struct JitCtx *jc, sljit_sw op_w)
{
    JSContext *ctx = jc->ctx;
    int op = (int)op_w;
    int r;
    jc->sf->cur_sp = jc->sp;
    if (op == OP_add)
        r = js_add_slow(ctx, jc->sp);
    else if (op == OP_shl || op == OP_sar || op == OP_shr ||
             op == OP_and || op == OP_or || op == OP_xor)
        r = js_binary_logic_slow(ctx, jc->sp, (OPCodeEnum)op);
    else
        r = js_binary_arith_slow(ctx, jc->sp, (OPCodeEnum)op);
    jc->sp--;
    return r < 0 ? -1 : 0;
}

static sljit_s32 jit_helper_unary(struct JitCtx *jc, sljit_sw op_w)
{
    JSContext *ctx = jc->ctx;
    int op = (int)op_w;
    jc->sf->cur_sp = jc->sp;
    if (op == OP_lnot) {
        int b = JS_ToBool(ctx, jc->sp[-1]);
        JS_FreeValue(ctx, jc->sp[-1]);
        if (b < 0) {
            jc->sp--;
            return -1;
        }
        jc->sp[-1] = JS_NewBool(ctx, !b);
        return 0;
    }
    if (op == OP_not) {
        int r = js_not_slow(ctx, jc->sp);
        return r < 0 ? -1 : 0;
    }
    return js_unary_arith_slow(ctx, jc->sp, (OPCodeEnum)op) < 0 ? -1 : 0;
}

static sljit_s32 jit_helper_post_inc_dec(struct JitCtx *jc, sljit_sw op_w)
{
    JSContext *ctx = jc->ctx;
    jc->sf->cur_sp = jc->sp;
    int r = js_unary_arith_slow(ctx, jc->sp + 1, (OPCodeEnum)op_w);
    if (r < 0)
        return -1;
    jc->sp++;
    return 0;
}

static sljit_s32 jit_helper_cmp(struct JitCtx *jc, sljit_sw op_w)
{
    JSContext *ctx = jc->ctx;
    int op = (int)op_w;
    int r;
    jc->sf->cur_sp = jc->sp;
    if (op == OP_eq || op == OP_neq)
        r = js_eq_slow(ctx, jc->sp, op == OP_neq);
    else if (op == OP_strict_eq || op == OP_strict_neq)
        r = js_strict_eq_slow(ctx, jc->sp, op == OP_strict_neq);
    else
        r = js_relational_slow(ctx, jc->sp, (OPCodeEnum)op);
    jc->sp--;
    return r < 0 ? -1 : 0;
}

static sljit_s32 jit_helper_to_bool(struct JitCtx *jc)
{
    JSContext *ctx = jc->ctx;
    JSValue v = *--jc->sp;
    int b = JS_ToBool(ctx, v);
    JS_FreeValue(ctx, v);
    return b;
}

static sljit_s32 jit_helper_add_loc(struct JitCtx *jc, sljit_sw idx)
{
    JSContext *ctx = jc->ctx;
    JSValue ops[2];
    jc->sf->cur_sp = jc->sp;
    ops[0] = jc->var_buf[idx];
    ops[1] = *--jc->sp;
    if (js_add_slow(ctx, ops + 2) < 0)
        return -1;
    jc->var_buf[idx] = ops[0];
    return 0;
}

static sljit_s32 jit_helper_inc_dec_loc(struct JitCtx *jc, sljit_sw idx, sljit_sw op_w)
{
    JSContext *ctx = jc->ctx;
    JSValue v = jc->var_buf[idx];
    jc->sf->cur_sp = jc->sp;
    if (js_unary_arith_slow(ctx, &v + 1, (OPCodeEnum)op_w) < 0)
        return -1;
    jc->var_buf[idx] = v;
    return 0;
}

static sljit_s32 jit_helper_inc_dec_loc_chk(struct JitCtx *jc, sljit_sw idx, sljit_sw op_chk)
{
    int op = (int)(op_chk & 0xffff);
    int chk = (int)((op_chk >> 16) & 1);
    if (chk && unlikely(JS_IsUninitialized(jc->var_buf[idx]))) {
        JS_ThrowReferenceErrorUninitialized2(jc->ctx, jc->b, (int)idx, true);
        return -1;
    }
    return jit_helper_inc_dec_loc(jc, idx, (sljit_sw)op);
}

static sljit_s32 jit_helper_cmp_locs(struct JitCtx *jc, sljit_sw v1_p, sljit_sw v2_p, sljit_sw op_w)
{
    JSContext *ctx = jc->ctx;
    int v1 = (int)(v1_p & 0xffff), chk1 = (int)((v1_p >> 16) & 1);
    int v2 = (int)(v2_p & 0xffff), chk2 = (int)((v2_p >> 16) & 1);
    JSValue a = jc->var_buf[v1], b = jc->var_buf[v2];
    if (chk1 && unlikely(JS_IsUninitialized(a))) {
        JS_ThrowReferenceErrorUninitialized2(ctx, jc->b, v1, true);
        return -1;
    }
    if (chk2 && unlikely(JS_IsUninitialized(b))) {
        JS_ThrowReferenceErrorUninitialized2(ctx, jc->b, v2, true);
        return -1;
    }
    jc->sp[0] = JS_DupValue(ctx, a);
    jc->sp[1] = JS_DupValue(ctx, b);
    jc->sp += 2;
    if (jit_helper_cmp(jc, op_w) < 0)
        return -1;
    return jit_helper_to_bool(jc);
}

static sljit_s32 jit_helper_binop_locs(struct JitCtx *jc, sljit_sw v1_p, sljit_sw v2_p, sljit_sw op_w)
{
    JSContext *ctx = jc->ctx;
    int v1 = (int)(v1_p & 0xffff), chk1 = (int)((v1_p >> 16) & 1);
    int v2 = (int)(v2_p & 0xffff), chk2 = (int)((v2_p >> 16) & 1);
    JSValue a = jc->var_buf[v1], b = jc->var_buf[v2];
    if (chk1 && unlikely(JS_IsUninitialized(a))) {
        JS_ThrowReferenceErrorUninitialized2(ctx, jc->b, v1, true);
        return -1;
    }
    if (chk2 && unlikely(JS_IsUninitialized(b))) {
        JS_ThrowReferenceErrorUninitialized2(ctx, jc->b, v2, true);
        return -1;
    }
    jc->sp[0] = JS_DupValue(ctx, a);
    jc->sp[1] = JS_DupValue(ctx, b);
    jc->sp += 2;
    return jit_helper_binop(jc, op_w);
}

static sljit_s32 jit_helper_get_array_el_locs(struct JitCtx *jc, sljit_sw arr_p, sljit_sw idx_p, sljit_sw lnot_mode)
{
    JSContext *ctx = jc->ctx;
    int v1 = (int)(arr_p & 0xffff), chk1 = (int)((arr_p >> 16) & 1);
    int v2 = (int)(idx_p & 0xffff), chk2 = (int)((idx_p >> 16) & 1);
    JSValue a = jc->var_buf[v1], b = jc->var_buf[v2];
    if (chk1 && unlikely(JS_IsUninitialized(a))) {
        JS_ThrowReferenceErrorUninitialized2(ctx, jc->b, v1, true);
        return -1;
    }
    if (chk2 && unlikely(JS_IsUninitialized(b))) {
        JS_ThrowReferenceErrorUninitialized2(ctx, jc->b, v2, true);
        return -1;
    }
    jc->sp[0] = JS_DupValue(ctx, a);
    jc->sp[1] = JS_DupValue(ctx, b);
    jc->sp += 2;
    if (jit_helper_get_array_el(jc, 0) < 0)
        return -1;
    if (lnot_mode) {
        int bv = jit_helper_to_bool(jc);
        return (bv < 0) ? -1 : !bv;
    }
    return 0;
}

static sljit_s32 jit_helper_to_propkey(struct JitCtx *jc)
{
    JSContext *ctx = jc->ctx;
    JSValue v = jc->sp[-1];
    JSValue k = JS_ToPropertyKey(ctx, v);
    JS_FreeValue(ctx, v);
    if (JS_IsException(k)) {
        jc->sp--;
        return -1;
    }
    jc->sp[-1] = k;
    return 0;
}

static sljit_s32 jit_helper_to_object(struct JitCtx *jc)
{
    JSContext *ctx = jc->ctx;
    JSValue v = jc->sp[-1];
    JSValue o = JS_ToObject(ctx, v);
    JS_FreeValue(ctx, v);
    if (JS_IsException(o)) {
        jc->sp--;
        return -1;
    }
    jc->sp[-1] = o;
    return 0;
}

static void jit_helper_typeof(struct JitCtx *jc)
{
    JSContext *ctx = jc->ctx;
    JSValue v = jc->sp[-1];
    JSAtom a = JS_ATOM_undefined;
    switch (JS_VALUE_GET_NORM_TAG(v)) {
    case JS_TAG_INT:
    case JS_TAG_FLOAT64: a = JS_ATOM_number; break;
    case JS_TAG_BOOL: a = JS_ATOM_boolean; break;
    case JS_TAG_STRING:
    case JS_TAG_STRING_ROPE: a = JS_ATOM_string; break;
    case JS_TAG_SYMBOL: a = JS_ATOM_symbol; break;
    case JS_TAG_BIG_INT:
    case JS_TAG_SHORT_BIG_INT: a = JS_ATOM_bigint; break;
    case JS_TAG_OBJECT:
        a = JS_IsFunction(ctx, v) ? JS_ATOM_function : JS_ATOM_object;
        break;
    case JS_TAG_NULL: a = JS_ATOM_object; break;
    default: break;
    }
    JS_FreeValue(ctx, v);
    jc->sp[-1] = JS_AtomToString(ctx, a);
}

static void jit_helper_push_empty_string(struct JitCtx *jc)
{
    *jc->sp++ = js_empty_string(jc->ctx->rt);
}

static sljit_s32 jit_helper_push_this(struct JitCtx *jc)
{
    JSContext *ctx = jc->ctx;
    JSValue this_obj = jc->this_obj;
    JSValue val;
    if (!jc->b->is_strict_mode) {
        uint32_t tag = JS_VALUE_GET_TAG(this_obj);
        if (likely(tag == JS_TAG_OBJECT)) {
            val = js_dup(this_obj);
        } else if (tag == JS_TAG_NULL || tag == JS_TAG_UNDEFINED) {
            val = js_dup(ctx->global_obj);
        } else {
            val = JS_ToObject(ctx, this_obj);
            if (JS_IsException(val))
                return -1;
        }
    } else {
        val = js_dup(this_obj);
    }
    *jc->sp++ = val;
    return 0;
}

static sljit_s32 jit_helper_get_var_ref(struct JitCtx *jc, sljit_sw idx, sljit_sw check)
{
    JSContext *ctx = jc->ctx;
    JSObject *fn_obj = JS_VALUE_GET_OBJ(jc->sf->cur_func);
    JSVarRef **var_refs = fn_obj->u.func.var_refs;
    JSValue val = *var_refs[idx]->pvalue;
    if (check && unlikely(JS_IsUninitialized(val))) {
        JS_ThrowReferenceErrorUninitialized2(ctx, jc->b, (int)idx, true);
        return -1;
    }
    *jc->sp++ = js_dup(val);
    return 0;
}

/* mode: 0=put, 1=set, 2=put_check, 3=put_check_init */
static sljit_s32 jit_helper_put_var_ref(struct JitCtx *jc, sljit_sw idx, sljit_sw mode)
{
    JSContext *ctx = jc->ctx;
    JSObject *fn_obj = JS_VALUE_GET_OBJ(jc->sf->cur_func);
    JSVarRef **var_refs = fn_obj->u.func.var_refs;
    if (mode == 2 && unlikely(JS_IsUninitialized(*var_refs[idx]->pvalue))) {
        JS_ThrowReferenceErrorUninitialized2(ctx, jc->b, (int)idx, true);
        return -1;
    }
    if (mode == 3 && unlikely(!JS_IsUninitialized(*var_refs[idx]->pvalue))) {
        JS_ThrowReferenceErrorUninitialized2(ctx, jc->b, (int)idx, true);
        return -1;
    }
    if (mode == 1) {
        set_value(ctx, var_refs[idx]->pvalue, js_dup(jc->sp[-1]));
    } else {
        set_value(ctx, var_refs[idx]->pvalue, jc->sp[-1]);
        jc->sp--;
    }
    return 0;
}

/* ---- pre-scan: is every opcode in the supported int fast-path set? ---- */
static int jit_prescan(JSFunctionBytecode *b, int *n_ic_out)
{
    const uint8_t *p = b->byte_code_buf;
    const uint8_t *end = p + b->byte_code_len;
    int nic = 0;
    while (p < end) {
        int op = *p;
        switch (op) {
        case OP_nop:
        case OP_push_i8: case OP_push_i16: case OP_push_i32:
        case OP_push_0: case OP_push_1: case OP_push_2: case OP_push_3:
        case OP_push_4: case OP_push_5: case OP_push_6: case OP_push_7:
        case OP_push_minus1:
        case OP_push_const8: case OP_push_const: case OP_push_atom_value:
        case OP_push_empty_string: case OP_push_this:
        case OP_undefined: case OP_null: case OP_push_false: case OP_push_true:
        case OP_drop: case OP_nip: case OP_nip1:
        case OP_dup: case OP_dup1: case OP_dup2: case OP_swap:
        case OP_insert3: case OP_perm3:
        case OP_rot3l: case OP_rot3r:
        case OP_get_loc: case OP_get_loc0: case OP_get_loc1:
        case OP_get_loc2: case OP_get_loc3: case OP_get_loc8:
        case OP_put_loc: case OP_put_loc0: case OP_put_loc1:
        case OP_put_loc2: case OP_put_loc3: case OP_put_loc8:
        case OP_set_loc: case OP_set_loc0: case OP_set_loc1:
        case OP_set_loc2: case OP_set_loc3: case OP_set_loc8:
        case OP_set_loc_uninitialized:
        case OP_get_loc_check: case OP_put_loc_check: case OP_put_loc_check_init:
        case OP_get_var_ref: case OP_put_var_ref: case OP_set_var_ref:
        case OP_get_var_ref_check: case OP_put_var_ref_check: case OP_put_var_ref_check_init:
        case OP_get_var_ref0: case OP_get_var_ref1: case OP_get_var_ref2: case OP_get_var_ref3:
        case OP_put_var_ref0: case OP_put_var_ref1: case OP_put_var_ref2: case OP_put_var_ref3:
        case OP_set_var_ref0: case OP_set_var_ref1: case OP_set_var_ref2: case OP_set_var_ref3:
        case OP_get_arg: case OP_put_arg: case OP_set_arg:
        case OP_get_arg0: case OP_get_arg1: case OP_get_arg2: case OP_get_arg3:
        case OP_put_arg0: case OP_put_arg1: case OP_put_arg2: case OP_put_arg3:
        case OP_set_arg0: case OP_set_arg1: case OP_set_arg2: case OP_set_arg3:
        case OP_inc_loc: case OP_dec_loc: case OP_add_loc:
        case OP_inc: case OP_dec: case OP_post_inc: case OP_post_dec:
        case OP_neg: case OP_plus:
        case OP_not: case OP_lnot:
        case OP_add: case OP_sub: case OP_mul: case OP_div: case OP_mod:
        case OP_shl: case OP_sar: case OP_shr:
        case OP_and: case OP_or: case OP_xor:
        case OP_lt: case OP_lte: case OP_gt: case OP_gte:
        case OP_eq: case OP_neq: case OP_strict_eq: case OP_strict_neq:
        case OP_goto: case OP_goto8: case OP_goto16:
        case OP_if_true: case OP_if_false:
        case OP_if_true8: case OP_if_false8:
        case OP_return: case OP_return_undef:
        case OP_get_var: case OP_get_var_undef:
        case OP_put_var: case OP_put_var_init:
        case OP_object: case OP_array_from:
        case OP_fclosure: case OP_fclosure8:
        case OP_call: case OP_call0: case OP_call1: case OP_call2: case OP_call3:
        case OP_call_method: case OP_call_constructor:
        case OP_get_array_el: case OP_get_array_el2: case OP_put_array_el:
        case OP_get_length: case OP_to_propkey: case OP_to_propkey2:
        case OP_to_object: case OP_typeof:
        case OP_is_undefined_or_null:
            break;
        case OP_get_field: case OP_get_field2: case OP_put_field: case OP_define_field:
            nic++;
            break;
        default:
            if (jit_trace_level >= 1)
                fprintf(stderr, "[jit] prescan refuse op=%d (idx %d)\n",
                        op, (int)(p - b->byte_code_buf));
            return 0;
        }
        p += short_opcode_info(op).size;
    }
    if (n_ic_out)
        *n_ic_out = nic;
    return 1;
}

/* linear lookup of the opcode index whose byte offset == off */
static int jit_find_label_idx(const int *offs, int nops, int off)
{
    int i;
    for (i = 0; i < nops; i++)
        if (offs[i] == off)
            return i;
    return -1;
}

static int jit_decode_get_loc(const uint8_t *p, int *lidx, int *chk)
{
    int op = *p;
    *chk = 0;
    switch (op) {
    case OP_get_loc0: case OP_get_loc1: case OP_get_loc2: case OP_get_loc3:
        *lidx = op - OP_get_loc0; return 1;
    case OP_get_loc8:
        *lidx = p[1]; return 1;
    case OP_get_loc:
        *lidx = (int)get_u16(p + 1); return 1;
    case OP_get_loc_check:
        *lidx = (int)get_u16(p + 1); *chk = 1; return 1;
    default: return 0;
    }
}

static int jit_decode_put_loc(const uint8_t *p, int *lidx, int *chk)
{
    int op = *p;
    *chk = 0;
    switch (op) {
    case OP_put_loc0: case OP_put_loc1: case OP_put_loc2: case OP_put_loc3:
        *lidx = op - OP_put_loc0; return 1;
    case OP_put_loc8:
        *lidx = p[1]; return 1;
    case OP_put_loc:
        *lidx = (int)get_u16(p + 1); return 1;
    case OP_put_loc_check:
        *lidx = (int)get_u16(p + 1); *chk = 1; return 1;
    default: return 0;
    }
}

static int jit_decode_cmp(int op, sljit_s32 *cond_out)
{
    switch (op) {
    case OP_lt:         *cond_out = SLJIT_SIG_LESS; return 1;
    case OP_lte:        *cond_out = SLJIT_SIG_LESS_EQUAL; return 1;
    case OP_gt:         *cond_out = SLJIT_SIG_GREATER; return 1;
    case OP_gte:        *cond_out = SLJIT_SIG_GREATER_EQUAL; return 1;
    case OP_eq:
    case OP_strict_eq:  *cond_out = SLJIT_EQUAL; return 1;
    case OP_neq:
    case OP_strict_neq: *cond_out = SLJIT_NOT_EQUAL; return 1;
    default: return 0;
    }
}

static sljit_s32 jit_invert_cond(sljit_s32 cond)
{
    switch (cond) {
    case SLJIT_SIG_LESS:          return SLJIT_SIG_GREATER_EQUAL;
    case SLJIT_SIG_LESS_EQUAL:    return SLJIT_SIG_GREATER;
    case SLJIT_SIG_GREATER:       return SLJIT_SIG_LESS_EQUAL;
    case SLJIT_SIG_GREATER_EQUAL: return SLJIT_SIG_LESS;
    case SLJIT_EQUAL:             return SLJIT_NOT_EQUAL;
    case SLJIT_NOT_EQUAL:         return SLJIT_EQUAL;
    default: return cond;
    }
}

static int jit_decode_branch(const uint8_t *p, int ooff, int *is_true, int *target)
{
    int op = *p, rel;
    if (op == OP_if_true || op == OP_if_false) {
        rel = (int)(int32_t)get_u32(p + 1);
        *is_true = (op == OP_if_true);
    } else if (op == OP_if_true8 || op == OP_if_false8) {
        rel = (int)(int8_t)p[1];
        *is_true = (op == OP_if_true8);
    } else {
        return 0;
    }
    *target = ooff + 1 + rel;
    return 1;
}

static void jit_emit_sync_call(struct sljit_compiler *c, void *fn, int nargs,
                               sljit_sw a1, sljit_sw a2, sljit_sw a3,
                               int check_exc, struct jit_fixup *fx, int *pnfx)
{
    sljit_s32 arg_types = (nargs == 0) ? SLJIT_ARGS1(W, P) :
                          (nargs == 1) ? SLJIT_ARGS2(W, P, W) :
                          (nargs == 2) ? SLJIT_ARGS3(W, P, W, W) :
                                         SLJIT_ARGS4(W, P, W, W, W);
    if (!check_exc)
        arg_types = (nargs == 0) ? SLJIT_ARGS1V(P) :
                    (nargs == 1) ? SLJIT_ARGS2V(P, W) :
                    (nargs == 2) ? SLJIT_ARGS3V(P, W, W) :
                                   SLJIT_ARGS4V(P, W, W, W);
    sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_JCTX), JCTX_SP, JREG_SP, 0);
    sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, JREG_JCTX, 0);
    if (nargs >= 1) sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_IMM, a1);
    if (nargs >= 2) sljit_emit_op1(c, SLJIT_MOV, SLJIT_R2, 0, SLJIT_IMM, a2);
    if (nargs >= 3) sljit_emit_op1(c, SLJIT_MOV, SLJIT_R3, 0, SLJIT_IMM, a3);
    sljit_emit_icall(c, SLJIT_CALL, arg_types, SLJIT_IMM, (sljit_sw)fn);
    sljit_emit_op1(c, SLJIT_MOV, JREG_SP, 0, SLJIT_MEM1(JREG_JCTX), JCTX_SP);
    if (check_exc) {
        struct sljit_jump *je = sljit_emit_cmp(c, SLJIT_SIG_LESS, SLJIT_RETURN_REG, 0, SLJIT_IMM, 0);
        fx[*pnfx].jmp = je;
        fx[*pnfx].kind = JIT_FX_EXC;
        (*pnfx)++;
    }
}

/* Emit inline JS_FreeValue for a stack slot [JREG_SP + stk_off] known to be JS_TAG_OBJECT,
   where `obj_reg` already holds JSObject *p (cannot be SLJIT_R0). */
static void jit_emit_free_obj_slot(struct sljit_compiler *c, sljit_s32 obj_reg, sljit_sw stk_off)
{
    struct sljit_jump *jslow, *jdone;
    sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_MEM1(obj_reg), JIT_RC_OFF);
    jslow = sljit_emit_cmp(c, SLJIT_SIG_LESS_EQUAL, SLJIT_R0, 0, SLJIT_IMM, 1);
    sljit_emit_op2(c, SLJIT_SUB | SLJIT_32, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_IMM, 1);
    sljit_emit_op1(c, SLJIT_MOV32, SLJIT_MEM1(obj_reg), JIT_RC_OFF, SLJIT_R0, 0);
    jdone = sljit_emit_jump(c, SLJIT_JUMP);
    sljit_set_label(jslow, sljit_emit_label(c));
    sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_JCTX), JCTX_CTX);
    sljit_emit_op2(c, SLJIT_ADD, SLJIT_R1, 0, JREG_SP, 0, SLJIT_IMM, stk_off);
    sljit_emit_icall(c, SLJIT_CALL, SLJIT_ARGS2V(P, P),
                     SLJIT_IMM, (sljit_sw)&jit_helper_free_val);
    sljit_set_label(jdone, sljit_emit_label(c));
}

/* ================================================================== */
/* Native Loop Region JIT (unboxed int32 / float64 / TypedArray loop) */
/* ================================================================== */

#define NLOOP_MAX_OPS   256
#define NLOOP_MAX_LOCS  24
#define NLOOP_MAX_ARRS  12
#define NLOOP_MAX_STK   16
#define NLOOP_LOC_OFF   0
#define NLOOP_ARR_OFF   (NLOOP_MAX_LOCS * 8)                 /* 192 */
#define NLOOP_TMP_OFF   (NLOOP_ARR_OFF + NLOOP_MAX_ARRS * 16) /* 384 */
#define NLOOP_FRAME_SZ  (NLOOP_TMP_OFF + 64)                /* 448 */

enum { NLOC_I32 = 0, NLOC_F64 = 1, NLOC_ARR = 2 };
enum {
    NVK_IMM_I32 = 0,
    NVK_LOC_I32,
    NVK_LOC_F64,
    NVK_LOC_ARR,
    NVK_REG_I32,
    NVK_REG_F64,
    NVK_ARR_EL2,
    NVK_TMP_I32
};

typedef struct {
    int kind;
    sljit_s32 reg;
    sljit_sw imm;
    int loc; /* index into nlocs[] or narrs[] */
} NLoopVal;

typedef struct {
    int var_idx;
    int type;    /* NLOC_I32, NLOC_F64, NLOC_ARR */
    int written; /* modified inside loop */
    int arr_idx; /* if NLOC_ARR: 0..narrs-1 */
    sljit_s32 reg; /* >= 0 if resident in SLJIT_S* */
} NLoopLoc;

static int nloop_decode_push_imm(const uint8_t *p, int32_t *out)
{
    int op = *p;
    if (op >= OP_push_0 && op <= OP_push_7) { *out = op - OP_push_0; return 1; }
    if (op == OP_push_minus1) { *out = -1; return 1; }
    if (op == OP_push_i8)     { *out = (int8_t)p[1]; return 1; }
    if (op == OP_push_i16)    { *out = (int16_t)get_u16(p + 1); return 1; }
    if (op == OP_push_i32)    { *out = (int32_t)get_u32(p + 1); return 1; }
    return 0;
}

static int nloop_find_loc(NLoopLoc *locs, int *nlocs, int var_idx)
{
    for (int i = 0; i < *nlocs; i++)
        if (locs[i].var_idx == var_idx) return i;
    if (*nlocs >= NLOOP_MAX_LOCS) return -1;
    int idx = (*nlocs)++;
    locs[idx].var_idx = var_idx;
    locs[idx].type = NLOC_I32;
    locs[idx].written = 0;
    locs[idx].arr_idx = -1;
    locs[idx].reg = -1;
    return idx;
}

static void nloop_emit_writeback(struct sljit_compiler *c, const NLoopLoc *locs, int nlocs)
{
    for (int k = 0; k < nlocs; k++) {
        if (!locs[k].written || locs[k].type == NLOC_ARR) continue;
        int v = locs[k].var_idx;
        if (locs[k].type == NLOC_I32) {
            if (locs[k].reg >= 0) {
                sljit_emit_op1(c, SLJIT_MOV32, SLJIT_MEM1(JREG_VAR), v * 16 + JSV_U, locs[k].reg, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_VAR), v * 16 + JSV_TAG, SLJIT_IMM, JS_TAG_INT);
            } else {
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + k * 8);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_VAR), v * 16 + JSV_U, SLJIT_R0, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_VAR), v * 16 + JSV_TAG, SLJIT_IMM, JS_TAG_INT);
            }
        } else {
            sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_FR0, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + k * 8);
            sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_MEM1(JREG_VAR), v * 16 + JSV_U, SLJIT_FR0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_VAR), v * 16 + JSV_TAG, SLJIT_IMM, JS_TAG_FLOAT64);
        }
    }
    sljit_emit_op1(c, SLJIT_MOV, SLJIT_S1, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_TMP_OFF + 16);
    sljit_emit_op1(c, SLJIT_MOV, SLJIT_S4, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_TMP_OFF + 24);
}

static int nloop_alloc_ireg(const NLoopVal *stk, int sp)
{
    int used[4] = {0, 0, 0, 0};
    for (int i = 0; i < sp; i++) {
        if (stk[i].kind == NVK_REG_I32) {
            for (int r = 0; r < 4; r++)
                if (stk[i].reg == SLJIT_R(r)) used[r] = 1;
        }
    }
    for (int r = 0; r < 4; r++)
        if (!used[r]) return SLJIT_R(r);
    return -1;
}

static int nloop_alloc_freg(const NLoopVal *stk, int sp)
{
    int used[4] = {0, 0, 0, 0};
    for (int i = 0; i < sp; i++) {
        if (stk[i].kind == NVK_REG_F64) {
            for (int r = 0; r < 4; r++)
                if (stk[i].reg == SLJIT_FR(r)) used[r] = 1;
        }
    }
    for (int r = 0; r < 4; r++)
        if (!used[r]) return SLJIT_FR(r);
    return -1;
}

static void nloop_to_ireg(struct sljit_compiler *c, const NLoopLoc *locs, const NLoopVal *v, sljit_s32 reg)
{
    if (v->kind == NVK_REG_I32) {
        if (v->reg != reg) sljit_emit_op1(c, SLJIT_MOV_S32, reg, 0, v->reg, 0);
    } else if (v->kind == NVK_IMM_I32) {
        sljit_emit_op1(c, SLJIT_MOV, reg, 0, SLJIT_IMM, v->imm);
    } else if (v->kind == NVK_LOC_I32) {
        if (locs[v->loc].reg >= 0) {
            if (locs[v->loc].reg != reg)
                sljit_emit_op1(c, SLJIT_MOV_S32, reg, 0, locs[v->loc].reg, 0);
        } else {
            sljit_emit_op1(c, SLJIT_MOV_S32, reg, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + v->loc * 8);
        }
    } else if (v->kind == NVK_TMP_I32) {
        sljit_emit_op1(c, SLJIT_MOV_S32, reg, 0, SLJIT_MEM1(SLJIT_SP), v->imm);
    }
}

static void nloop_to_freg(struct sljit_compiler *c, const NLoopLoc *locs, const NLoopVal *v, sljit_s32 freg)
{
    if (v->kind == NVK_REG_F64) {
        if (v->reg != freg) sljit_emit_fop1(c, SLJIT_MOV_F64, freg, 0, v->reg, 0);
    } else if (v->kind == NVK_LOC_F64) {
        sljit_emit_fop1(c, SLJIT_MOV_F64, freg, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + v->loc * 8);
    } else if (v->kind == NVK_LOC_I32) {
        if (locs[v->loc].reg >= 0)
            sljit_emit_fop1(c, SLJIT_CONV_F64_FROM_S32, freg, 0, locs[v->loc].reg, 0);
        else
            sljit_emit_fop1(c, SLJIT_CONV_F64_FROM_S32, freg, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + v->loc * 8);
    } else if (v->kind == NVK_REG_I32) {
        sljit_emit_fop1(c, SLJIT_CONV_F64_FROM_S32, freg, 0, v->reg, 0);
    } else if (v->kind == NVK_IMM_I32) {
        sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_IMM, v->imm);
        sljit_emit_fop1(c, SLJIT_CONV_F64_FROM_SW, freg, 0, SLJIT_R0, 0);
    }
}

/* Try to compile a pure numeric + TypedArray loop region starting at `i_head`.
   Returns 1 if emitted (with `i_head` pre-header + native loop + Tier 2 fallback), 0 otherwise. */
static int jit_try_emit_nloop(struct sljit_compiler *c, const uint8_t *bc,
                              const int *offs, int nops, int i_head,
                              struct jit_fixup *fx, int *pnfx,
                              int *i_end_out)
{
    int head_off = offs[i_head];
    int i_end = -1;
    for (int k = nops - 1; k > i_head; k--) {
        const uint8_t *kp = bc + offs[k];
        int kop = *kp, is_t, tgt;
        if (jit_decode_branch(kp, offs[k], &is_t, &tgt) && tgt == head_off) {
            i_end = k;
            break;
        }
        if ((kop == OP_goto || kop == OP_goto8 || kop == OP_goto16)) {
            int rel = (kop == OP_goto) ? (int)(int32_t)get_u32(kp + 1)
                    : (kop == OP_goto16) ? (int)(int16_t)get_u16(kp + 1)
                    : (int)(int8_t)kp[1];
            if (offs[k] + 1 + rel == head_off) { i_end = k; break; }
        }
    }
    if (i_end < 0 || i_end - i_head < 3 || i_end - i_head >= NLOOP_MAX_OPS)
        return 0;

    int end_off = offs[i_end] + short_opcode_info(bc[offs[i_end]]).size;
    NLoopLoc locs[NLOOP_MAX_LOCS];
    int nlocs = 0, narrs = 0;

    /* Pass 1: validate opcodes, stack depth, branch targets, and classify locals */
    for (int pass = 0; pass < 2; pass++) {
        int stk_type[NLOOP_MAX_STK];
        int stk_loc[NLOOP_MAX_STK];
        int sp = 0;
        for (int k = i_head; k <= i_end; k++) {
            const uint8_t *kp = bc + offs[k];
            int op = *kp, v, chk;
            int32_t imm;
            if (jit_decode_get_loc(kp, &v, &chk)) {
                int li = nloop_find_loc(locs, &nlocs, v);
                if (li < 0 || sp >= NLOOP_MAX_STK) return 0;
                stk_type[sp] = locs[li].type;
                stk_loc[sp] = li;
                sp++;
            } else if (jit_decode_put_loc(kp, &v, &chk) ||
                       op == OP_set_loc0 || op == OP_set_loc1 || op == OP_set_loc2 ||
                       op == OP_set_loc3 || op == OP_set_loc8 || op == OP_set_loc) {
                int is_set = (op >= OP_set_loc0 && op <= OP_set_loc3) || op == OP_set_loc8 || op == OP_set_loc;
                if (is_set) {
                    v = (op == OP_set_loc) ? (int)get_u16(kp + 1)
                      : (op == OP_set_loc8) ? (int)kp[1] : (op - OP_set_loc0);
                }
                if (sp < 1) return 0;
                int li = nloop_find_loc(locs, &nlocs, v);
                if (li < 0 || locs[li].type == NLOC_ARR) return 0;
                locs[li].written = 1;
                if (stk_type[sp - 1] == NLOC_F64) locs[li].type = NLOC_F64;
                if (!is_set) sp--;
            } else if (nloop_decode_push_imm(kp, &imm)) {
                if (sp >= NLOOP_MAX_STK) return 0;
                stk_type[sp] = NLOC_I32; stk_loc[sp] = -1; sp++;
            } else if (op == OP_inc_loc || op == OP_dec_loc) {
                int li = nloop_find_loc(locs, &nlocs, kp[1]);
                if (li < 0 || locs[li].type == NLOC_ARR) return 0;
                locs[li].written = 1;
            } else if (op == OP_add_loc) {
                if (sp < 1) return 0;
                int li = nloop_find_loc(locs, &nlocs, kp[1]);
                if (li < 0 || locs[li].type == NLOC_ARR) return 0;
                locs[li].written = 1;
                if (stk_type[sp - 1] == NLOC_F64) locs[li].type = NLOC_F64;
                sp--;
            } else if (op == OP_post_inc || op == OP_post_dec) {
                if (sp < 1 || sp >= NLOOP_MAX_STK) return 0;
                stk_type[sp] = stk_type[sp - 1]; stk_loc[sp] = -1; sp++;
            } else if (op == OP_add || op == OP_sub || op == OP_mul) {
                if (sp < 2) return 0;
                int t = (stk_type[sp - 2] == NLOC_F64 || stk_type[sp - 1] == NLOC_F64) ? NLOC_F64 : NLOC_I32;
                sp--;
                stk_type[sp - 1] = t; stk_loc[sp - 1] = -1;
            } else if (op == OP_lt || op == OP_lte || op == OP_gt || op == OP_gte ||
                       op == OP_eq || op == OP_neq || op == OP_strict_eq || op == OP_strict_neq) {
                if (sp < 2) return 0;
                sp--;
                stk_type[sp - 1] = NLOC_I32; stk_loc[sp - 1] = -1;
            } else if (op == OP_lnot) {
                if (sp < 1) return 0;
                stk_type[sp - 1] = NLOC_I32; stk_loc[sp - 1] = -1;
            } else if (op == OP_set_loc_uninitialized) {
                int li = nloop_find_loc(locs, &nlocs, (int)get_u16(kp + 1));
                if (li < 0 || locs[li].type == NLOC_ARR) return 0;
                locs[li].written = 1;
            } else if (op == OP_to_propkey || op == OP_to_propkey2) {
                if (sp < 1 || stk_type[sp - 1] != NLOC_I32) return 0;
            } else if (op == OP_swap) {
                if (sp < 2) return 0;
                int tt = stk_type[sp - 1]; stk_type[sp - 1] = stk_type[sp - 2]; stk_type[sp - 2] = tt;
                int tl = stk_loc[sp - 1]; stk_loc[sp - 1] = stk_loc[sp - 2]; stk_loc[sp - 2] = tl;
            } else if (op == OP_dup) {
                if (k + 2 <= i_end && bc[offs[k + 1]] == OP_is_undefined_or_null) {
                    int is_t, tgt;
                    if (jit_decode_branch(bc + offs[k + 2], offs[k + 2], &is_t, &tgt)) {
                        if (!is_t) {
                            int ti = jit_find_label_idx(offs, nops, tgt);
                            if (ti > k + 2 && ti <= i_end) { k = ti - 1; continue; }
                        } else {
                            k += 2; continue;
                        }
                    }
                }
                if (sp < 1 || sp >= NLOOP_MAX_STK) return 0;
                stk_type[sp] = stk_type[sp - 1]; stk_loc[sp] = stk_loc[sp - 1]; sp++;
            } else if (op == OP_dup2) {
                if (sp < 2 || sp + 2 > NLOOP_MAX_STK) return 0;
                stk_type[sp] = stk_type[sp - 2]; stk_loc[sp] = stk_loc[sp - 2];
                stk_type[sp + 1] = stk_type[sp - 1]; stk_loc[sp + 1] = stk_loc[sp - 1];
                sp += 2;
            } else if (op == OP_is_undefined_or_null) {
                if (sp < 1) return 0;
                if (k + 1 <= i_end) {
                    int is_t, tgt;
                    if (jit_decode_branch(bc + offs[k + 1], offs[k + 1], &is_t, &tgt) && is_t) {
                        k++; /* never null/undef -> if_true never taken */
                        continue;
                    }
                }
                stk_type[sp - 1] = NLOC_I32; stk_loc[sp - 1] = -1;
            } else if (op == OP_drop) {
                if (sp < 1) return 0;
                sp--;
            } else if (op == OP_get_array_el || op == OP_get_array_el2) {
                if (sp < 2 || stk_loc[sp - 2] < 0) return 0;
                int ali = stk_loc[sp - 2];
                locs[ali].type = NLOC_ARR;
                if (locs[ali].arr_idx < 0) {
                    if (narrs >= NLOOP_MAX_ARRS) return 0;
                    locs[ali].arr_idx = narrs++;
                }
                int is_lnot_next = (k + 1 <= i_end && bc[offs[k + 1]] == OP_lnot);
                int rtype = is_lnot_next ? NLOC_I32 : NLOC_F64;
                if (op == OP_get_array_el) {
                    sp--;
                    stk_type[sp - 1] = rtype; stk_loc[sp - 1] = -1;
                } else {
                    if (sp >= NLOOP_MAX_STK) return 0;
                    stk_type[sp] = rtype; stk_loc[sp] = -1; sp++;
                }
            } else if (op == OP_put_array_el) {
                if (sp < 3 || stk_loc[sp - 3] < 0) return 0;
                int ali = stk_loc[sp - 3];
                locs[ali].type = NLOC_ARR;
                if (locs[ali].arr_idx < 0) {
                    if (narrs >= NLOOP_MAX_ARRS) return 0;
                    locs[ali].arr_idx = narrs++;
                }
                sp -= 3;
            } else if (op == OP_goto || op == OP_goto8 || op == OP_goto16 ||
                       op == OP_if_true || op == OP_if_true8 || op == OP_if_false || op == OP_if_false8) {
                int tgt;
                if (op == OP_goto || op == OP_goto8 || op == OP_goto16) {
                    int rel = (op == OP_goto) ? (int)(int32_t)get_u32(kp + 1)
                            : (op == OP_goto16) ? (int)(int16_t)get_u16(kp + 1)
                            : (int)(int8_t)kp[1];
                    tgt = offs[k] + 1 + rel;
                } else {
                    int is_t;
                    jit_decode_branch(kp, offs[k], &is_t, &tgt);
                    if (sp < 1) return 0;
                    sp--;
                }
                if (tgt < head_off) { if (jit_trace_level >= 1) fprintf(stderr, "[nloop] head=%d end=%d fail tgt<head op=%d\n", i_head, i_end, op); return 0; }
                if (sp != 0) { if (jit_trace_level >= 1) fprintf(stderr, "[nloop] head=%d end=%d fail sp=%d at br op=%d\n", i_head, i_end, sp, op); return 0; } /* require empty stack at all branches */
            } else {
                if (jit_trace_level >= 1) fprintf(stderr, "[nloop] head=%d end=%d unsupported op=%d at k=%d\n", i_head, i_end, op, k);
                return 0;
            }
        }
        if (sp != 0) { if (jit_trace_level >= 1) fprintf(stderr, "[nloop] head=%d end=%d end sp=%d\n", i_head, i_end, sp); return 0; }
    }

    /* Ensure no NLOC_ARR was also marked written */
    for (int k = 0; k < nlocs; k++)
        if (locs[k].type == NLOC_ARR && locs[k].written) { if (jit_trace_level >= 1) fprintf(stderr, "[nloop] head=%d end=%d arr written\n", i_head, i_end); return 0; }
    if (jit_trace_level >= 1) fprintf(stderr, "[nloop] COMPILED head=%d end=%d nlocs=%d narrs=%d\n", i_head, i_end, nlocs, narrs);

    /* ---- Emit Pre-header at labels[i_head] ---- */
    struct sljit_jump *gf[NLOOP_MAX_LOCS * 4 + 8];
    int ngf = 0;
    for (int k = 0; k < nlocs; k++) {
        int v = locs[k].var_idx;
        if (locs[k].type == NLOC_ARR) {
            int ai = locs[k].arr_idx;
            struct sljit_jump *j_u8, *j_ok;
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_VAR), v * 16 + JSV_TAG);
            gf[ngf++] = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_OBJECT);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R2, 0, SLJIT_MEM1(JREG_VAR), v * 16 + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV_U16, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R2),
                           (sljit_sw)offsetof(JSObject, class_id));
            j_u8 = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_CLASS_UINT8_ARRAY);
            gf[ngf++] = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_CLASS_FLOAT64_ARRAY);
            sljit_emit_op1(c, SLJIT_MOV32, SLJIT_MEM1(SLJIT_SP), NLOOP_ARR_OFF + ai * 16 + 12, SLJIT_IMM, 1);
            j_ok = sljit_emit_jump(c, SLJIT_JUMP);
            sljit_set_label(j_u8, sljit_emit_label(c));
            sljit_emit_op1(c, SLJIT_MOV32, SLJIT_MEM1(SLJIT_SP), NLOOP_ARR_OFF + ai * 16 + 12, SLJIT_IMM, 0);
            sljit_set_label(j_ok, sljit_emit_label(c));
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_R2),
                           (sljit_sw)offsetof(JSObject, u.array.u.ptr));
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), NLOOP_ARR_OFF + ai * 16 + 0, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV_U32, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_R2),
                           (sljit_sw)offsetof(JSObject, u.array.count));
            sljit_emit_op1(c, SLJIT_MOV32, SLJIT_MEM1(SLJIT_SP), NLOOP_ARR_OFF + ai * 16 + 8, SLJIT_R0, 0);
        } else if (locs[k].type == NLOC_I32) {
            struct sljit_jump *j_int, *j_done;
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_VAR), v * 16 + JSV_TAG);
            j_int = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_INT);
            if (locs[k].written) {
                struct sljit_jump *j_u = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_UNDEFINED);
                gf[ngf++] = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_UNINITIALIZED);
                sljit_set_label(j_u, sljit_emit_label(c));
                sljit_emit_op1(c, SLJIT_MOV32, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + k * 8, SLJIT_IMM, 0);
                j_done = sljit_emit_jump(c, SLJIT_JUMP);
            } else {
                gf[ngf++] = sljit_emit_jump(c, SLJIT_JUMP);
                j_done = NULL;
            }
            sljit_set_label(j_int, sljit_emit_label(c));
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_MEM1(JREG_VAR), v * 16 + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV32, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + k * 8, SLJIT_R0, 0);
            if (j_done) sljit_set_label(j_done, sljit_emit_label(c));
        } else { /* NLOC_F64 */
            struct sljit_jump *j_f64, *j_int, *j_done;
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_VAR), v * 16 + JSV_TAG);
            j_f64 = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_FLOAT64);
            j_int = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_INT);
            if (locs[k].written) {
                struct sljit_jump *j_u = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_UNDEFINED);
                gf[ngf++] = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_UNINITIALIZED);
                sljit_set_label(j_u, sljit_emit_label(c));
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + k * 8, SLJIT_IMM, 0);
                j_done = sljit_emit_jump(c, SLJIT_JUMP);
            } else {
                gf[ngf++] = sljit_emit_jump(c, SLJIT_JUMP);
                j_done = NULL;
            }
            sljit_set_label(j_int, sljit_emit_label(c));
            sljit_emit_fop1(c, SLJIT_CONV_F64_FROM_S32, SLJIT_FR0, 0, SLJIT_MEM1(JREG_VAR), v * 16 + JSV_U);
            sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + k * 8, SLJIT_FR0, 0);
            struct sljit_jump *j_done2 = sljit_emit_jump(c, SLJIT_JUMP);
            sljit_set_label(j_f64, sljit_emit_label(c));
            sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_FR0, 0, SLJIT_MEM1(JREG_VAR), v * 16 + JSV_U);
            sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + k * 8, SLJIT_FR0, 0);
            struct sljit_label *ld = sljit_emit_label(c);
            if (j_done) sljit_set_label(j_done, ld);
            sljit_set_label(j_done2, ld);
        }
    }

    /* Save JREG_SP (S1) and JREG_ARG (S4) and load up to 4 hot NLOC_I32 locals into S5, S6, S1, S4 */
    {
        static const sljit_s32 sregs[4] = { SLJIT_S5, SLJIT_S6, SLJIT_S1, SLJIT_S4 };
        int nsr = 0;
        sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), NLOOP_TMP_OFF + 16, SLJIT_S1, 0);
        sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), NLOOP_TMP_OFF + 24, SLJIT_S4, 0);
        for (int pass_w = 1; pass_w >= 0 && nsr < 4; pass_w--) {
            for (int k = nlocs - 1; k >= 0 && nsr < 4; k--) {
                if (locs[k].type == NLOC_I32 && locs[k].written == pass_w && locs[k].reg < 0) {
                    locs[k].reg = sregs[nsr++];
                    sljit_emit_op1(c, SLJIT_MOV_S32, locs[k].reg, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + k * 8);
                }
            }
        }
    }

    /* ---- Emit Native Loop Body ---- */
    struct sljit_label *nlabels[NLOOP_MAX_OPS];
    struct { struct sljit_jump *jmp; int target_off; } nfx[NLOOP_MAX_OPS * 3];
    int nnfx = 0;
    struct sljit_jump *oob_jumps[NLOOP_MAX_OPS];
    int noob = 0;

    NLoopVal vstk[NLOOP_MAX_STK];
    int sp = 0;

    for (int k = i_head; k <= i_end; k++) {
        const uint8_t *kp = bc + offs[k];
        int op = *kp, v, chk;
        int32_t imm;
        nlabels[k - i_head] = sljit_emit_label(c);

        if (jit_decode_get_loc(kp, &v, &chk)) {
            int li = nloop_find_loc(locs, &nlocs, v);
            vstk[sp].kind = (locs[li].type == NLOC_ARR) ? NVK_LOC_ARR
                          : (locs[li].type == NLOC_F64) ? NVK_LOC_F64 : NVK_LOC_I32;
            vstk[sp].loc = li;
            sp++;
        } else if (jit_decode_put_loc(kp, &v, &chk) ||
                   op == OP_set_loc0 || op == OP_set_loc1 || op == OP_set_loc2 ||
                   op == OP_set_loc3 || op == OP_set_loc8 || op == OP_set_loc) {
            int is_set = (op >= OP_set_loc0 && op <= OP_set_loc3) || op == OP_set_loc8 || op == OP_set_loc;
            if (is_set) {
                v = (op == OP_set_loc) ? (int)get_u16(kp + 1)
                  : (op == OP_set_loc8) ? (int)kp[1] : (op - OP_set_loc0);
            }
            int li = nloop_find_loc(locs, &nlocs, v);
            NLoopVal *top = &vstk[sp - 1];
            if (locs[li].type == NLOC_I32) {
                if (locs[li].reg >= 0) {
                    nloop_to_ireg(c, locs, top, locs[li].reg);
                } else if (top->kind == NVK_IMM_I32) {
                    sljit_emit_op1(c, SLJIT_MOV32, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + li * 8, SLJIT_IMM, top->imm);
                } else {
                    nloop_to_ireg(c, locs, top, SLJIT_R0);
                    sljit_emit_op1(c, SLJIT_MOV32, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + li * 8, SLJIT_R0, 0);
                }
            } else {
                nloop_to_freg(c, locs, top, SLJIT_FR0);
                sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + li * 8, SLJIT_FR0, 0);
            }
            if (!is_set) sp--;
        } else if (nloop_decode_push_imm(kp, &imm)) {
            vstk[sp].kind = NVK_IMM_I32; vstk[sp].imm = imm; sp++;
        } else if (op == OP_inc_loc || op == OP_dec_loc) {
            int li = nloop_find_loc(locs, &nlocs, kp[1]);
            int delta = (op == OP_inc_loc) ? 1 : -1;
            if (locs[li].reg >= 0) {
                sljit_emit_op2(c, SLJIT_ADD | SLJIT_32, locs[li].reg, 0, locs[li].reg, 0, SLJIT_IMM, delta);
            } else {
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + li * 8);
                sljit_emit_op2(c, SLJIT_ADD, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_IMM, delta);
                sljit_emit_op1(c, SLJIT_MOV32, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + li * 8, SLJIT_R0, 0);
            }
        } else if (op == OP_add_loc) {
            int li = nloop_find_loc(locs, &nlocs, kp[1]);
            NLoopVal *top = &vstk[--sp];
            if (locs[li].type == NLOC_I32) {
                if (locs[li].reg >= 0) {
                    if (top->kind == NVK_IMM_I32) {
                        sljit_emit_op2(c, SLJIT_ADD | SLJIT_32, locs[li].reg, 0, locs[li].reg, 0, SLJIT_IMM, top->imm);
                    } else if (top->kind == NVK_LOC_I32 && locs[top->loc].reg >= 0) {
                        sljit_emit_op2(c, SLJIT_ADD | SLJIT_32, locs[li].reg, 0, locs[li].reg, 0, locs[top->loc].reg, 0);
                    } else {
                        nloop_to_ireg(c, locs, top, SLJIT_R1);
                        sljit_emit_op2(c, SLJIT_ADD | SLJIT_32, locs[li].reg, 0, locs[li].reg, 0, SLJIT_R1, 0);
                    }
                } else {
                    nloop_to_ireg(c, locs, top, SLJIT_R1);
                    sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + li * 8);
                    sljit_emit_op2(c, SLJIT_ADD, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_R1, 0);
                    sljit_emit_op1(c, SLJIT_MOV32, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + li * 8, SLJIT_R0, 0);
                }
            } else {
                nloop_to_freg(c, locs, top, SLJIT_FR1);
                sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_FR0, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + li * 8);
                sljit_emit_fop2(c, SLJIT_ADD_F64, SLJIT_FR0, 0, SLJIT_FR0, 0, SLJIT_FR1, 0);
                sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_MEM1(SLJIT_SP), NLOOP_LOC_OFF + li * 8, SLJIT_FR0, 0);
            }
        } else if (op == OP_post_inc || op == OP_post_dec) {
            int delta = (op == OP_post_inc) ? 1 : -1;
            sljit_s32 r = nloop_alloc_ireg(vstk, sp);
            if (r < 0) r = SLJIT_R3;
            nloop_to_ireg(c, locs, &vstk[sp - 1], r);
            sljit_emit_op2(c, SLJIT_ADD | SLJIT_32, r, 0, r, 0, SLJIT_IMM, delta);
            vstk[sp].kind = NVK_REG_I32; vstk[sp].reg = r; sp++;
        } else if (op == OP_add || op == OP_sub || op == OP_mul) {
            NLoopVal *b = &vstk[sp - 1], *a = &vstk[sp - 2];
            int is_f = (a->kind == NVK_REG_F64 || a->kind == NVK_LOC_F64 ||
                        b->kind == NVK_REG_F64 || b->kind == NVK_LOC_F64);
            if (is_f) {
                sljit_s32 fop = (op == OP_add) ? SLJIT_ADD_F64 : (op == OP_sub) ? SLJIT_SUB_F64 : SLJIT_MUL_F64;
                sljit_s32 fa = (a->kind == NVK_REG_F64) ? a->reg : nloop_alloc_freg(vstk, sp);
                if (fa < 0) fa = SLJIT_FR2;
                nloop_to_freg(c, locs, a, fa);
                sljit_s32 fb = (b->kind == NVK_REG_F64) ? b->reg : SLJIT_FR3;
                nloop_to_freg(c, locs, b, fb);
                sljit_emit_fop2(c, fop, fa, 0, fa, 0, fb, 0);
                sp--;
                vstk[sp - 1].kind = NVK_REG_F64; vstk[sp - 1].reg = fa;
            } else {
                sljit_s32 aop = (op == OP_add) ? SLJIT_ADD : (op == OP_sub) ? SLJIT_SUB : SLJIT_MUL;
                sljit_s32 ra = (a->kind == NVK_REG_I32) ? a->reg : nloop_alloc_ireg(vstk, sp);
                if (ra < 0) ra = SLJIT_R2;
                nloop_to_ireg(c, locs, a, ra);
                if (b->kind == NVK_IMM_I32) {
                    sljit_emit_op2(c, aop | SLJIT_32, ra, 0, ra, 0, SLJIT_IMM, b->imm);
                } else if (b->kind == NVK_LOC_I32 && locs[b->loc].reg >= 0) {
                    sljit_emit_op2(c, aop | SLJIT_32, ra, 0, ra, 0, locs[b->loc].reg, 0);
                } else {
                    sljit_s32 rb = (b->kind == NVK_REG_I32) ? b->reg : SLJIT_R3;
                    nloop_to_ireg(c, locs, b, rb);
                    sljit_emit_op2(c, aop | SLJIT_32, ra, 0, ra, 0, rb, 0);
                }
                sp--;
                vstk[sp - 1].kind = NVK_REG_I32; vstk[sp - 1].reg = ra;
            }
        } else if (op == OP_lt || op == OP_lte || op == OP_gt || op == OP_gte ||
                   op == OP_eq || op == OP_neq || op == OP_strict_eq || op == OP_strict_neq) {
            NLoopVal *b = &vstk[sp - 1], *a = &vstk[sp - 2];
            sljit_s32 ct;
            switch (op) {
            case OP_lt:  ct = SLJIT_SIG_LESS; break;
            case OP_lte: ct = SLJIT_SIG_LESS_EQUAL; break;
            case OP_gt:  ct = SLJIT_SIG_GREATER; break;
            case OP_gte: ct = SLJIT_SIG_GREATER_EQUAL; break;
            case OP_eq: case OP_strict_eq: ct = SLJIT_EQUAL; break;
            default:     ct = SLJIT_NOT_EQUAL; break;
            }
            /* Fast fuse cmp + if_true / if_false */
            if (k + 1 <= i_end) {
                const uint8_t *np = bc + offs[k + 1];
                int is_t, tgt;
                if (jit_decode_branch(np, offs[k + 1], &is_t, &tgt)) {
                    sljit_s32 bcond = is_t ? ct : jit_invert_cond(ct);
                    sljit_s32 ra = (a->kind == NVK_LOC_I32 && locs[a->loc].reg >= 0) ? locs[a->loc].reg : SLJIT_R0;
                    if (ra == SLJIT_R0) nloop_to_ireg(c, locs, a, SLJIT_R0);
                    if (b->kind == NVK_IMM_I32) {
                        struct sljit_jump *jb = sljit_emit_cmp(c, bcond | SLJIT_32, ra, 0, SLJIT_IMM, b->imm);
                        nfx[nnfx].jmp = jb; nfx[nnfx].target_off = tgt; nnfx++;
                    } else if (b->kind == NVK_LOC_I32 && locs[b->loc].reg >= 0) {
                        struct sljit_jump *jb = sljit_emit_cmp(c, bcond | SLJIT_32, ra, 0, locs[b->loc].reg, 0);
                        nfx[nnfx].jmp = jb; nfx[nnfx].target_off = tgt; nnfx++;
                    } else {
                        nloop_to_ireg(c, locs, b, SLJIT_R1);
                        struct sljit_jump *jb = sljit_emit_cmp(c, bcond | SLJIT_32, ra, 0, SLJIT_R1, 0);
                        nfx[nnfx].jmp = jb; nfx[nnfx].target_off = tgt; nnfx++;
                    }
                    sp -= 2;
                    k++; /* skip branch */
                    continue;
                }
            }
            sljit_s32 rd = nloop_alloc_ireg(vstk, sp);
            if (rd < 0) rd = SLJIT_R0;
            nloop_to_ireg(c, locs, a, SLJIT_R0);
            nloop_to_ireg(c, locs, b, SLJIT_R1);
            sljit_emit_op2u(c, SLJIT_SUB | SLJIT_32 | SLJIT_SET_Z | SLJIT_SET_SIG_LESS | SLJIT_SET_SIG_GREATER,
                            SLJIT_R0, 0, SLJIT_R1, 0);
            sljit_emit_op_flags(c, SLJIT_MOV, rd, 0, ct);
            sp--;
            vstk[sp - 1].kind = NVK_REG_I32; vstk[sp - 1].reg = rd;
        } else if (op == OP_lnot) {
            NLoopVal *a = &vstk[sp - 1];
            if (k + 1 <= i_end) {
                const uint8_t *np = bc + offs[k + 1];
                int is_t, tgt;
                if (jit_decode_branch(np, offs[k + 1], &is_t, &tgt)) {
                    nloop_to_ireg(c, locs, a, SLJIT_R0);
                    struct sljit_jump *jb = sljit_emit_cmp(c, is_t ? SLJIT_EQUAL : SLJIT_NOT_EQUAL,
                                                           SLJIT_R0, 0, SLJIT_IMM, 0);
                    nfx[nnfx].jmp = jb; nfx[nnfx].target_off = tgt; nnfx++;
                    sp--;
                    k++;
                    continue;
                }
            }
            sljit_s32 rd = nloop_alloc_ireg(vstk, sp);
            if (rd < 0) rd = SLJIT_R0;
            nloop_to_ireg(c, locs, a, SLJIT_R0);
            sljit_emit_op2u(c, SLJIT_SUB | SLJIT_32 | SLJIT_SET_Z, SLJIT_R0, 0, SLJIT_IMM, 0);
            sljit_emit_op_flags(c, SLJIT_MOV, rd, 0, SLJIT_EQUAL);
            vstk[sp - 1].kind = NVK_REG_I32; vstk[sp - 1].reg = rd;
        } else if (op == OP_set_loc_uninitialized || op == OP_to_propkey || op == OP_to_propkey2) {
            /* no-op in unboxed native loop */
        } else if (op == OP_swap) {
            NLoopVal tmp = vstk[sp - 1]; vstk[sp - 1] = vstk[sp - 2]; vstk[sp - 2] = tmp;
        } else if (op == OP_dup) {
            if (k + 2 <= i_end && bc[offs[k + 1]] == OP_is_undefined_or_null) {
                int is_t, tgt;
                if (jit_decode_branch(bc + offs[k + 2], offs[k + 2], &is_t, &tgt)) {
                    if (!is_t) {
                        int ti = jit_find_label_idx(offs, nops, tgt);
                        if (ti > k + 2 && ti <= i_end) { k = ti - 1; continue; }
                    } else {
                        k += 2; continue;
                    }
                }
            }
            vstk[sp] = vstk[sp - 1]; sp++;
        } else if (op == OP_dup2) {
            if (vstk[sp - 1].kind == NVK_REG_I32) {
                sljit_emit_op1(c, SLJIT_MOV32, SLJIT_MEM1(SLJIT_SP), NLOOP_TMP_OFF + 8, vstk[sp - 1].reg, 0);
                vstk[sp - 1].kind = NVK_TMP_I32;
                vstk[sp - 1].imm = NLOOP_TMP_OFF + 8;
            }
            vstk[sp] = vstk[sp - 2];
            vstk[sp + 1] = vstk[sp - 1];
            sp += 2;
        } else if (op == OP_is_undefined_or_null) {
            if (k + 1 <= i_end) {
                int is_t, tgt;
                if (jit_decode_branch(bc + offs[k + 1], offs[k + 1], &is_t, &tgt) && is_t) {
                    k++;
                    continue;
                }
            }
            vstk[sp - 1].kind = NVK_IMM_I32;
            vstk[sp - 1].imm = 0;
        } else if (op == OP_drop) {
            sp--;
        } else if (op == OP_get_array_el || op == OP_get_array_el2) {
            int keep_obj = (op == OP_get_array_el2);
            int ali = vstk[sp - 2].loc;
            int ai = locs[ali].arr_idx;
            int is_lnot_next = (k + 1 <= i_end && bc[offs[k + 1]] == OP_lnot);

            nloop_to_ireg(c, locs, &vstk[sp - 1], SLJIT_R0);
            sljit_emit_op1(c, SLJIT_MOV_U32, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_ARR_OFF + ai * 16 + 8);
            oob_jumps[noob++] = sljit_emit_cmp(c, SLJIT_GREATER_EQUAL | SLJIT_32, SLJIT_R0, 0, SLJIT_R1, 0);

            if (keep_obj) {
                /* Save index in NLOOP_TMP_OFF for upcoming put_array_el */
                sljit_emit_op1(c, SLJIT_MOV32, SLJIT_MEM1(SLJIT_SP), NLOOP_TMP_OFF, SLJIT_R0, 0);
            }

            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_ARR_OFF + ai * 16 + 0);
            sljit_emit_op1(c, SLJIT_MOV_U32, SLJIT_R2, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_ARR_OFF + ai * 16 + 12);

            if (is_lnot_next) {
                sljit_s32 rd = nloop_alloc_ireg(vstk, sp - 2);
                if (rd < 0) rd = SLJIT_R0;
                struct sljit_jump *jf64 = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R2, 0, SLJIT_IMM, 0);
                sljit_emit_op1(c, SLJIT_MOV_U8, rd, 0, SLJIT_MEM2(SLJIT_R1, SLJIT_R0), 0);
                struct sljit_jump *jdone = sljit_emit_jump(c, SLJIT_JUMP);
                sljit_set_label(jf64, sljit_emit_label(c));
                sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_FR0, 0, SLJIT_MEM2(SLJIT_R1, SLJIT_R0), 3);
                sljit_emit_fop1(c, SLJIT_CONV_SW_FROM_F64, rd, 0, SLJIT_FR0, 0); /* truthy check for non-zero */
                sljit_set_label(jdone, sljit_emit_label(c));
                if (!keep_obj) sp--; else { vstk[sp - 2].kind = NVK_ARR_EL2; sp++; }
                vstk[sp - 1].kind = NVK_REG_I32; vstk[sp - 1].reg = rd;
            } else {
                sljit_s32 fd = nloop_alloc_freg(vstk, sp);
                if (fd < 0) fd = SLJIT_FR0;
                struct sljit_jump *ju8 = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R2, 0, SLJIT_IMM, 0);
                sljit_emit_fop1(c, SLJIT_MOV_F64, fd, 0, SLJIT_MEM2(SLJIT_R1, SLJIT_R0), 3);
                struct sljit_jump *jdone = sljit_emit_jump(c, SLJIT_JUMP);
                sljit_set_label(ju8, sljit_emit_label(c));
                sljit_emit_op1(c, SLJIT_MOV_U8, SLJIT_R0, 0, SLJIT_MEM2(SLJIT_R1, SLJIT_R0), 0);
                sljit_emit_fop1(c, SLJIT_CONV_F64_FROM_SW, fd, 0, SLJIT_R0, 0);
                sljit_set_label(jdone, sljit_emit_label(c));
                if (!keep_obj) {
                    sp--;
                    vstk[sp - 1].kind = NVK_REG_F64; vstk[sp - 1].reg = fd;
                } else {
                    vstk[sp - 2].kind = NVK_ARR_EL2;
                    vstk[sp].kind = NVK_REG_F64; vstk[sp].reg = fd;
                    sp++;
                }
            }
        } else if (op == OP_put_array_el) {
            int ali = vstk[sp - 3].loc;
            int ai = locs[ali].arr_idx;
            NLoopVal *val = &vstk[sp - 1];
            struct sljit_jump *j_skip;
            if (vstk[sp - 3].kind == NVK_ARR_EL2) {
                sljit_emit_op1(c, SLJIT_MOV_U32, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_TMP_OFF);
                j_skip = NULL; /* already bounds-checked in get_array_el2 */
            } else {
                nloop_to_ireg(c, locs, &vstk[sp - 2], SLJIT_R0);
                sljit_emit_op1(c, SLJIT_MOV_U32, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_ARR_OFF + ai * 16 + 8);
                j_skip = sljit_emit_cmp(c, SLJIT_GREATER_EQUAL | SLJIT_32, SLJIT_R0, 0, SLJIT_R1, 0);
            }
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_ARR_OFF + ai * 16 + 0);
            sljit_emit_op1(c, SLJIT_MOV_U32, SLJIT_R2, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_ARR_OFF + ai * 16 + 12);
            if (val->kind == NVK_IMM_I32 || val->kind == NVK_LOC_I32 || val->kind == NVK_REG_I32) {
                struct sljit_jump *jf64 = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R2, 0, SLJIT_IMM, 0);
                nloop_to_ireg(c, locs, val, SLJIT_R2);
                sljit_emit_op1(c, SLJIT_MOV_U8, SLJIT_MEM2(SLJIT_R1, SLJIT_R0), 0, SLJIT_R2, 0);
                struct sljit_jump *jd = sljit_emit_jump(c, SLJIT_JUMP);
                sljit_set_label(jf64, sljit_emit_label(c));
                nloop_to_freg(c, locs, val, SLJIT_FR0);
                sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_MEM2(SLJIT_R1, SLJIT_R0), 3, SLJIT_FR0, 0);
                sljit_set_label(jd, sljit_emit_label(c));
            } else {
                struct sljit_jump *ju8 = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R2, 0, SLJIT_IMM, 0);
                nloop_to_freg(c, locs, val, SLJIT_FR0);
                sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_MEM2(SLJIT_R1, SLJIT_R0), 3, SLJIT_FR0, 0);
                struct sljit_jump *jd = sljit_emit_jump(c, SLJIT_JUMP);
                sljit_set_label(ju8, sljit_emit_label(c));
                nloop_to_freg(c, locs, val, SLJIT_FR0);
                sljit_emit_fop1(c, SLJIT_CONV_SW_FROM_F64, SLJIT_R2, 0, SLJIT_FR0, 0);
                sljit_emit_op1(c, SLJIT_MOV_U8, SLJIT_MEM2(SLJIT_R1, SLJIT_R0), 0, SLJIT_R2, 0);
                sljit_set_label(jd, sljit_emit_label(c));
            }
            if (j_skip) sljit_set_label(j_skip, sljit_emit_label(c));
            sp -= 3;
        } else if (op == OP_goto || op == OP_goto8 || op == OP_goto16) {
            int rel = (op == OP_goto) ? (int)(int32_t)get_u32(kp + 1)
                    : (op == OP_goto16) ? (int)(int16_t)get_u16(kp + 1)
                    : (int)(int8_t)kp[1];
            int tgt = offs[k] + 1 + rel;
            struct sljit_jump *jg = sljit_emit_jump(c, SLJIT_JUMP);
            nfx[nnfx].jmp = jg; nfx[nnfx].target_off = tgt; nnfx++;
        } else if (op == OP_if_true || op == OP_if_true8 || op == OP_if_false || op == OP_if_false8) {
            int is_t, tgt;
            jit_decode_branch(kp, offs[k], &is_t, &tgt);
            nloop_to_ireg(c, locs, &vstk[--sp], SLJIT_R0);
            struct sljit_jump *jb = sljit_emit_cmp(c, is_t ? SLJIT_NOT_EQUAL : SLJIT_EQUAL,
                                                   SLJIT_R0, 0, SLJIT_IMM, 0);
            nfx[nnfx].jmp = jb; nfx[nnfx].target_off = tgt; nnfx++;
        }
    }

    /* Fallthrough at end of loop region -> exit to end_off */
    nloop_emit_writeback(c, locs, nlocs);
    {
        struct sljit_jump *j_end = sljit_emit_jump(c, SLJIT_JUMP);
        fx[*pnfx].jmp = j_end; fx[*pnfx].kind = JIT_FX_BRANCH; fx[*pnfx].target_off = end_off; (*pnfx)++;
    }

    /* Resolve internal native-loop jumps and external exit jumps */
    for (int j = 0; j < nnfx; j++) {
        int tgt = nfx[j].target_off;
        if (tgt >= head_off && tgt < end_off) {
            int ti = jit_find_label_idx(offs + i_head, i_end - i_head + 1, tgt);
            if (ti >= 0) {
                sljit_set_label(nfx[j].jmp, nlabels[ti]);
                continue;
            }
        }
        /* External exit: emit writeback stub then branch to Tier 2 target */
        sljit_set_label(nfx[j].jmp, sljit_emit_label(c));
        nloop_emit_writeback(c, locs, nlocs);
        struct sljit_jump *je = sljit_emit_jump(c, SLJIT_JUMP);
        fx[*pnfx].jmp = je; fx[*pnfx].kind = JIT_FX_BRANCH; fx[*pnfx].target_off = tgt; (*pnfx)++;
    }

    /* OOB read bail: writeback + bail at head_off (or cur) */
    if (noob > 0) {
        struct sljit_label *l_oob = sljit_emit_label(c);
        for (int j = 0; j < noob; j++)
            sljit_set_label(oob_jumps[j], l_oob);
        nloop_emit_writeback(c, locs, nlocs);
        sljit_emit_const(c, SLJIT_MOV, JREG_PC, 0, (sljit_sw)(bc + head_off));
        struct sljit_jump *jb = sljit_emit_jump(c, SLJIT_JUMP);
        fx[*pnfx].jmp = jb; fx[*pnfx].kind = JIT_FX_BAIL; (*pnfx)++;
    }

    /* Guard failures -> fall through to normal Tier 2 code for [i_head .. i_end] */
    {
        struct sljit_label *l_t2 = sljit_emit_label(c);
        for (int j = 0; j < ngf; j++)
            sljit_set_label(gf[j], l_t2);
    }

    *i_end_out = i_end;
    return 1;
}

static int jit_codegen(JSContext *ctx, JSFunctionBytecode *b,
                       struct JSJITEntry *e)
{
    struct sljit_compiler *c;
    struct sljit_label **labels;
    struct sljit_label *bail_label, *done_label, *exc_label;
    struct jit_fixup *fx;
    struct sljit_jump *j;
    int nfx = 0, nops = 0, i, nic = 0, ic_idx = 0;
    int *offs;
    const uint8_t *p, *end;
    uint8_t *bc = b->byte_code_buf;
    uint8_t *is_target = NULL;
    JitFieldIC *ics = NULL;
    void *code;

    if (!jit_prescan(b, &nic))
        return 0;
    if (b->byte_code_len <= 0)
        return 0;

    if (nic > 0) {
        ics = (JitFieldIC *)js_mallocz_rt(ctx->rt, (size_t)nic * sizeof(JitFieldIC));
        if (!ics)
            return 0;
    }

    /* count opcodes + record offsets */
    offs = js_mallocz(ctx, sizeof(int) * (b->byte_code_len + 1));
    if (!offs) {
        js_free_rt(ctx->rt, ics);
        return 0;
    }
    p = bc; end = bc + b->byte_code_len;
    while (p < end) {
        int op = *p;
        offs[nops++] = (int)(p - bc);
        p += short_opcode_info(op).size;
    }

    labels = js_mallocz(ctx, sizeof(struct sljit_label *) * (nops + 1));
    fx = js_mallocz(ctx, sizeof(struct jit_fixup) * (nops * 6 + 16));
    is_target = js_mallocz(ctx, (size_t)nops + 1);
    if (!labels || !fx || !is_target) {
        js_free_rt(ctx->rt, ics);
        js_free(ctx, offs);
        js_free(ctx, labels);
        js_free(ctx, fx);
        js_free(ctx, is_target);
        return 0;
    }
    for (int k = 0; k < nops; k++) {
        const uint8_t *kp = bc + offs[k];
        int kop = *kp, rel = 0, tgt = -1;
        if (kop == OP_goto || kop == OP_if_true || kop == OP_if_false) {
            rel = (int)(int32_t)get_u32(kp + 1);
            tgt = offs[k] + 1 + rel;
        } else if (kop == OP_goto16) {
            rel = (int)(int16_t)get_u16(kp + 1);
            tgt = offs[k] + 1 + rel;
        } else if (kop == OP_goto8 || kop == OP_if_true8 || kop == OP_if_false8) {
            rel = (int)(int8_t)kp[1];
            tgt = offs[k] + 1 + rel;
        }
        if (tgt >= 0) {
            int ti = jit_find_label_idx(offs, nops, tgt);
            if (ti >= 0)
                is_target[ti] = 1;
        }
    }

    c = sljit_create_compiler(NULL);
    if (!c) {
        js_free_rt(ctx->rt, ics);
        js_free(ctx, offs); js_free(ctx, labels); js_free(ctx, fx); js_free(ctx, is_target);
        return 0;
    }

    /* entry: 1 pointer arg (JitCtx*) -> S0; sp/pc/var/arg cached in S1..S4, S5/S6 for nloop */
    sljit_emit_enter(c, 0, SLJIT_ARGS1(W, P), 4 | SLJIT_ENTER_FLOAT(4), 7, NLOOP_FRAME_SZ);
    sljit_emit_op1(c, SLJIT_MOV, JREG_SP, 0, SLJIT_MEM1(JREG_JCTX), JCTX_SP);
    sljit_emit_op1(c, SLJIT_MOV, JREG_VAR, 0, SLJIT_MEM1(JREG_JCTX), JCTX_VARBUF);
    sljit_emit_op1(c, SLJIT_MOV, JREG_ARG, 0, SLJIT_MEM1(JREG_JCTX), JCTX_ARGBUF);

    /* emit pass */
    int nloop_skip_until = -1;
    int nloop_hits = 0;
    p = bc; i = 0;
    while (p < end) {
        int op = *p;
        int ooff = (int)(p - bc);
        sljit_sw slot;        /* var/arg base reg */
        int idx = 0;

        labels[i] = sljit_emit_label(c);
        if (i > nloop_skip_until) {
            int nloop_end = -1;
            if (jit_try_emit_nloop(c, bc, offs, nops, i, fx, &nfx, &nloop_end)) {
                nloop_skip_until = nloop_end;
                nloop_hits++;
            }
        }

        /* ---- Peephole Superinstructions (when interior ops are not branch targets) ---- */
        {
            int v1, chk1, v2, chk2, v3, chk3;
            sljit_s32 ccond;
            int is_true, target;
            if (jit_decode_get_loc(p, &v1, &chk1)) {
                const uint8_t *p1 = p + short_opcode_info(op).size;
                /* Pattern 1: get_loc(v1) + get_loc(v2) + cmp + if_false/if_true */
                if (i + 3 < nops && !is_target[i + 1] && !is_target[i + 2] && !is_target[i + 3] &&
                    jit_decode_get_loc(p1, &v2, &chk2)) {
                    const uint8_t *p2 = p1 + short_opcode_info(*p1).size;
                    const uint8_t *p3 = p2 + short_opcode_info(*p2).size;
                    if (jit_decode_cmp(*p2, &ccond) &&
                        jit_decode_branch(p3, (int)(p3 - bc), &is_true, &target)) {
                        sljit_s32 bcond = is_true ? ccond : jit_invert_cond(ccond);
                        struct sljit_jump *j_slow, *j_br, *j_done;
                        sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_VAR), v1 * 16 + JSV_TAG);
                        sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_VAR), v2 * 16 + JSV_TAG);
                        sljit_emit_op2(c, SLJIT_OR, SLJIT_R2, 0, SLJIT_R0, 0, SLJIT_R1, 0);
                        j_slow = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R2, 0, SLJIT_IMM, 0);
                        sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_MEM1(JREG_VAR), v1 * 16 + JSV_U);
                        sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_MEM1(JREG_VAR), v2 * 16 + JSV_U);
                        j_br = sljit_emit_cmp(c, bcond, SLJIT_R0, 0, SLJIT_R1, 0);
                        fx[nfx].jmp = j_br; fx[nfx].kind = JIT_FX_BRANCH; fx[nfx].target_off = target; nfx++;
                        j_done = sljit_emit_jump(c, SLJIT_JUMP);
                        sljit_set_label(j_slow, sljit_emit_label(c));
                        jit_emit_sync_call(c, (void *)&jit_helper_cmp_locs, 3,
                                           (sljit_sw)(v1 | (chk1 << 16)),
                                           (sljit_sw)(v2 | (chk2 << 16)),
                                           (sljit_sw)(*p2), 1, fx, &nfx);
                        j_br = sljit_emit_cmp(c, is_true ? SLJIT_NOT_EQUAL : SLJIT_EQUAL,
                                              SLJIT_RETURN_REG, 0, SLJIT_IMM, 0);
                        fx[nfx].jmp = j_br; fx[nfx].kind = JIT_FX_BRANCH; fx[nfx].target_off = target; nfx++;
                        sljit_set_label(j_done, sljit_emit_label(c));
                        p = p3 + short_opcode_info(*p3).size;
                        i += 4;
                        continue;
                    }
                    /* Pattern 2: get_loc(arr) + get_loc(idx) + get_array_el + lnot + if_false/if_true */
                    if (i + 4 < nops && !is_target[i + 4] && *p2 == OP_get_array_el &&
                        *p3 == OP_lnot) {
                        const uint8_t *p4 = p3 + short_opcode_info(*p3).size;
                        if (jit_decode_branch(p4, (int)(p4 - bc), &is_true, &target)) {
                            struct sljit_jump *j_no, *j_ni, *j_oob, *j_nu8, *j_br, *j_done;
                            struct sljit_label *l_slow;
                            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_VAR), v1 * 16 + JSV_TAG);
                            j_no = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_OBJECT);
                            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_VAR), v2 * 16 + JSV_TAG);
                            j_ni = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_INT);
                            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R2, 0, SLJIT_MEM1(JREG_VAR), v1 * 16 + JSV_U);
                            sljit_emit_op1(c, SLJIT_MOV_U32, SLJIT_R0, 0, SLJIT_MEM1(JREG_VAR), v2 * 16 + JSV_U);
                            sljit_emit_op1(c, SLJIT_MOV_U32, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R2),
                                           (sljit_sw)offsetof(JSObject, u.array.count));
                            j_oob = sljit_emit_cmp(c, SLJIT_GREATER_EQUAL, SLJIT_R0, 0, SLJIT_R1, 0);
                            sljit_emit_op1(c, SLJIT_MOV_U16, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R2),
                                           (sljit_sw)offsetof(JSObject, class_id));
                            j_nu8 = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_CLASS_UINT8_ARRAY);
                            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R2),
                                           (sljit_sw)offsetof(JSObject, u.array.u.ptr));
                            sljit_emit_op1(c, SLJIT_MOV_U8, SLJIT_R0, 0, SLJIT_MEM2(SLJIT_R1, SLJIT_R0), 0);
                            /* lnot: val == 0 -> true(1); if_false branches when val != 0; if_true branches when val == 0 */
                            j_br = sljit_emit_cmp(c, is_true ? SLJIT_EQUAL : SLJIT_NOT_EQUAL,
                                                  SLJIT_R0, 0, SLJIT_IMM, 0);
                            fx[nfx].jmp = j_br; fx[nfx].kind = JIT_FX_BRANCH; fx[nfx].target_off = target; nfx++;
                            j_done = sljit_emit_jump(c, SLJIT_JUMP);
                            l_slow = sljit_emit_label(c);
                            sljit_set_label(j_no, l_slow);
                            sljit_set_label(j_ni, l_slow);
                            sljit_set_label(j_oob, l_slow);
                            sljit_set_label(j_nu8, l_slow);
                            jit_emit_sync_call(c, (void *)&jit_helper_get_array_el_locs, 3,
                                               (sljit_sw)(v1 | (chk1 << 16)),
                                               (sljit_sw)(v2 | (chk2 << 16)),
                                               1, 1, fx, &nfx);
                            j_br = sljit_emit_cmp(c, is_true ? SLJIT_NOT_EQUAL : SLJIT_EQUAL,
                                                  SLJIT_RETURN_REG, 0, SLJIT_IMM, 0);
                            fx[nfx].jmp = j_br; fx[nfx].kind = JIT_FX_BRANCH; fx[nfx].target_off = target; nfx++;
                            sljit_set_label(j_done, sljit_emit_label(c));
                            p = p4 + short_opcode_info(*p4).size;
                            i += 5;
                            continue;
                        }
                    }
                }
                /* Pattern 3: get_loc(v1) + get_loc(v2) + (add|sub|mul) */
                if (i + 2 < nops && !is_target[i + 1] && !is_target[i + 2] &&
                    jit_decode_get_loc(p1, &v2, &chk2)) {
                    const uint8_t *p2 = p1 + short_opcode_info(*p1).size;
                    int bop = *p2;
                    if (bop == OP_add || bop == OP_sub || bop == OP_mul) {
                        sljit_s32 aop = (bop == OP_add) ? SLJIT_ADD : (bop == OP_sub) ? SLJIT_SUB : SLJIT_MUL;
                        struct sljit_jump *j_slow, *j_ovf, *j_negz = NULL, *j_done;
                        struct sljit_label *l_slow;
                        sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_VAR), v1 * 16 + JSV_TAG);
                        sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_VAR), v2 * 16 + JSV_TAG);
                        sljit_emit_op2(c, SLJIT_OR, SLJIT_R2, 0, SLJIT_R0, 0, SLJIT_R1, 0);
                        j_slow = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R2, 0, SLJIT_IMM, 0);
                        sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_MEM1(JREG_VAR), v1 * 16 + JSV_U);
                        sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_MEM1(JREG_VAR), v2 * 16 + JSV_U);
                        if (bop == OP_mul)
                            sljit_emit_op2(c, SLJIT_OR | SLJIT_32, SLJIT_R2, 0, SLJIT_R0, 0, SLJIT_R1, 0);
                        sljit_emit_op2(c, aop, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_R1, 0);
                        sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_R0, 0);
                        j_ovf = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_R1, 0);
                        if (bop == OP_mul) {
                            struct sljit_jump *j_nz = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, 0);
                            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R2, 0, SLJIT_R2, 0);
                            j_negz = sljit_emit_cmp(c, SLJIT_SIG_LESS, SLJIT_R2, 0, SLJIT_IMM, 0);
                            sljit_set_label(j_nz, sljit_emit_label(c));
                        }
                        sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_U, SLJIT_R0, 0);
                        sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_TAG, SLJIT_IMM, 0);
                        sljit_emit_op2(c, SLJIT_ADD, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
                        j_done = sljit_emit_jump(c, SLJIT_JUMP);
                        l_slow = sljit_emit_label(c);
                        sljit_set_label(j_slow, l_slow);
                        sljit_set_label(j_ovf, l_slow);
                        if (j_negz) sljit_set_label(j_negz, l_slow);
                        jit_emit_sync_call(c, (void *)&jit_helper_binop_locs, 3,
                                           (sljit_sw)(v1 | (chk1 << 16)),
                                           (sljit_sw)(v2 | (chk2 << 16)),
                                           (sljit_sw)bop, 1, fx, &nfx);
                        sljit_set_label(j_done, sljit_emit_label(c));
                        p = p2 + short_opcode_info(bop).size;
                        i += 3;
                        continue;
                    }
                }
                /* Pattern 4: get_loc(v1) + (post_inc|post_dec) + put_loc(v1) + drop */
                if (i + 3 < nops && !is_target[i + 1] && !is_target[i + 2] && !is_target[i + 3] &&
                    (*p1 == OP_post_inc || *p1 == OP_post_dec)) {
                    const uint8_t *p2 = p1 + short_opcode_info(*p1).size;
                    const uint8_t *p3 = p2 + short_opcode_info(*p2).size;
                    if (jit_decode_put_loc(p2, &v3, &chk3) && v3 == v1 && *p3 == OP_drop) {
                        int delta = (*p1 == OP_post_inc) ? 1 : -1;
                        int sop = (*p1 == OP_post_inc) ? OP_inc_loc : OP_dec_loc;
                        struct sljit_jump *j_slow, *j_ovf, *j_done;
                        struct sljit_label *l_slow;
                        sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_VAR), v1 * 16 + JSV_TAG);
                        j_slow = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, 0);
                        sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_MEM1(JREG_VAR), v1 * 16 + JSV_U);
                        sljit_emit_op2(c, SLJIT_ADD, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_IMM, delta);
                        sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_R0, 0);
                        j_ovf = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_R1, 0);
                        sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_VAR), v1 * 16 + JSV_U, SLJIT_R0, 0);
                        j_done = sljit_emit_jump(c, SLJIT_JUMP);
                        l_slow = sljit_emit_label(c);
                        sljit_set_label(j_slow, l_slow);
                        sljit_set_label(j_ovf, l_slow);
                        jit_emit_sync_call(c, (void *)&jit_helper_inc_dec_loc_chk, 2,
                                           (sljit_sw)v1, (sljit_sw)(sop | ((chk1 | chk3) << 16)),
                                           0, 1, fx, &nfx);
                        sljit_set_label(j_done, sljit_emit_label(c));
                        p = p3 + short_opcode_info(*p3).size;
                        i += 4;
                        continue;
                    }
                }
            }
        }

        switch (op) {
        /* ---- push immediate ints ---- */
        case OP_push_0: case OP_push_1: case OP_push_2: case OP_push_3:
        case OP_push_4: case OP_push_5: case OP_push_6: case OP_push_7:
            slot = (sljit_sw)(op - OP_push_0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_U, SLJIT_IMM, slot);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_TAG, SLJIT_IMM, 0);
            sljit_emit_op2(c, SLJIT_ADD, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            break;
        case OP_push_minus1:
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_U, SLJIT_IMM, (sljit_sw)-1);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_TAG, SLJIT_IMM, 0);
            sljit_emit_op2(c, SLJIT_ADD, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            break;
        case OP_push_i8:
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_U, SLJIT_IMM, (sljit_sw)get_i8(p + 1));
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_TAG, SLJIT_IMM, 0);
            sljit_emit_op2(c, SLJIT_ADD, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            break;
        case OP_push_i16:
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_U, SLJIT_IMM, (sljit_sw)get_i16(p + 1));
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_TAG, SLJIT_IMM, 0);
            sljit_emit_op2(c, SLJIT_ADD, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            break;
        case OP_push_i32:
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_U, SLJIT_IMM, (sljit_sw)get_i32(p + 1));
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_TAG, SLJIT_IMM, 0);
            sljit_emit_op2(c, SLJIT_ADD, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            break;
        case OP_nop:
            break;
        case OP_push_const8:
        case OP_push_const: {
            int cidx = (op == OP_push_const8) ? (int)p[1] : (int)get_u32(p + 1);
            JSValue cv = b->cpool[cidx];
            int64_t ctag = JS_VALUE_GET_TAG(cv);
            if (ctag >= 0) {
                sljit_sw cu = 0;
                memcpy(&cu, &cv.u, sizeof(cu));
                sljit_emit_const(c, SLJIT_MOV, SLJIT_R0, 0, cu);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_U, SLJIT_R0, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_TAG, SLJIT_IMM, (sljit_sw)ctag);
                sljit_emit_op2(c, SLJIT_ADD, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            } else {
                jit_emit_sync_call(c, (void *)&jit_helper_push_const, 1, (sljit_sw)cidx, 0, 0, 0, fx, &nfx);
            }
            break;
        }
        case OP_push_atom_value: {
            JSAtom atom = (JSAtom)get_u32(p + 1);
            jit_emit_sync_call(c, (void *)&jit_helper_push_atom_val, 1, (sljit_sw)atom, 0, 0, 1, fx, &nfx);
            break;
        }
        case OP_push_empty_string:
            jit_emit_sync_call(c, (void *)&jit_helper_push_empty_string, 0, 0, 0, 0, 0, fx, &nfx);
            break;
        case OP_push_this:
            jit_emit_sync_call(c, (void *)&jit_helper_push_this, 0, 0, 0, 0, 1, fx, &nfx);
            break;
        case OP_get_var_ref0: case OP_get_var_ref1: case OP_get_var_ref2: case OP_get_var_ref3:
        case OP_get_var_ref: case OP_get_var_ref_check: {
            int idx = (op >= OP_get_var_ref0 && op <= OP_get_var_ref3)
                      ? (op - OP_get_var_ref0) : (int)get_u16(p + 1);
            int chk = (op == OP_get_var_ref_check) ? 1 : 0;
            jit_emit_sync_call(c, (void *)&jit_helper_get_var_ref, 2,
                               (sljit_sw)idx, (sljit_sw)chk, 0, chk, fx, &nfx);
            break;
        }
        case OP_put_var_ref0: case OP_put_var_ref1: case OP_put_var_ref2: case OP_put_var_ref3:
        case OP_set_var_ref0: case OP_set_var_ref1: case OP_set_var_ref2: case OP_set_var_ref3:
        case OP_put_var_ref: case OP_set_var_ref:
        case OP_put_var_ref_check: case OP_put_var_ref_check_init: {
            int idx;
            int mode = 0;
            if (op >= OP_put_var_ref0 && op <= OP_put_var_ref3) {
                idx = op - OP_put_var_ref0; mode = 0;
            } else if (op >= OP_set_var_ref0 && op <= OP_set_var_ref3) {
                idx = op - OP_set_var_ref0; mode = 1;
            } else {
                idx = (int)get_u16(p + 1);
                if (op == OP_set_var_ref) mode = 1;
                else if (op == OP_put_var_ref_check) mode = 2;
                else if (op == OP_put_var_ref_check_init) mode = 3;
            }
            jit_emit_sync_call(c, (void *)&jit_helper_put_var_ref, 2,
                               (sljit_sw)idx, (sljit_sw)mode, 0,
                               (mode >= 2) ? 1 : 0, fx, &nfx);
            break;
        }
        /* ---- push trivial immediates ---- */
        case OP_undefined:
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_U, SLJIT_IMM, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_TAG, SLJIT_IMM, JS_TAG_UNDEFINED);
            sljit_emit_op2(c, SLJIT_ADD, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            break;
        case OP_null:
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_U, SLJIT_IMM, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_TAG, SLJIT_IMM, JS_TAG_NULL);
            sljit_emit_op2(c, SLJIT_ADD, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            break;
        case OP_push_false:
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_U, SLJIT_IMM, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_TAG, SLJIT_IMM, JS_TAG_BOOL);
            sljit_emit_op2(c, SLJIT_ADD, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            break;
        case OP_push_true:
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_U, SLJIT_IMM, 1);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_TAG, SLJIT_IMM, JS_TAG_BOOL);
            sljit_emit_op2(c, SLJIT_ADD, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            break;
        /* ---- get_loc / get_arg / get_loc_check : inline for ALL types ---- */
        case OP_set_loc_uninitialized: {
            int lidx = (int)get_u16(p + 1);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_VAR), lidx * 16 + JSV_U, SLJIT_IMM, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_VAR), lidx * 16 + JSV_TAG, SLJIT_IMM, JS_TAG_UNINITIALIZED);
            break;
        }
        case OP_get_loc_check: {
            struct sljit_jump *juninit, *jnrc;
            idx = (int)get_u16(p + 1); slot = JREG_VAR;
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1((sljit_s32)slot), idx * 16 + JSV_TAG);
            juninit = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_TAG_UNINITIALIZED);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1((sljit_s32)slot), idx * 16 + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_U, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_TAG, SLJIT_R1, 0);
            sljit_emit_op2(c, SLJIT_ADD, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            jnrc = sljit_emit_cmp(c, SLJIT_SIG_GREATER_EQUAL, SLJIT_R1, 0, SLJIT_IMM, 0);
            sljit_emit_op2(c, SLJIT_ADD | SLJIT_32, SLJIT_MEM1(SLJIT_R0), JIT_RC_OFF,
                           SLJIT_MEM1(SLJIT_R0), JIT_RC_OFF, SLJIT_IMM, 1);
            sljit_set_label(jnrc, sljit_emit_label(c));
            {
                struct sljit_jump *jok = sljit_emit_jump(c, SLJIT_JUMP);
                sljit_set_label(juninit, sljit_emit_label(c));
                jit_emit_sync_call(c, (void *)&jit_helper_throw_uninit, 0, 0, 0, 0, 1, fx, &nfx);
                sljit_set_label(jok, sljit_emit_label(c));
            }
            break;
        }
        case OP_get_loc0: idx=0; slot=JREG_VAR; goto get_loc_common;
        case OP_get_loc1: idx=1; slot=JREG_VAR; goto get_loc_common;
        case OP_get_loc2: idx=2; slot=JREG_VAR; goto get_loc_common;
        case OP_get_loc3: idx=3; slot=JREG_VAR; goto get_loc_common;
        case OP_get_loc8: idx=p[1]; slot=JREG_VAR; goto get_loc_common;
        case OP_get_loc: idx=(int)get_u16(p+1); slot=JREG_VAR; goto get_loc_common;
        case OP_get_arg: idx=(int)get_u16(p+1); slot=JREG_ARG; goto get_loc_common;
        case OP_get_arg0: idx=0; slot=JREG_ARG; goto get_loc_common;
        case OP_get_arg1: idx=1; slot=JREG_ARG; goto get_loc_common;
        case OP_get_arg2: idx=2; slot=JREG_ARG; goto get_loc_common;
        case OP_get_arg3: idx=3; slot=JREG_ARG; goto get_loc_common;
        get_loc_common: {
            struct sljit_jump *jnrc;
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1((sljit_s32)slot), idx*16+JSV_U);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1((sljit_s32)slot), idx*16+JSV_TAG);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_U, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_TAG, SLJIT_R1, 0);
            sljit_emit_op2(c, SLJIT_ADD, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            jnrc = sljit_emit_cmp(c, SLJIT_SIG_GREATER_EQUAL, SLJIT_R1, 0, SLJIT_IMM, 0);
            sljit_emit_op2(c, SLJIT_ADD | SLJIT_32, SLJIT_MEM1(SLJIT_R0), JIT_RC_OFF,
                           SLJIT_MEM1(SLJIT_R0), JIT_RC_OFF, SLJIT_IMM, 1);
            sljit_set_label(jnrc, sljit_emit_label(c));
            break;
        }
        /* ---- put_loc / put_arg / put_loc_check / set_loc ---- */
        case OP_put_loc_check:
        case OP_put_loc_check_init:
            idx = (int)get_u16(p + 1); slot = JREG_VAR; goto put_loc_common;
        case OP_put_loc0: idx=0; slot=JREG_VAR; goto put_loc_common;
        case OP_put_loc1: idx=1; slot=JREG_VAR; goto put_loc_common;
        case OP_put_loc2: idx=2; slot=JREG_VAR; goto put_loc_common;
        case OP_put_loc3: idx=3; slot=JREG_VAR; goto put_loc_common;
        case OP_put_loc8: idx=p[1]; slot=JREG_VAR; goto put_loc_common;
        case OP_put_loc: idx=(int)get_u16(p+1); slot=JREG_VAR; goto put_loc_common;
        case OP_put_arg: idx=(int)get_u16(p+1); slot=JREG_ARG; goto put_loc_common;
        case OP_put_arg0: idx=0; slot=JREG_ARG; goto put_loc_common;
        case OP_put_arg1: idx=1; slot=JREG_ARG; goto put_loc_common;
        case OP_put_arg2: idx=2; slot=JREG_ARG; goto put_loc_common;
        case OP_put_arg3: idx=3; slot=JREG_ARG; goto put_loc_common;
        put_loc_common: {
            struct sljit_jump *jslow, *jdone;
            sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1((sljit_s32)slot), idx*16+JSV_TAG);
            jslow = sljit_emit_cmp(c, SLJIT_SIG_LESS, SLJIT_R1, 0, SLJIT_IMM, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), JSV_U);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1((sljit_s32)slot), idx*16+JSV_U, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), JSV_TAG);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1((sljit_s32)slot), idx*16+JSV_TAG, SLJIT_R0, 0);
            jdone = sljit_emit_jump(c, SLJIT_JUMP);
            sljit_set_label(jslow, sljit_emit_label(c));
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_JCTX), JCTX_CTX);
            sljit_emit_op2(c, SLJIT_ADD, SLJIT_R1, 0, (sljit_s32)slot, 0, SLJIT_IMM, (sljit_sw)(idx * 16));
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R2, 0, JREG_SP, 0);
            sljit_emit_icall(c, SLJIT_CALL, SLJIT_ARGS3V(P, P, P),
                             SLJIT_IMM, (sljit_sw)&jit_helper_put_slot);
            sljit_set_label(jdone, sljit_emit_label(c));
            break;
        }
        case OP_set_loc0: idx=0; slot=JREG_VAR; goto set_loc_common;
        case OP_set_loc1: idx=1; slot=JREG_VAR; goto set_loc_common;
        case OP_set_loc2: idx=2; slot=JREG_VAR; goto set_loc_common;
        case OP_set_loc3: idx=3; slot=JREG_VAR; goto set_loc_common;
        case OP_set_loc8: idx=p[1]; slot=JREG_VAR; goto set_loc_common;
        case OP_set_loc: idx=(int)get_u16(p+1); slot=JREG_VAR; goto set_loc_common;
        case OP_set_arg: idx=(int)get_u16(p+1); slot=JREG_ARG; goto set_loc_common;
        case OP_set_arg0: idx=0; slot=JREG_ARG; goto set_loc_common;
        case OP_set_arg1: idx=1; slot=JREG_ARG; goto set_loc_common;
        case OP_set_arg2: idx=2; slot=JREG_ARG; goto set_loc_common;
        case OP_set_arg3: idx=3; slot=JREG_ARG; goto set_loc_common;
        set_loc_common: {
            /* keep sp[-1] on stack (dup if refcounted), store copy into slot */
            struct sljit_jump *jslow, *jdone, *jnrc;
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1((sljit_s32)slot), idx*16+JSV_TAG);
            jslow = sljit_emit_cmp(c, SLJIT_SIG_LESS, SLJIT_R1, 0, SLJIT_IMM, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE+JSV_U);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1((sljit_s32)slot), idx*16+JSV_U, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE+JSV_TAG);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1((sljit_s32)slot), idx*16+JSV_TAG, SLJIT_R1, 0);
            jnrc = sljit_emit_cmp(c, SLJIT_SIG_GREATER_EQUAL, SLJIT_R1, 0, SLJIT_IMM, 0);
            sljit_emit_op2(c, SLJIT_ADD | SLJIT_32, SLJIT_MEM1(SLJIT_R0), JIT_RC_OFF,
                           SLJIT_MEM1(SLJIT_R0), JIT_RC_OFF, SLJIT_IMM, 1);
            sljit_set_label(jnrc, sljit_emit_label(c));
            jdone = sljit_emit_jump(c, SLJIT_JUMP);
            sljit_set_label(jslow, sljit_emit_label(c));
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_JCTX), JCTX_CTX);
            sljit_emit_op2(c, SLJIT_ADD, SLJIT_R1, 0, (sljit_s32)slot, 0, SLJIT_IMM, (sljit_sw)(idx * 16));
            sljit_emit_op2(c, SLJIT_SUB, SLJIT_R2, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            sljit_emit_icall(c, SLJIT_CALL, SLJIT_ARGS3V(P, P, P),
                             SLJIT_IMM, (sljit_sw)&jit_helper_put_slot);
            /* if put_slot moved sp[-1], we also need sp[-1] dup'd */
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1((sljit_s32)slot), idx*16+JSV_TAG);
            jnrc = sljit_emit_cmp(c, SLJIT_SIG_GREATER_EQUAL, SLJIT_R1, 0, SLJIT_IMM, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1((sljit_s32)slot), idx*16+JSV_U);
            sljit_emit_op2(c, SLJIT_ADD | SLJIT_32, SLJIT_MEM1(SLJIT_R0), JIT_RC_OFF,
                           SLJIT_MEM1(SLJIT_R0), JIT_RC_OFF, SLJIT_IMM, 1);
            sljit_set_label(jnrc, sljit_emit_label(c));
            sljit_set_label(jdone, sljit_emit_label(c));
            break;
        }
        /* ---- stack manipulation ---- */
        case OP_drop: {
            struct sljit_jump *jnrc;
            sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), JSV_TAG);
            jnrc = sljit_emit_cmp(c, SLJIT_SIG_GREATER_EQUAL, SLJIT_R1, 0, SLJIT_IMM, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R2, 0, SLJIT_MEM1(JREG_SP), JSV_U);
            jit_emit_free_obj_slot(c, SLJIT_R2, 0);
            sljit_set_label(jnrc, sljit_emit_label(c));
            break;
        }
        case OP_nip: {
            /* a b -> b (free a at sp[-2], move sp[-1] to sp[-2]) */
            struct sljit_jump *jnrc;
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG);
            jnrc = sljit_emit_cmp(c, SLJIT_SIG_GREATER_EQUAL, SLJIT_R1, 0, SLJIT_IMM, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R2, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U);
            jit_emit_free_obj_slot(c, SLJIT_R2, -2 * JSV_SIZE);
            sljit_set_label(jnrc, sljit_emit_label(c));
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG, SLJIT_R0, 0);
            sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            break;
        }
        case OP_nip1: {
            /* a b c -> b c (free a at sp[-3], shift sp[-2], sp[-1] down) */
            struct sljit_jump *jnrc;
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -3 * JSV_SIZE + JSV_TAG);
            jnrc = sljit_emit_cmp(c, SLJIT_SIG_GREATER_EQUAL, SLJIT_R1, 0, SLJIT_IMM, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R2, 0, SLJIT_MEM1(JREG_SP), -3 * JSV_SIZE + JSV_U);
            jit_emit_free_obj_slot(c, SLJIT_R2, -3 * JSV_SIZE);
            sljit_set_label(jnrc, sljit_emit_label(c));
            for (int k = -2; k <= -1; k++) {
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), k * JSV_SIZE + JSV_U);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), (k - 1) * JSV_SIZE + JSV_U, SLJIT_R0, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), k * JSV_SIZE + JSV_TAG);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), (k - 1) * JSV_SIZE + JSV_TAG, SLJIT_R0, 0);
            }
            sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            break;
        }
        case OP_dup: {
            struct sljit_jump *jnrc;
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_U, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_TAG, SLJIT_R1, 0);
            sljit_emit_op2(c, SLJIT_ADD, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            jnrc = sljit_emit_cmp(c, SLJIT_SIG_GREATER_EQUAL, SLJIT_R1, 0, SLJIT_IMM, 0);
            sljit_emit_op2(c, SLJIT_ADD | SLJIT_32, SLJIT_MEM1(SLJIT_R0), JIT_RC_OFF,
                           SLJIT_MEM1(SLJIT_R0), JIT_RC_OFF, SLJIT_IMM, 1);
            sljit_set_label(jnrc, sljit_emit_label(c));
            break;
        }
        case OP_dup1: {
            /* a b -> a a b */
            struct sljit_jump *jnrc;
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_U, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_TAG, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG, SLJIT_R1, 0);
            sljit_emit_op2(c, SLJIT_ADD, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            jnrc = sljit_emit_cmp(c, SLJIT_SIG_GREATER_EQUAL, SLJIT_R1, 0, SLJIT_IMM, 0);
            sljit_emit_op2(c, SLJIT_ADD | SLJIT_32, SLJIT_MEM1(SLJIT_R0), JIT_RC_OFF,
                           SLJIT_MEM1(SLJIT_R0), JIT_RC_OFF, SLJIT_IMM, 1);
            sljit_set_label(jnrc, sljit_emit_label(c));
            break;
        }
        case OP_dup2: {
            /* a b -> a b a b */
            for (int k = 0; k < 2; k++) {
                struct sljit_jump *jnrc;
                sljit_sw src_off = (-2 + k) * JSV_SIZE;
                sljit_sw dst_off = k * JSV_SIZE;
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), src_off + JSV_U);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), src_off + JSV_TAG);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), dst_off + JSV_U, SLJIT_R0, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), dst_off + JSV_TAG, SLJIT_R1, 0);
                jnrc = sljit_emit_cmp(c, SLJIT_SIG_GREATER_EQUAL, SLJIT_R1, 0, SLJIT_IMM, 0);
                sljit_emit_op2(c, SLJIT_ADD | SLJIT_32, SLJIT_MEM1(SLJIT_R0), JIT_RC_OFF,
                               SLJIT_MEM1(SLJIT_R0), JIT_RC_OFF, SLJIT_IMM, 1);
                sljit_set_label(jnrc, sljit_emit_label(c));
            }
            sljit_emit_op2(c, SLJIT_ADD, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, 2 * JSV_SIZE);
            break;
        }
        case OP_swap: {
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U, SLJIT_R1, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG, SLJIT_R1, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG, SLJIT_R0, 0);
            break;
        }
        case OP_insert3: {
            /* obj prop a -> a obj prop a (dup top below 3) */
            struct sljit_jump *jnrc;
            for (int k = -1; k >= -3; k--) {
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), k * JSV_SIZE + JSV_U);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), (k + 1) * JSV_SIZE + JSV_U, SLJIT_R0, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), k * JSV_SIZE + JSV_TAG);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), (k + 1) * JSV_SIZE + JSV_TAG, SLJIT_R0, 0);
            }
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), JSV_U);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), JSV_TAG);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -3 * JSV_SIZE + JSV_U, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -3 * JSV_SIZE + JSV_TAG, SLJIT_R1, 0);
            sljit_emit_op2(c, SLJIT_ADD, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            jnrc = sljit_emit_cmp(c, SLJIT_SIG_GREATER_EQUAL, SLJIT_R1, 0, SLJIT_IMM, 0);
            sljit_emit_op2(c, SLJIT_ADD | SLJIT_32, SLJIT_MEM1(SLJIT_R0), JIT_RC_OFF,
                           SLJIT_MEM1(SLJIT_R0), JIT_RC_OFF, SLJIT_IMM, 1);
            sljit_set_label(jnrc, sljit_emit_label(c));
            break;
        }
        case OP_perm3: {
            /* obj a b -> a obj b (swap sp[-3] and sp[-2]) */
            for (int f = 0; f < 2; f++) {
                sljit_sw fo = (f == 0) ? JSV_U : JSV_TAG;
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -3 * JSV_SIZE + fo);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + fo);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -3 * JSV_SIZE + fo, SLJIT_R1, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + fo, SLJIT_R0, 0);
            }
            break;
        }
        case OP_rot3l: {
            /* x a b -> a b x */
            for (int f = 0; f < 2; f++) {
                sljit_sw fo = (f == 0) ? JSV_U : JSV_TAG;
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -3 * JSV_SIZE + fo);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + fo);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -3 * JSV_SIZE + fo, SLJIT_R1, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + fo);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + fo, SLJIT_R1, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + fo, SLJIT_R0, 0);
            }
            break;
        }
        case OP_rot3r: {
            /* a b x -> x a b */
            for (int f = 0; f < 2; f++) {
                sljit_sw fo = (f == 0) ? JSV_U : JSV_TAG;
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + fo);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + fo);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + fo, SLJIT_R1, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -3 * JSV_SIZE + fo);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + fo, SLJIT_R1, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -3 * JSV_SIZE + fo, SLJIT_R0, 0);
            }
            break;
        }

        /* ---- add / sub / mul / div : int32 + float64 inline, else helper ---- */
        case OP_add: case OP_sub: case OP_mul: case OP_div: {
            struct sljit_jump *j_not_int = NULL, *j_ovf = NULL, *j_negz = NULL;
            struct sljit_jump *j_f1, *j_f2, *j_i1, *j_i2, *j_l1, *j_l2, *j_done;
            struct sljit_label *l_float, *l_slow, *l_done;

            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);

            if (op != OP_div) {
                sljit_s32 aop = (op == OP_add) ? SLJIT_ADD : (op == OP_sub) ? SLJIT_SUB : SLJIT_MUL;
                sljit_emit_op2(c, SLJIT_OR, SLJIT_R2, 0, SLJIT_R0, 0, SLJIT_R1, 0);
                j_not_int = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R2, 0, SLJIT_IMM, 0);
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U);
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
                if (op == OP_mul)
                    sljit_emit_op2(c, SLJIT_OR | SLJIT_32, SLJIT_R2, 0, SLJIT_R0, 0, SLJIT_R1, 0);
                sljit_emit_op2(c, aop, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_R1, 0);
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_R0, 0);
                j_ovf = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_R1, 0);
                if (op == OP_mul) {
                    struct sljit_jump *j_nz = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, 0);
                    sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R2, 0, SLJIT_R2, 0);
                    j_negz = sljit_emit_cmp(c, SLJIT_SIG_LESS, SLJIT_R2, 0, SLJIT_IMM, 0);
                    sljit_set_label(j_nz, sljit_emit_label(c));
                }
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U, SLJIT_R0, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG, SLJIT_IMM, 0);
                sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
                j_done = sljit_emit_jump(c, SLJIT_JUMP);
                l_float = sljit_emit_label(c);
                sljit_set_label(j_not_int, l_float);
                sljit_set_label(j_ovf, l_float);
                if (j_negz) sljit_set_label(j_negz, l_float);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            } else {
                j_done = NULL;
            }

            /* Float64 fast path: both operands must be INT(0) or FLOAT64(8) */
            j_f1 = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_FLOAT64);
            j_i1 = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_INT);
            sljit_set_label(j_f1, sljit_emit_label(c));
            j_f2 = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_TAG_FLOAT64);
            j_i2 = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_TAG_INT);
            sljit_set_label(j_f2, sljit_emit_label(c));

            /* Load op1 -> FR0 */
            j = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_FLOAT64);
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R2, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U);
            sljit_emit_fop1(c, SLJIT_CONV_F64_FROM_S32, SLJIT_FR0, 0, SLJIT_R2, 0);
            j_l1 = sljit_emit_jump(c, SLJIT_JUMP);
            sljit_set_label(j, sljit_emit_label(c));
            sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_FR0, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U);
            sljit_set_label(j_l1, sljit_emit_label(c));

            /* Load op2 -> FR1 */
            j = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_TAG_FLOAT64);
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R2, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            sljit_emit_fop1(c, SLJIT_CONV_F64_FROM_S32, SLJIT_FR1, 0, SLJIT_R2, 0);
            j_l2 = sljit_emit_jump(c, SLJIT_JUMP);
            sljit_set_label(j, sljit_emit_label(c));
            sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_FR1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            sljit_set_label(j_l2, sljit_emit_label(c));

            sljit_emit_fop2(c, (op == OP_add) ? SLJIT_ADD_F64 :
                               (op == OP_sub) ? SLJIT_SUB_F64 :
                               (op == OP_mul) ? SLJIT_MUL_F64 : SLJIT_DIV_F64,
                            SLJIT_FR0, 0, SLJIT_FR0, 0, SLJIT_FR1, 0);
            sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U, SLJIT_FR0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG, SLJIT_IMM, JS_TAG_FLOAT64);
            sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            {
                struct sljit_jump *j_fdone = sljit_emit_jump(c, SLJIT_JUMP);
                l_slow = sljit_emit_label(c);
                sljit_set_label(j_i1, l_slow);
                sljit_set_label(j_i2, l_slow);
                jit_emit_sync_call(c, (void *)&jit_helper_binop, 1, (sljit_sw)op, 0, 0, 1, fx, &nfx);
                l_done = sljit_emit_label(c);
                if (j_done) sljit_set_label(j_done, l_done);
                sljit_set_label(j_fdone, l_done);
            }
            break;
        }
        /* ---- mod / bitwise : int fast path, else helper ---- */
        case OP_mod: case OP_shl: case OP_sar: case OP_shr:
        case OP_and: case OP_or: case OP_xor: {
            struct sljit_jump *j_slow, *j_slow2 = NULL, *j_slow3 = NULL, *j_done;
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            sljit_emit_op2(c, SLJIT_OR, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_R1, 0);
            j_slow = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, 0);
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            if (op == OP_mod) {
                j_slow2 = sljit_emit_cmp(c, SLJIT_SIG_LESS, SLJIT_R0, 0, SLJIT_IMM, 0);
                j_slow3 = sljit_emit_cmp(c, SLJIT_SIG_LESS_EQUAL, SLJIT_R1, 0, SLJIT_IMM, 0);
                sljit_emit_op0(c, SLJIT_DIVMOD_S32);
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_R1, 0);
            } else if (op == OP_shl || op == OP_sar || op == OP_shr) {
                sljit_emit_op2(c, SLJIT_AND, SLJIT_R1, 0, SLJIT_R1, 0, SLJIT_IMM, 31);
                sljit_s32 sop = (op == OP_shl) ? (SLJIT_SHL | SLJIT_32) :
                                (op == OP_sar) ? (SLJIT_ASHR | SLJIT_32) : (SLJIT_LSHR | SLJIT_32);
                sljit_emit_op2(c, sop, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_R1, 0);
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_R0, 0);
                if (op == OP_shr)
                    j_slow2 = sljit_emit_cmp(c, SLJIT_SIG_LESS, SLJIT_R0, 0, SLJIT_IMM, 0);
            } else {
                sljit_s32 bop = (op == OP_and) ? SLJIT_AND : (op == OP_or) ? SLJIT_OR : SLJIT_XOR;
                sljit_emit_op2(c, bop, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_R1, 0);
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_R0, 0);
            }
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG, SLJIT_IMM, 0);
            sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            j_done = sljit_emit_jump(c, SLJIT_JUMP);
            {
                struct sljit_label *lslow = sljit_emit_label(c);
                sljit_set_label(j_slow, lslow);
                if (j_slow2) sljit_set_label(j_slow2, lslow);
                if (j_slow3) sljit_set_label(j_slow3, lslow);
            }
            jit_emit_sync_call(c, (void *)&jit_helper_binop, 1, (sljit_sw)op, 0, 0, 1, fx, &nfx);
            sljit_set_label(j_done, sljit_emit_label(c));
            break;
        }
        /* ---- unary / inc_loc / dec_loc / add_loc ---- */
        case OP_inc_loc: case OP_dec_loc: {
            int lidx = p[1];
            int delta = (op == OP_inc_loc) ? 1 : -1;
            struct sljit_jump *j_slow, *j_ovf, *j_done;
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_VAR), lidx * 16 + JSV_TAG);
            j_slow = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, 0);
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_MEM1(JREG_VAR), lidx * 16 + JSV_U);
            sljit_emit_op2(c, SLJIT_ADD, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_IMM, delta);
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_R0, 0);
            j_ovf = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_R1, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_VAR), lidx * 16 + JSV_U, SLJIT_R0, 0);
            j_done = sljit_emit_jump(c, SLJIT_JUMP);
            {
                struct sljit_label *ls = sljit_emit_label(c);
                sljit_set_label(j_slow, ls);
                sljit_set_label(j_ovf, ls);
            }
            jit_emit_sync_call(c, (void *)&jit_helper_inc_dec_loc, 2, (sljit_sw)lidx, (sljit_sw)op, 0, 1, fx, &nfx);
            sljit_set_label(j_done, sljit_emit_label(c));
            break;
        }
        case OP_add_loc: {
            int ai = p[1];
            struct sljit_jump *j_slow, *j_ovf, *j_done;
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_VAR), ai * 16 + JSV_TAG);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            sljit_emit_op2(c, SLJIT_OR, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_R1, 0);
            j_slow = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, 0);
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_MEM1(JREG_VAR), ai * 16 + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            sljit_emit_op2(c, SLJIT_ADD, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_R1, 0);
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_R0, 0);
            j_ovf = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_R1, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_VAR), ai * 16 + JSV_U, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_VAR), ai * 16 + JSV_TAG, SLJIT_IMM, 0);
            sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            j_done = sljit_emit_jump(c, SLJIT_JUMP);
            {
                struct sljit_label *ls = sljit_emit_label(c);
                sljit_set_label(j_slow, ls);
                sljit_set_label(j_ovf, ls);
            }
            jit_emit_sync_call(c, (void *)&jit_helper_add_loc, 1, (sljit_sw)ai, 0, 0, 1, fx, &nfx);
            sljit_set_label(j_done, sljit_emit_label(c));
            break;
        }
        case OP_post_inc: case OP_post_dec: {
            int delta = (op == OP_post_inc) ? 1 : -1;
            struct sljit_jump *j_slow, *j_ovf, *j_done;
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            j_slow = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_INT);
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            sljit_emit_op2(c, SLJIT_ADD, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_IMM, delta);
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_R0, 0);
            j_ovf = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_R1, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_U, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_TAG, SLJIT_IMM, JS_TAG_INT);
            sljit_emit_op2(c, SLJIT_ADD, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            j_done = sljit_emit_jump(c, SLJIT_JUMP);
            {
                struct sljit_label *ls = sljit_emit_label(c);
                sljit_set_label(j_slow, ls);
                sljit_set_label(j_ovf, ls);
            }
            jit_emit_sync_call(c, (void *)&jit_helper_post_inc_dec, 1, (sljit_sw)op, 0, 0, 1, fx, &nfx);
            sljit_set_label(j_done, sljit_emit_label(c));
            break;
        }
        case OP_lnot: {
            struct sljit_jump *j_ti, *j_tb, *j_done;
            struct sljit_label *l_slow;
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            j_ti = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_INT);
            j_tb = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_BOOL);
            sljit_set_label(j_ti, sljit_emit_label(c));
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            sljit_emit_op2u(c, SLJIT_SUB | SLJIT_SET_Z, SLJIT_R0, 0, SLJIT_IMM, 0);
            sljit_emit_op_flags(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_EQUAL);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG, SLJIT_IMM, JS_TAG_BOOL);
            j_done = sljit_emit_jump(c, SLJIT_JUMP);
            l_slow = sljit_emit_label(c);
            sljit_set_label(j_tb, l_slow);
            jit_emit_sync_call(c, (void *)&jit_helper_unary, 1, (sljit_sw)op, 0, 0, 1, fx, &nfx);
            sljit_set_label(j_done, sljit_emit_label(c));
            break;
        }
        case OP_inc: case OP_dec: case OP_neg: case OP_plus:
        case OP_not: {
            jit_emit_sync_call(c, (void *)&jit_helper_unary, 1, (sljit_sw)op, 0, 0, 1, fx, &nfx);
            break;
        }
        /* ---- comparisons : int32 + float64 inline, else helper ---- */
        case OP_lt: case OP_lte: case OP_gt: case OP_gte:
        case OP_eq: case OP_neq: case OP_strict_eq: case OP_strict_neq: {
            sljit_s32 ct, fct;
            struct sljit_jump *j_not_int, *j_f1, *j_f2, *j_done, *j_fdone;
            switch (op) {
            case OP_lt: ct = SLJIT_SIG_LESS; fct = SLJIT_ORDERED_LESS; break;
            case OP_lte: ct = SLJIT_SIG_LESS_EQUAL; fct = SLJIT_ORDERED_LESS_EQUAL; break;
            case OP_gt: ct = SLJIT_SIG_GREATER; fct = SLJIT_ORDERED_GREATER; break;
            case OP_gte: ct = SLJIT_SIG_GREATER_EQUAL; fct = SLJIT_ORDERED_GREATER_EQUAL; break;
            case OP_eq: case OP_strict_eq: ct = SLJIT_EQUAL; fct = SLJIT_ORDERED_EQUAL; break;
            default: ct = SLJIT_NOT_EQUAL; fct = SLJIT_UNORDERED_OR_NOT_EQUAL; break;
            }
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            sljit_emit_op2(c, SLJIT_OR, SLJIT_R2, 0, SLJIT_R0, 0, SLJIT_R1, 0);
            j_not_int = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R2, 0, SLJIT_IMM, 0);

            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U, SLJIT_IMM, 1);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG, SLJIT_IMM, JS_TAG_BOOL);
            j = sljit_emit_cmp(c, ct, SLJIT_R0, 0, SLJIT_R1, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U, SLJIT_IMM, 0);
            sljit_set_label(j, sljit_emit_label(c));
            sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            j_done = sljit_emit_jump(c, SLJIT_JUMP);

            sljit_set_label(j_not_int, sljit_emit_label(c));
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            j_f1 = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_FLOAT64);
            j_f2 = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_TAG_FLOAT64);
            sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_FR0, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U);
            sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_FR1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U, SLJIT_IMM, 1);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG, SLJIT_IMM, JS_TAG_BOOL);
            j = sljit_emit_fcmp(c, fct, SLJIT_FR0, 0, SLJIT_FR1, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U, SLJIT_IMM, 0);
            sljit_set_label(j, sljit_emit_label(c));
            sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            j_fdone = sljit_emit_jump(c, SLJIT_JUMP);

            {
                struct sljit_label *ls = sljit_emit_label(c);
                sljit_set_label(j_f1, ls);
                sljit_set_label(j_f2, ls);
            }
            jit_emit_sync_call(c, (void *)&jit_helper_cmp, 1, (sljit_sw)op, 0, 0, 1, fx, &nfx);
            {
                struct sljit_label *ld = sljit_emit_label(c);
                sljit_set_label(j_done, ld);
                sljit_set_label(j_fdone, ld);
            }
            break;
        }
        /* ---- TypedArray / Array element access ---- */
        case OP_get_array_el:
        case OP_get_array_el2: {
            int keep_obj = (op == OP_get_array_el2);
            struct sljit_jump *j_no, *j_ni, *j_oob, *j_not_u8, *j_not_u8c, *j_not_i32, *j_not_f64;
            struct sljit_jump *j_not_f32, *j_not_u16, *j_not_u32, *j_u32_big;
            struct sljit_jump *j_u8_ok, *j_i32_ok, *j_f64_ok, *j_f32_ok, *j_u16_ok, *j_u32_ok;
            struct sljit_label *l_slow, *l_done;

            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG);
            j_no = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_OBJECT);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            j_ni = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_INT);

            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R2, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV_U32, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV_U32, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R2),
                           (sljit_sw)offsetof(JSObject, u.array.count));
            j_oob = sljit_emit_cmp(c, SLJIT_GREATER_EQUAL, SLJIT_R0, 0, SLJIT_R1, 0);

            sljit_emit_op1(c, SLJIT_MOV_U16, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R2),
                           (sljit_sw)offsetof(JSObject, class_id));

            /* Uint8Array / Uint8ClampedArray */
            j_not_u8 = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_CLASS_UINT8_ARRAY);
            j_not_u8c = NULL;
            {
                struct sljit_label *l_u8 = sljit_emit_label(c);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R2),
                               (sljit_sw)offsetof(JSObject, u.array.u.ptr));
                sljit_emit_op1(c, SLJIT_MOV_U8, SLJIT_R0, 0, SLJIT_MEM2(SLJIT_R1, SLJIT_R0), 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U, SLJIT_R0, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG, SLJIT_IMM, JS_TAG_INT);
                if (!keep_obj) {
                    jit_emit_free_obj_slot(c, SLJIT_R2, -2 * JSV_SIZE);
                    sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
                    sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U, SLJIT_R0, 0);
                    sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG, SLJIT_IMM, JS_TAG_INT);
                    sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
                }
                j_u8_ok = sljit_emit_jump(c, SLJIT_JUMP);

                sljit_set_label(j_not_u8, sljit_emit_label(c));
                j_not_u8c = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_CLASS_UINT8C_ARRAY);
                sljit_set_label(j_not_u8c, l_u8);
            }

            /* Int32Array */
            j_not_i32 = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_CLASS_INT32_ARRAY);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R2),
                           (sljit_sw)offsetof(JSObject, u.array.u.ptr));
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_MEM2(SLJIT_R1, SLJIT_R0), 2);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG, SLJIT_IMM, JS_TAG_INT);
            if (!keep_obj) {
                jit_emit_free_obj_slot(c, SLJIT_R2, -2 * JSV_SIZE);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U, SLJIT_R0, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG, SLJIT_IMM, JS_TAG_INT);
                sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            }
            j_i32_ok = sljit_emit_jump(c, SLJIT_JUMP);

            /* Float64Array */
            sljit_set_label(j_not_i32, sljit_emit_label(c));
            j_not_f64 = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_CLASS_FLOAT64_ARRAY);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R2),
                           (sljit_sw)offsetof(JSObject, u.array.u.ptr));
            sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_FR0, 0, SLJIT_MEM2(SLJIT_R1, SLJIT_R0), 3);
            sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U, SLJIT_FR0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG, SLJIT_IMM, JS_TAG_FLOAT64);
            if (!keep_obj) {
                jit_emit_free_obj_slot(c, SLJIT_R2, -2 * JSV_SIZE);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U, SLJIT_R0, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG, SLJIT_IMM, JS_TAG_FLOAT64);
                sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            }
            j_f64_ok = sljit_emit_jump(c, SLJIT_JUMP);

            /* Float32Array */
            sljit_set_label(j_not_f64, sljit_emit_label(c));
            j_not_f32 = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_CLASS_FLOAT32_ARRAY);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R2),
                           (sljit_sw)offsetof(JSObject, u.array.u.ptr));
            sljit_emit_fop1(c, SLJIT_MOV_F32, SLJIT_FR0, 0, SLJIT_MEM2(SLJIT_R1, SLJIT_R0), 2);
            sljit_emit_fop1(c, SLJIT_CONV_F64_FROM_F32, SLJIT_FR0, 0, SLJIT_FR0, 0);
            sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U, SLJIT_FR0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG, SLJIT_IMM, JS_TAG_FLOAT64);
            if (!keep_obj) {
                jit_emit_free_obj_slot(c, SLJIT_R2, -2 * JSV_SIZE);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U, SLJIT_R0, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG, SLJIT_IMM, JS_TAG_FLOAT64);
                sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            }
            j_f32_ok = sljit_emit_jump(c, SLJIT_JUMP);

            /* Uint16Array */
            sljit_set_label(j_not_f32, sljit_emit_label(c));
            j_not_u16 = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_CLASS_UINT16_ARRAY);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R2),
                           (sljit_sw)offsetof(JSObject, u.array.u.ptr));
            sljit_emit_op1(c, SLJIT_MOV_U16, SLJIT_R0, 0, SLJIT_MEM2(SLJIT_R1, SLJIT_R0), 1);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG, SLJIT_IMM, JS_TAG_INT);
            if (!keep_obj) {
                jit_emit_free_obj_slot(c, SLJIT_R2, -2 * JSV_SIZE);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U, SLJIT_R0, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG, SLJIT_IMM, JS_TAG_INT);
                sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            }
            j_u16_ok = sljit_emit_jump(c, SLJIT_JUMP);

            /* Uint32Array (< 0x80000000 fits in JS_TAG_INT) */
            sljit_set_label(j_not_u16, sljit_emit_label(c));
            j_not_u32 = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_CLASS_UINT32_ARRAY);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R2),
                           (sljit_sw)offsetof(JSObject, u.array.u.ptr));
            sljit_emit_op1(c, SLJIT_MOV_U32, SLJIT_R0, 0, SLJIT_MEM2(SLJIT_R1, SLJIT_R0), 2);
            j_u32_big = sljit_emit_cmp(c, SLJIT_GREATER_EQUAL, SLJIT_R0, 0, SLJIT_IMM, (sljit_sw)0x80000000ULL);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG, SLJIT_IMM, JS_TAG_INT);
            if (!keep_obj) {
                jit_emit_free_obj_slot(c, SLJIT_R2, -2 * JSV_SIZE);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U, SLJIT_R0, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG, SLJIT_IMM, JS_TAG_INT);
                sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            }
            j_u32_ok = sljit_emit_jump(c, SLJIT_JUMP);

            l_slow = sljit_emit_label(c);
            sljit_set_label(j_no, l_slow);
            sljit_set_label(j_ni, l_slow);
            sljit_set_label(j_oob, l_slow);
            sljit_set_label(j_not_u32, l_slow);
            sljit_set_label(j_u32_big, l_slow);
            jit_emit_sync_call(c, (void *)&jit_helper_get_array_el, 1, (sljit_sw)keep_obj, 0, 0, 1, fx, &nfx);

            l_done = sljit_emit_label(c);
            sljit_set_label(j_u8_ok, l_done);
            sljit_set_label(j_i32_ok, l_done);
            sljit_set_label(j_f64_ok, l_done);
            sljit_set_label(j_f32_ok, l_done);
            sljit_set_label(j_u16_ok, l_done);
            sljit_set_label(j_u32_ok, l_done);
            break;
        }
        case OP_put_array_el: {
            struct sljit_jump *j_no, *j_ni, *j_oob, *j_not_u8, *j_u8_nv, *j_not_f64, *j_f64_f, *j_f64_nv;
            struct sljit_jump *j_u8_ok, *j_f64_ok, *j_f64_cv;
            struct sljit_label *l_slow, *l_done;

            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -3 * JSV_SIZE + JSV_TAG);
            j_no = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_OBJECT);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG);
            j_ni = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_INT);

            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R2, 0, SLJIT_MEM1(JREG_SP), -3 * JSV_SIZE + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV_U32, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV_U32, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R2),
                           (sljit_sw)offsetof(JSObject, u.array.count));
            j_oob = sljit_emit_cmp(c, SLJIT_GREATER_EQUAL, SLJIT_R0, 0, SLJIT_R1, 0);

            sljit_emit_op1(c, SLJIT_MOV_U16, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R2),
                           (sljit_sw)offsetof(JSObject, class_id));

            /* Uint8Array */
            j_not_u8 = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_CLASS_UINT8_ARRAY);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            j_u8_nv = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_TAG_INT);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R2),
                           (sljit_sw)offsetof(JSObject, u.array.u.ptr));
            sljit_emit_op1(c, SLJIT_MOV_U8, SLJIT_R3, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV_U8, SLJIT_MEM2(SLJIT_R1, SLJIT_R0), 0, SLJIT_R3, 0);
            jit_emit_free_obj_slot(c, SLJIT_R2, -3 * JSV_SIZE);
            sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, 3 * JSV_SIZE);
            j_u8_ok = sljit_emit_jump(c, SLJIT_JUMP);

            /* Float64Array */
            sljit_set_label(j_not_u8, sljit_emit_label(c));
            j_not_f64 = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_CLASS_FLOAT64_ARRAY);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            j_f64_f = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_TAG_FLOAT64);
            j_f64_nv = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_TAG_INT);
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            sljit_emit_fop1(c, SLJIT_CONV_F64_FROM_S32, SLJIT_FR0, 0, SLJIT_R1, 0);
            j_f64_cv = sljit_emit_jump(c, SLJIT_JUMP);
            sljit_set_label(j_f64_f, sljit_emit_label(c));
            sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_FR0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            sljit_set_label(j_f64_cv, sljit_emit_label(c));
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R2),
                           (sljit_sw)offsetof(JSObject, u.array.u.ptr));
            sljit_emit_fop1(c, SLJIT_MOV_F64, SLJIT_MEM2(SLJIT_R1, SLJIT_R0), 3, SLJIT_FR0, 0);
            jit_emit_free_obj_slot(c, SLJIT_R2, -3 * JSV_SIZE);
            sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, 3 * JSV_SIZE);
            j_f64_ok = sljit_emit_jump(c, SLJIT_JUMP);

            l_slow = sljit_emit_label(c);
            sljit_set_label(j_no, l_slow);
            sljit_set_label(j_ni, l_slow);
            sljit_set_label(j_oob, l_slow);
            sljit_set_label(j_u8_nv, l_slow);
            sljit_set_label(j_not_f64, l_slow);
            sljit_set_label(j_f64_nv, l_slow);
            jit_emit_sync_call(c, (void *)&jit_helper_put_array_el, 0, 0, 0, 0, 1, fx, &nfx);

            l_done = sljit_emit_label(c);
            sljit_set_label(j_u8_ok, l_done);
            sljit_set_label(j_f64_ok, l_done);
            break;
        }
        /* ---- Object field IC fast paths ---- */
        case OP_get_field:
        case OP_get_field2: {
            JSAtom atom = (JSAtom)get_u32(p + 1);
            int keep_obj = (op == OP_get_field2);
            JitFieldIC *ic = &ics[ic_idx++];
            struct sljit_jump *j_no, *j_ns, *j_own, *j_np, *j_nhs, *jnrc, *j_done;
            struct sljit_label *l_slow, *l_load;

            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            j_no = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_OBJECT);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R2, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_R2),
                           (sljit_sw)offsetof(JSObject, shape));
            sljit_emit_const(c, SLJIT_MOV, SLJIT_R1, 0, (sljit_sw)ic);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R3, 0, SLJIT_MEM1(SLJIT_R1),
                           (sljit_sw)offsetof(JitFieldIC, shape));
            j_ns = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_R3, 0);

            /* Check own vs 1-hop proto IC */
            sljit_emit_op1(c, SLJIT_MOV_U8, SLJIT_R3, 0, SLJIT_MEM1(SLJIT_R1),
                           (sljit_sw)offsetof(JitFieldIC, is_proto));
            j_own = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R3, 0, SLJIT_IMM, 0);

            /* 1-hop proto: R0 = p->shape->proto; check proto && proto->shape == ic->holder_shape */
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_R0),
                           (sljit_sw)offsetof(JSShape, proto));
            j_np = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R3, 0, SLJIT_MEM1(SLJIT_R0),
                           (sljit_sw)offsetof(JSObject, shape));
            {
                struct sljit_jump *j_l;
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), NLOOP_TMP_OFF, SLJIT_R0, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_R1),
                               (sljit_sw)offsetof(JitFieldIC, holder_shape));
                j_nhs = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R3, 0, SLJIT_R0, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_TMP_OFF);
                j_l = sljit_emit_jump(c, SLJIT_JUMP);

                sljit_set_label(j_own, sljit_emit_label(c));
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_R2, 0);
                l_load = sljit_emit_label(c);
                sljit_set_label(j_l, l_load);
            }

            /* IC hit: R0 = holder (p or proto), R1 = ic, R2 = receiver p */
            sljit_emit_op1(c, SLJIT_MOV_U32, SLJIT_R3, 0, SLJIT_MEM1(SLJIT_R1),
                           (sljit_sw)offsetof(JitFieldIC, offset));
            sljit_emit_op2(c, SLJIT_SHL, SLJIT_R3, 0, SLJIT_R3, 0, SLJIT_IMM, 4); /* * sizeof(JSProperty)=16 */
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R0),
                           (sljit_sw)offsetof(JSObject, prop));
            sljit_emit_op2(c, SLJIT_ADD, SLJIT_R1, 0, SLJIT_R1, 0, SLJIT_R3, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_R1), JSV_U);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R3, 0, SLJIT_MEM1(SLJIT_R1), JSV_TAG);
            /* dup if refcounted */
            jnrc = sljit_emit_cmp(c, SLJIT_SIG_GREATER_EQUAL, SLJIT_R3, 0, SLJIT_IMM, 0);
            sljit_emit_op2(c, SLJIT_ADD | SLJIT_32, SLJIT_MEM1(SLJIT_R0), JIT_RC_OFF,
                           SLJIT_MEM1(SLJIT_R0), JIT_RC_OFF, SLJIT_IMM, 1);
            sljit_set_label(jnrc, sljit_emit_label(c));

            if (keep_obj) {
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_U, SLJIT_R0, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), JSV_TAG, SLJIT_R3, 0);
                sljit_emit_op2(c, SLJIT_ADD, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            } else {
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), NLOOP_TMP_OFF, SLJIT_R0, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), NLOOP_TMP_OFF + 8, SLJIT_R3, 0);
                jit_emit_free_obj_slot(c, SLJIT_R2, -JSV_SIZE);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_TMP_OFF);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U, SLJIT_R0, 0);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_SP), NLOOP_TMP_OFF + 8);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG, SLJIT_R0, 0);
            }
            j_done = sljit_emit_jump(c, SLJIT_JUMP);

            l_slow = sljit_emit_label(c);
            sljit_set_label(j_no, l_slow);
            sljit_set_label(j_ns, l_slow);
            sljit_set_label(j_np, l_slow);
            sljit_set_label(j_nhs, l_slow);
            jit_emit_sync_call(c, (void *)&jit_helper_get_field_ic, 3,
                               (sljit_sw)atom, (sljit_sw)ic, (sljit_sw)keep_obj, 1, fx, &nfx);
            sljit_set_label(j_done, sljit_emit_label(c));
            break;
        }
        case OP_put_field: {
            JSAtom atom = (JSAtom)get_u32(p + 1);
            JitFieldIC *ic = &ics[ic_idx++];
            struct sljit_jump *j_no, *j_ns, *j_nw, *j_oldrc, *j_newrc, *j_done;
            struct sljit_label *l_slow;

            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_TAG);
            j_no = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_OBJECT);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            j_newrc = sljit_emit_cmp(c, SLJIT_SIG_LESS, SLJIT_R0, 0, SLJIT_IMM, 0);

            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R2, 0, SLJIT_MEM1(JREG_SP), -2 * JSV_SIZE + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_R2),
                           (sljit_sw)offsetof(JSObject, shape));
            sljit_emit_const(c, SLJIT_MOV, SLJIT_R1, 0, (sljit_sw)ic);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R3, 0, SLJIT_MEM1(SLJIT_R1),
                           (sljit_sw)offsetof(JitFieldIC, shape));
            j_ns = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_R3, 0);
            sljit_emit_op1(c, SLJIT_MOV_U8, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_R1),
                           (sljit_sw)offsetof(JitFieldIC, is_writable));
            j_nw = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, 0);

            sljit_emit_op1(c, SLJIT_MOV_U32, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_R1),
                           (sljit_sw)offsetof(JitFieldIC, offset));
            sljit_emit_op2(c, SLJIT_SHL, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_IMM, 4);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R2),
                           (sljit_sw)offsetof(JSObject, prop));
            sljit_emit_op2(c, SLJIT_ADD, SLJIT_R1, 0, SLJIT_R1, 0, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_R1), JSV_TAG);
            j_oldrc = sljit_emit_cmp(c, SLJIT_SIG_LESS, SLJIT_R0, 0, SLJIT_IMM, 0);

            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(SLJIT_R1), JSV_U, SLJIT_R0, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(SLJIT_R1), JSV_TAG, SLJIT_R0, 0);
            jit_emit_free_obj_slot(c, SLJIT_R2, -2 * JSV_SIZE);
            sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, 2 * JSV_SIZE);
            j_done = sljit_emit_jump(c, SLJIT_JUMP);

            l_slow = sljit_emit_label(c);
            sljit_set_label(j_no, l_slow);
            sljit_set_label(j_newrc, l_slow);
            sljit_set_label(j_ns, l_slow);
            sljit_set_label(j_nw, l_slow);
            sljit_set_label(j_oldrc, l_slow);
            jit_emit_sync_call(c, (void *)&jit_helper_put_field_ic, 2,
                               (sljit_sw)atom, (sljit_sw)ic, 0, 1, fx, &nfx);
            sljit_set_label(j_done, sljit_emit_label(c));
            break;
        }
        /* ---- Object / Array / Closure / Global / Call helpers ---- */
        case OP_get_var:
        case OP_get_var_undef: {
            JSAtom atom = (JSAtom)get_u32(p + 1);
            jit_emit_sync_call(c, (void *)&jit_helper_get_var, 2,
                               (sljit_sw)atom, (sljit_sw)(op - OP_get_var_undef), 0, 1, fx, &nfx);
            break;
        }
        case OP_put_var:
        case OP_put_var_init: {
            JSAtom atom = (JSAtom)get_u32(p + 1);
            jit_emit_sync_call(c, (void *)&jit_helper_put_var, 2,
                               (sljit_sw)atom, (sljit_sw)(op - OP_put_var), 0, 1, fx, &nfx);
            break;
        }
        case OP_object:
            jit_emit_sync_call(c, (void *)&jit_helper_object, 0, 0, 0, 0, 1, fx, &nfx);
            break;
        case OP_array_from: {
            int argc = (int)get_u16(p + 1);
            jit_emit_sync_call(c, (void *)&jit_helper_array_from, 1, (sljit_sw)argc, 0, 0, 1, fx, &nfx);
            break;
        }
        case OP_define_field: {
            JSAtom atom = (JSAtom)get_u32(p + 1);
            JitFieldIC *ic = &ics[ic_idx++];
            jit_emit_sync_call(c, (void *)&jit_helper_define_field, 2, (sljit_sw)atom, (sljit_sw)ic, 0, 1, fx, &nfx);
            break;
        }
        case OP_fclosure:
        case OP_fclosure8: {
            int cidx = (op == OP_fclosure8) ? (int)p[1] : (int)get_u32(p + 1);
            jit_emit_sync_call(c, (void *)&jit_helper_fclosure, 1, (sljit_sw)cidx, 0, 0, 1, fx, &nfx);
            break;
        }
        case OP_call0: case OP_call1: case OP_call2: case OP_call3:
        case OP_call: {
            int argc = (op == OP_call) ? (int)get_u16(p + 1) : (op - OP_call0);
            jit_emit_sync_call(c, (void *)&jit_helper_call, 2, (sljit_sw)argc, 0, 0, 1, fx, &nfx);
            break;
        }
        case OP_call_method: {
            int argc = (int)get_u16(p + 1);
            jit_emit_sync_call(c, (void *)&jit_helper_call, 2, (sljit_sw)argc, 1, 0, 1, fx, &nfx);
            break;
        }
        case OP_call_constructor: {
            int argc = (int)get_u16(p + 1);
            jit_emit_sync_call(c, (void *)&jit_helper_call, 2, (sljit_sw)argc, 2, 0, 1, fx, &nfx);
            break;
        }
        case OP_get_length:
            jit_emit_sync_call(c, (void *)&jit_helper_get_length, 0, 0, 0, 0, 1, fx, &nfx);
            break;
        case OP_to_propkey:
        case OP_to_propkey2: {
            struct sljit_jump *j_fast;
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            j_fast = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_INT);
            jit_emit_sync_call(c, (void *)&jit_helper_to_propkey, 0, 0, 0, 0, 1, fx, &nfx);
            sljit_set_label(j_fast, sljit_emit_label(c));
            break;
        }
        case OP_to_object:
            jit_emit_sync_call(c, (void *)&jit_helper_to_object, 0, 0, 0, 0, 1, fx, &nfx);
            break;
        case OP_typeof:
            jit_emit_sync_call(c, (void *)&jit_helper_typeof, 0, 0, 0, 0, 0, fx, &nfx);
            break;
        case OP_is_undefined_or_null: {
            struct sljit_jump *j_u, *j_n, *j_nrc, *j_done;
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            j_u = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_TAG_UNDEFINED);
            j_n = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, JS_TAG_NULL);
            j_nrc = sljit_emit_cmp(c, SLJIT_SIG_GREATER_EQUAL, SLJIT_R1, 0, SLJIT_IMM, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R2, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            jit_emit_free_obj_slot(c, SLJIT_R2, -JSV_SIZE);
            sljit_set_label(j_nrc, sljit_emit_label(c));
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U, SLJIT_IMM, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG, SLJIT_IMM, JS_TAG_BOOL);
            j_done = sljit_emit_jump(c, SLJIT_JUMP);
            {
                struct sljit_label *lt = sljit_emit_label(c);
                sljit_set_label(j_u, lt);
                sljit_set_label(j_n, lt);
            }
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U, SLJIT_IMM, 1);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG, SLJIT_IMM, JS_TAG_BOOL);
            sljit_set_label(j_done, sljit_emit_label(c));
            break;
        }
        /* ---- goto* : poll interrupts only on backward branches ---- */
        case OP_goto: case OP_goto8: case OP_goto16: {
            int rel, target;
            if (op == OP_goto) rel = (int)(int32_t)get_u32(p + 1);
            else if (op == OP_goto16) rel = (int)(int16_t)get_u16(p + 1);
            else rel = (int)(int8_t)p[1];
            target = ooff + 1 + rel;
            if (target <= ooff) {
                struct sljit_jump *jslow, *jcont;
                struct sljit_label *lslow;
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_JCTX), JCTX_CTX);
                sljit_emit_op1(c, SLJIT_MOV_U32, SLJIT_R1, 0,
                               SLJIT_MEM1(SLJIT_R0),
                               (sljit_sw)offsetof(JSContext, interrupt_counter));
                sljit_emit_op2(c, SLJIT_SUB, SLJIT_R1, 0, SLJIT_R1, 0, SLJIT_IMM, 1);
                sljit_emit_op1(c, SLJIT_MOV32, SLJIT_MEM1(SLJIT_R0),
                               (sljit_sw)offsetof(JSContext, interrupt_counter),
                               SLJIT_R1, 0);
                jslow = sljit_emit_cmp(c, SLJIT_SIG_LESS_EQUAL, SLJIT_R1, 0, SLJIT_IMM, 0);
                j = sljit_emit_jump(c, SLJIT_JUMP);
                fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BRANCH; fx[nfx].target_off = target; nfx++;
                lslow = sljit_emit_label(c);
                sljit_set_label(jslow, lslow);
                sljit_emit_const(c, SLJIT_MOV, JREG_PC, 0, (sljit_sw)(bc + ooff));
                sljit_emit_icall(c, SLJIT_CALL, SLJIT_ARGS1(W, P),
                                 SLJIT_IMM, (sljit_sw)&jit_poll_slow);
                j = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_RETURN_REG, 0, SLJIT_IMM, 0);
                fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
                jcont = sljit_emit_jump(c, SLJIT_JUMP);
                fx[nfx].jmp = jcont; fx[nfx].kind = JIT_FX_BRANCH; fx[nfx].target_off = target; nfx++;
            } else {
                j = sljit_emit_jump(c, SLJIT_JUMP);
                fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BRANCH; fx[nfx].target_off = target; nfx++;
            }
            break;
        }
        /* ---- if_* : int/bool fast path, else jit_helper_to_bool ---- */
        case OP_if_true8: case OP_if_false8: case OP_if_true: case OP_if_false: {
            int rel, target; sljit_s32 ct;
            struct sljit_jump *j_ti, *j_tb, *j_have;
            if (op == OP_if_true || op == OP_if_false) rel = (int)(int32_t)get_u32(p + 1);
            else rel = (int)(int8_t)p[1];
            target = ooff + 1 + rel;
            ct = (op == OP_if_true || op == OP_if_true8) ? SLJIT_NOT_EQUAL : SLJIT_EQUAL;

            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            j_ti = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_INT);
            j_tb = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, JS_TAG_BOOL);
            sljit_set_label(j_ti, sljit_emit_label(c));
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            j_have = sljit_emit_jump(c, SLJIT_JUMP);

            sljit_set_label(j_tb, sljit_emit_label(c));
            jit_emit_sync_call(c, (void *)&jit_helper_to_bool, 0, 0, 0, 0, 1, fx, &nfx);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_RETURN_REG, 0);

            sljit_set_label(j_have, sljit_emit_label(c));
            j = sljit_emit_cmp(c, ct, SLJIT_R0, 0, SLJIT_IMM, 0);
            fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BRANCH; fx[nfx].target_off = target; nfx++;
            break;
        }
        /* ---- return ---- */
        case OP_return: {
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_JCTX), JCTX_PRET);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_U);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(SLJIT_R0), JSV_U, SLJIT_R1, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(JREG_SP), -JSV_SIZE + JSV_TAG);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(SLJIT_R0), JSV_TAG, SLJIT_R1, 0);
            sljit_emit_op2(c, SLJIT_SUB, JREG_SP, 0, JREG_SP, 0, SLJIT_IMM, JSV_SIZE);
            j = sljit_emit_jump(c, SLJIT_JUMP);
            fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_DONE; nfx++;
            break;
        }
        case OP_return_undef: {
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(JREG_JCTX), JCTX_PRET);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(SLJIT_R0), JSV_U, SLJIT_IMM, 0);
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(SLJIT_R0), JSV_TAG, SLJIT_IMM, JS_TAG_UNDEFINED);
            j = sljit_emit_jump(c, SLJIT_JUMP);
            fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_DONE; nfx++;
            break;
        }
        default:
            j = sljit_emit_jump(c, SLJIT_JUMP);
            fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
            break;
        }

        p += short_opcode_info(op).size;
        i++;
    }

    /* unreachable fall-through safety */
    sljit_emit_const(c, SLJIT_MOV, JREG_PC, 0, (sljit_sw)bc);
    j = sljit_emit_jump(c, SLJIT_JUMP);
    fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;

    /* bail epilogue: write sp/pc back, return JS_JIT_BAIL */
    bail_label = sljit_emit_label(c);
    sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_JCTX), JCTX_SP, JREG_SP, 0);
    sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_JCTX), JCTX_PC, JREG_PC, 0);
    sljit_emit_return(c, SLJIT_MOV, SLJIT_IMM, JS_JIT_BAIL);

    /* exc epilogue: unwind stack, set *pret_val = JS_EXCEPTION, return JS_JIT_DONE */
    exc_label = sljit_emit_label(c);
    sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_JCTX), JCTX_SP, JREG_SP, 0);
    sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, JREG_JCTX, 0);
    sljit_emit_icall(c, SLJIT_CALL, SLJIT_ARGS1V(P), SLJIT_IMM, (sljit_sw)&jit_helper_unwind);
    sljit_emit_return(c, SLJIT_MOV, SLJIT_IMM, JS_JIT_DONE);

    /* done epilogue: write sp back, return JS_JIT_DONE */
    done_label = sljit_emit_label(c);
    sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(JREG_JCTX), JCTX_SP, JREG_SP, 0);
    sljit_emit_return(c, SLJIT_MOV, SLJIT_IMM, JS_JIT_DONE);

    /* resolve all deferred jumps */
    for (i = 0; i < nfx; i++) {
        if (fx[i].kind == JIT_FX_BAIL)
            sljit_set_label(fx[i].jmp, bail_label);
        else if (fx[i].kind == JIT_FX_EXC)
            sljit_set_label(fx[i].jmp, exc_label);
        else if (fx[i].kind == JIT_FX_DONE)
            sljit_set_label(fx[i].jmp, done_label);
        else {
            int li = jit_find_label_idx(offs, nops, fx[i].target_off);
            if (li >= 0)
                sljit_set_label(fx[i].jmp, labels[li]);
            else
                sljit_set_label(fx[i].jmp, bail_label);
        }
    }

    if (sljit_get_compiler_error(c)) {
        sljit_free_compiler(c);
        js_free_rt(ctx->rt, ics);
        js_free(ctx, offs); js_free(ctx, labels); js_free(ctx, fx); js_free(ctx, is_target);
        return 0;
    }
    code = sljit_generate_code(c, 0, NULL);
    sljit_free_compiler(c);
    js_free(ctx, offs); js_free(ctx, labels); js_free(ctx, fx); js_free(ctx, is_target);
    if (!code) {
        js_free_rt(ctx->rt, ics);
        return 0;
    }
    e->code = code;
    e->ic_data = ics;
    e->ic_count = (uint32_t)nic;
    e->status = JS_JIT_STATUS_COMPILED;
    return 1;
}

/* ================================================================== */
/* Tier 1 Native Worker (Unboxed Register-Resident SLJIT Codegen)      */
/* ================================================================== */

#define JIT_NAT_MAX_ARGS  3
#define JIT_NAT_MAX_VARS  32
#define JIT_NAT_MAX_STACK 32

enum {
    VK_IMM = 0,
    VK_LOC,
    VK_ARG,
    VK_STK,
    VK_SELF_FN
};

enum {
    VTYPE_INT = 0,
    VTYPE_BOOL,
    VTYPE_FN
};

struct JitVSlot {
    int kind;
    int type;
    int32_t imm;
    int idx;
};

struct JitNatLayout {
    int argc;
    int varc;
    int max_stk;
    int reg_var_start;   /* first SLJIT_S reg index for vars (1+argc .. 4) */
    int reg_var_count;   /* how many vars live in SLJIT_S regs */
    int local_size;      /* SLJIT frame bytes */
};

static int jit_nat_var_is_reg(const struct JitNatLayout *L, int vidx,
                              sljit_s32 *preg, sljit_sw *poff)
{
    if (vidx < L->reg_var_count) {
        *preg = (sljit_s32)(SLJIT_S(L->reg_var_start + vidx));
        *poff = 0;
        return 1;
    }
    *preg = SLJIT_SP;
    *poff = (sljit_sw)((vidx - L->reg_var_count) * 8);
    return 0;
}

static int jit_nat_arg_is_reg(const struct JitNatLayout *L, int aidx,
                              sljit_s32 *preg, sljit_sw *poff)
{
    (void)L;
    *preg = (sljit_s32)(SLJIT_S(1 + aidx));
    *poff = 0;
    return 1;
}

static sljit_sw jit_nat_stk_off(const struct JitNatLayout *L, int d)
{
    int mem_vars = (L->varc > L->reg_var_count) ? (L->varc - L->reg_var_count) : 0;
    return (sljit_sw)((mem_vars + d) * 8);
}

static void jit_nat_vslot_operand(const struct JitNatLayout *L,
                                  const struct JitVSlot *v,
                                  int d, sljit_s32 *pr, sljit_sw *pw)
{
    sljit_s32 r;
    sljit_sw off;
    if (v->kind == VK_IMM) {
        *pr = SLJIT_IMM;
        *pw = (sljit_sw)v->imm;
    } else if (v->kind == VK_LOC) {
        if (jit_nat_var_is_reg(L, v->idx, &r, &off)) {
            *pr = r; *pw = 0;
        } else {
            *pr = SLJIT_MEM1(r); *pw = off;
        }
    } else if (v->kind == VK_ARG) {
        jit_nat_arg_is_reg(L, v->idx, &r, &off);
        *pr = r; *pw = 0;
    } else {
        *pr = SLJIT_MEM1(SLJIT_SP);
        *pw = jit_nat_stk_off(L, d);
    }
}

static void jit_nat_load_reg(struct sljit_compiler *c, const struct JitNatLayout *L,
                             const struct JitVSlot *v, int d, sljit_s32 dst)
{
    sljit_s32 r;
    sljit_sw w;
    jit_nat_vslot_operand(L, v, d, &r, &w);
    if (r == dst && w == 0)
        return;
    sljit_emit_op1(c, SLJIT_MOV_S32, dst, 0, r, w);
}

static void jit_nat_spill_slot(struct sljit_compiler *c, const struct JitNatLayout *L,
                               struct JitVSlot *v, int d)
{
    if (v->kind == VK_STK)
        return;
    sljit_s32 r;
    sljit_sw w;
    jit_nat_vslot_operand(L, v, d, &r, &w);
    sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_MEM1(SLJIT_SP), jit_nat_stk_off(L, d), r, w);
    v->kind = VK_STK;
}

static void jit_nat_spill_all(struct sljit_compiler *c, const struct JitNatLayout *L,
                              struct JitVSlot *vstk, int vsp)
{
    for (int d = 0; d < vsp; d++)
        jit_nat_spill_slot(c, L, &vstk[d], d);
}

static void jit_nat_materialize_alias(struct sljit_compiler *c, const struct JitNatLayout *L,
                                      struct JitVSlot *vstk, int vsp, int kind, int idx)
{
    for (int d = 0; d < vsp; d++) {
        if (vstk[d].kind == kind && vstk[d].idx == idx)
            jit_nat_spill_slot(c, L, &vstk[d], d);
    }
}

static int jit_nat_op_stack_delta(const JSFunctionBytecode *b, const uint8_t *p,
                                  int *pop_cnt, int *push_cnt, int *out_target,
                                  int ooff, int *is_uncond, int *is_branch,
                                  int *out_self_rec)
{
    int op = *p;
    *pop_cnt = 0; *push_cnt = 0; *out_target = -1;
    *is_uncond = 0; *is_branch = 0;

    switch (op) {
    case OP_push_0: case OP_push_1: case OP_push_2: case OP_push_3:
    case OP_push_4: case OP_push_5: case OP_push_6: case OP_push_7:
    case OP_push_minus1:
    case OP_push_i8: case OP_push_i16: case OP_push_i32:
    case OP_push_false: case OP_push_true:
        *push_cnt = 1;
        return 1;
    case OP_push_const8:
        if (JS_VALUE_GET_TAG(b->cpool[p[1]]) != JS_TAG_INT) return 0;
        *push_cnt = 1;
        return 1;
    case OP_push_const:
        if (JS_VALUE_GET_TAG(b->cpool[get_u32(p + 1)]) != JS_TAG_INT) return 0;
        *push_cnt = 1;
        return 1;
    case OP_get_loc0_loc1:
        *push_cnt = 2;
        return 1;
    case OP_get_loc0: case OP_get_loc1: case OP_get_loc2: case OP_get_loc3:
    case OP_set_loc_uninitialized:
        return 1;
    case OP_get_loc8: case OP_get_loc: case OP_get_loc_check:
    case OP_get_arg0: case OP_get_arg1: case OP_get_arg2: case OP_get_arg3:
    case OP_get_arg:
        *push_cnt = 1;
        return 1;
    case OP_put_loc0: case OP_put_loc1: case OP_put_loc2: case OP_put_loc3:
    case OP_put_loc8: case OP_put_loc:
    case OP_put_loc_check: case OP_put_loc_check_init:
    case OP_put_arg0: case OP_put_arg1: case OP_put_arg2: case OP_put_arg3:
    case OP_put_arg:
        *pop_cnt = 1;
        return 1;
    case OP_set_loc0: case OP_set_loc1: case OP_set_loc2: case OP_set_loc3:
    case OP_set_loc8: case OP_set_loc:
    case OP_set_arg0: case OP_set_arg1: case OP_set_arg2: case OP_set_arg3:
    case OP_set_arg:
        *pop_cnt = 1; *push_cnt = 1;
        return 1;
    case OP_inc_loc: case OP_dec_loc:
        return 1;
    case OP_add_loc:
        *pop_cnt = 1;
        return 1;
    case OP_dup:
        *pop_cnt = 1; *push_cnt = 2;
        return 1;
    case OP_drop:
        *pop_cnt = 1;
        return 1;
    case OP_swap:
        *pop_cnt = 2; *push_cnt = 2;
        return 1;
    case OP_inc: case OP_dec: case OP_neg: case OP_plus: case OP_not: case OP_lnot:
        *pop_cnt = 1; *push_cnt = 1;
        return 1;
    case OP_post_inc: case OP_post_dec:
        *pop_cnt = 1; *push_cnt = 2;
        return 1;
    case OP_add: case OP_sub: case OP_mul: case OP_div: case OP_mod:
    case OP_shl: case OP_sar: case OP_shr: case OP_and: case OP_or: case OP_xor:
    case OP_lt: case OP_lte: case OP_gt: case OP_gte:
    case OP_eq: case OP_neq: case OP_strict_eq: case OP_strict_neq:
        *pop_cnt = 2; *push_cnt = 1;
        return 1;
    case OP_goto: case OP_goto8: case OP_goto16: {
        int rel = (op == OP_goto) ? (int)(int32_t)get_u32(p + 1) :
                  (op == OP_goto16) ? (int)(int16_t)get_u16(p + 1) : (int)(int8_t)p[1];
        *out_target = ooff + 1 + rel;
        *is_uncond = 1; *is_branch = 1;
        return 1;
    }
    case OP_if_true: case OP_if_false: case OP_if_true8: case OP_if_false8: {
        int rel = (op == OP_if_true || op == OP_if_false) ? (int)(int32_t)get_u32(p + 1) : (int)(int8_t)p[1];
        *out_target = ooff + 1 + rel;
        *pop_cnt = 1; *is_branch = 1;
        return 1;
    }
    case OP_get_var: {
        JSAtom atom = (JSAtom)get_u32(p + 1);
        if (b->func_name == JS_ATOM_NULL || atom != b->func_name)
            return 0;
        *push_cnt = 1;
        return 1;
    }
    case OP_call0: case OP_call1: case OP_call2: case OP_call3: {
        int argc = op - OP_call0;
        if (argc != b->arg_count || argc > JIT_NAT_MAX_ARGS)
            return 0;
        *pop_cnt = argc + 1;
        *push_cnt = 1;
        *out_self_rec = 1;
        return 1;
    }
    case OP_call: {
        int argc = (int)get_u16(p + 1);
        if (argc != b->arg_count || argc > JIT_NAT_MAX_ARGS)
            return 0;
        *pop_cnt = argc + 1;
        *push_cnt = 1;
        *out_self_rec = 1;
        return 1;
    }
    case OP_return:
        *pop_cnt = 1;
        *is_uncond = 1;
        return 1;
    case OP_nop:
        return 1;
    default:
        if (jit_trace_level >= 1)
            fprintf(stderr, "[jit-nat] unsupported op=%d at off=%d\n", op, ooff);
        return 0;
    }
}

static int jit_native_analyze(JSContext *ctx, JSFunctionBytecode *b,
                              const int *offs, int nops,
                              int *stk_depth, uint8_t *is_target,
                              int *max_stk_out, int *self_rec_out)
{
    int *q = NULL;
    int qh = 0, qt = 0;
    const uint8_t *bc = b->byte_code_buf;
    int max_stk = 0;
    int self_rec = 0;

    if (b->arg_count > JIT_NAT_MAX_ARGS || b->var_count > JIT_NAT_MAX_VARS ||
        b->stack_size > JIT_NAT_MAX_STACK || nops <= 0)
        return 0;

    for (int i = 0; i < nops; i++) {
        stk_depth[i] = -1;
        is_target[i] = 0;
    }

    q = js_mallocz(ctx, sizeof(int) * (nops + 4));
    if (!q) return 0;

    stk_depth[0] = 0;
    q[qt++] = 0;

    while (qh < qt) {
        int idx = q[qh++];
        int d = stk_depth[idx];
        const uint8_t *p = bc + offs[idx];
        int pop_c, push_c, target_off, is_uncond, is_branch;

        if (!jit_nat_op_stack_delta(b, p, &pop_c, &push_c, &target_off,
                                    offs[idx], &is_uncond, &is_branch, &self_rec)) {
            js_free(ctx, q);
            return 0;
        }
        if (d < pop_c) {
            js_free(ctx, q);
            return 0;
        }
        int nd = d - pop_c + push_c;
        if (nd > JIT_NAT_MAX_STACK) {
            js_free(ctx, q);
            return 0;
        }
        if (d > max_stk) max_stk = d;
        if (nd > max_stk) max_stk = nd;

        /* Also verify local/arg indices are within declared bounds */
        int op = *p;
        if (op == OP_get_loc8 || op == OP_put_loc8 || op == OP_set_loc8 ||
            op == OP_inc_loc || op == OP_dec_loc || op == OP_add_loc) {
            if (p[1] >= b->var_count) { js_free(ctx, q); return 0; }
        } else if (op == OP_get_loc || op == OP_put_loc || op == OP_set_loc) {
            if ((int)get_u16(p + 1) >= b->var_count) { js_free(ctx, q); return 0; }
        } else if (op == OP_get_loc0_loc1) {
            if (2 > b->var_count) { js_free(ctx, q); return 0; }
        } else if (op == OP_get_loc0 || op == OP_put_loc0 || op == OP_set_loc0) {
            if (0 >= b->var_count) { js_free(ctx, q); return 0; }
        } else if (op == OP_get_loc1 || op == OP_put_loc1 || op == OP_set_loc1) {
            if (1 >= b->var_count) { js_free(ctx, q); return 0; }
        } else if (op == OP_get_loc2 || op == OP_put_loc2 || op == OP_set_loc2) {
            if (2 >= b->var_count) { js_free(ctx, q); return 0; }
        } else if (op == OP_get_loc3 || op == OP_put_loc3 || op == OP_set_loc3) {
            if (3 >= b->var_count) { js_free(ctx, q); return 0; }
        } else if (op == OP_get_arg || op == OP_put_arg || op == OP_set_arg) {
            if ((int)get_u16(p + 1) >= b->arg_count) { js_free(ctx, q); return 0; }
        } else if (op >= OP_get_arg0 && op <= OP_set_arg3) {
            int ai = (op - OP_get_arg0) % 4;
            if (ai >= b->arg_count) { js_free(ctx, q); return 0; }
        }

        if (is_branch) {
            int ti;
            if (nd > 0) { js_free(ctx, q); return 0; }
            ti = jit_find_label_idx(offs, nops, target_off);
            if (ti < 0) { js_free(ctx, q); return 0; }
            is_target[ti] = 1;
            if (stk_depth[ti] == -1) {
                stk_depth[ti] = nd;
                q[qt++] = ti;
            } else if (stk_depth[ti] != nd) {
                js_free(ctx, q);
                return 0;
            }
        }
        if (!is_uncond) {
            int ni = idx + 1;
            if (ni >= nops) { js_free(ctx, q); return 0; }
            if (stk_depth[ni] == -1) {
                stk_depth[ni] = nd;
                q[qt++] = ni;
            } else if (stk_depth[ni] != nd) {
                js_free(ctx, q);
                return 0;
            }
        }
    }

    js_free(ctx, q);
    *max_stk_out = max_stk;
    *self_rec_out = self_rec;
    return 1;
}

static sljit_s32 jit_nat_cmp_cond(int op, int invert)
{
    if (!invert) {
        switch (op) {
        case OP_lt:  return SLJIT_SIG_LESS;
        case OP_lte: return SLJIT_SIG_LESS_EQUAL;
        case OP_gt:  return SLJIT_SIG_GREATER;
        case OP_gte: return SLJIT_SIG_GREATER_EQUAL;
        case OP_eq: case OP_strict_eq: return SLJIT_EQUAL;
        default:     return SLJIT_NOT_EQUAL;
        }
    } else {
        switch (op) {
        case OP_lt:  return SLJIT_SIG_GREATER_EQUAL;
        case OP_lte: return SLJIT_SIG_GREATER;
        case OP_gt:  return SLJIT_SIG_LESS_EQUAL;
        case OP_gte: return SLJIT_SIG_LESS;
        case OP_eq: case OP_strict_eq: return SLJIT_NOT_EQUAL;
        default:     return SLJIT_EQUAL;
        }
    }
}

static int jit_native_codegen(JSContext *ctx, JSFunctionBytecode *b,
                              struct JSJITEntry *e)
{
    int *offs = NULL, *stk_depth = NULL;
    uint8_t *is_target = NULL;
    struct sljit_label **labels = NULL;
    struct jit_fixup *fx = NULL;
    struct sljit_compiler *c = NULL;
    struct sljit_label *bail_label;
    struct sljit_jump *j;
    const uint8_t *bc = b->byte_code_buf;
    const uint8_t *p = bc, *end = bc + b->byte_code_len;
    int nops = 0, nfx = 0, max_stk = 0, self_rec = 0;

    if (b->byte_code_len <= 0)
        return 0;

    offs = js_mallocz(ctx, sizeof(int) * (b->byte_code_len + 1));
    if (!offs) return 0;
    while (p < end) {
        int op = *p;
        offs[nops++] = (int)(p - bc);
        p += short_opcode_info(op).size;
    }

    stk_depth = js_mallocz(ctx, sizeof(int) * (nops + 1));
    is_target = js_mallocz(ctx, sizeof(uint8_t) * (nops + 1));
    labels = js_mallocz(ctx, sizeof(struct sljit_label *) * (nops + 1));
    fx = js_mallocz(ctx, sizeof(struct jit_fixup) * (nops * 3 + 16));
    if (!stk_depth || !is_target || !labels || !fx) {
        js_free(ctx, offs); js_free(ctx, stk_depth); js_free(ctx, is_target);
        js_free(ctx, labels); js_free(ctx, fx);
        return 0;
    }

    if (!jit_native_analyze(ctx, b, offs, nops, stk_depth, is_target,
                            &max_stk, &self_rec)) {
        js_free(ctx, offs); js_free(ctx, stk_depth); js_free(ctx, is_target);
        js_free(ctx, labels); js_free(ctx, fx);
        return 0;
    }

    struct JitNatLayout L;
    L.argc = b->arg_count;
    L.varc = b->var_count;
    L.max_stk = max_stk;
    L.reg_var_start = 1 + L.argc; /* S0=ctx, S1..S(argc)=args */
    L.reg_var_count = (4 - L.argc > 0) ? (4 - L.argc) : 0;
    if (L.reg_var_count > L.varc) L.reg_var_count = L.varc;
    int mem_vars = (L.varc > L.reg_var_count) ? (L.varc - L.reg_var_count) : 0;
    int frame_words = mem_vars + max_stk + 2;
    L.local_size = ((frame_words * 8) + 15) & ~15;

    c = sljit_create_compiler(NULL);
    if (!c) {
        js_free(ctx, offs); js_free(ctx, stk_depth); js_free(ctx, is_target);
        js_free(ctx, labels); js_free(ctx, fx);
        return 0;
    }

    /* Native Worker: SLJIT_ARGS4(W, P, W, W, W) -> (ctx, a0, a1, a2) */
    sljit_emit_enter(c, 0, SLJIT_ARGS4(W, P, W, W, W), 4, 5, L.local_size);

    /* Initialize var slots to 0 so no garbage exists on native stack */
    for (int v = 0; v < L.varc; v++) {
        sljit_s32 vr; sljit_sw voff;
        if (jit_nat_var_is_reg(&L, v, &vr, &voff))
            sljit_emit_op1(c, SLJIT_MOV, vr, 0, SLJIT_IMM, 0);
        else
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_MEM1(vr), voff, SLJIT_IMM, 0);
    }

    struct JitVSlot vstk[JIT_NAT_MAX_STACK];
    int vsp = 0;

    for (int i = 0; i < nops; i++) {
        p = bc + offs[i];
        int op = *p;
        int ooff = offs[i];

        labels[i] = sljit_emit_label(c);

        if (stk_depth[i] == -1) {
            /* Unreachable dead bytecode */
            continue;
        }
        if (is_target[i]) {
            vsp = stk_depth[i];
            for (int d = 0; d < vsp; d++) {
                vstk[d].kind = VK_STK;
                vstk[d].type = VTYPE_INT;
                vstk[d].imm = 0;
                vstk[d].idx = d;
            }
        }

        switch (op) {
        case OP_push_0: case OP_push_1: case OP_push_2: case OP_push_3:
        case OP_push_4: case OP_push_5: case OP_push_6: case OP_push_7:
            vstk[vsp].kind = VK_IMM; vstk[vsp].type = VTYPE_INT;
            vstk[vsp].imm = (int32_t)(op - OP_push_0); vsp++;
            break;
        case OP_push_minus1:
            vstk[vsp].kind = VK_IMM; vstk[vsp].type = VTYPE_INT;
            vstk[vsp].imm = -1; vsp++;
            break;
        case OP_push_i8:
            vstk[vsp].kind = VK_IMM; vstk[vsp].type = VTYPE_INT;
            vstk[vsp].imm = (int32_t)get_i8(p + 1); vsp++;
            break;
        case OP_push_i16:
            vstk[vsp].kind = VK_IMM; vstk[vsp].type = VTYPE_INT;
            vstk[vsp].imm = (int32_t)get_i16(p + 1); vsp++;
            break;
        case OP_push_i32:
            vstk[vsp].kind = VK_IMM; vstk[vsp].type = VTYPE_INT;
            vstk[vsp].imm = (int32_t)get_i32(p + 1); vsp++;
            break;
        case OP_push_const8:
            vstk[vsp].kind = VK_IMM; vstk[vsp].type = VTYPE_INT;
            vstk[vsp].imm = JS_VALUE_GET_INT(b->cpool[p[1]]); vsp++;
            break;
        case OP_push_const:
            vstk[vsp].kind = VK_IMM; vstk[vsp].type = VTYPE_INT;
            vstk[vsp].imm = JS_VALUE_GET_INT(b->cpool[get_u32(p + 1)]); vsp++;
            break;
        case OP_push_false:
            vstk[vsp].kind = VK_IMM; vstk[vsp].type = VTYPE_BOOL;
            vstk[vsp].imm = 0; vsp++;
            break;
        case OP_push_true:
            vstk[vsp].kind = VK_IMM; vstk[vsp].type = VTYPE_BOOL;
            vstk[vsp].imm = 1; vsp++;
            break;

        case OP_get_loc0_loc1:
            vstk[vsp].kind = VK_LOC; vstk[vsp].type = VTYPE_INT;
            vstk[vsp].idx = 0; vsp++;
            vstk[vsp].kind = VK_LOC; vstk[vsp].type = VTYPE_INT;
            vstk[vsp].idx = 1; vsp++;
            break;
        case OP_get_loc0: case OP_get_loc1: case OP_get_loc2: case OP_get_loc3:
            vstk[vsp].kind = VK_LOC; vstk[vsp].type = VTYPE_INT;
            vstk[vsp].idx = op - OP_get_loc0; vsp++;
            break;
        case OP_get_loc8:
            vstk[vsp].kind = VK_LOC; vstk[vsp].type = VTYPE_INT;
            vstk[vsp].idx = p[1]; vsp++;
            break;
        case OP_set_loc_uninitialized:
            break;
        case OP_get_loc: case OP_get_loc_check:
            vstk[vsp].kind = VK_LOC; vstk[vsp].type = VTYPE_INT;
            vstk[vsp].idx = (int)get_u16(p + 1); vsp++;
            break;
        case OP_get_arg0: case OP_get_arg1: case OP_get_arg2: case OP_get_arg3:
            vstk[vsp].kind = VK_ARG; vstk[vsp].type = VTYPE_INT;
            vstk[vsp].idx = op - OP_get_arg0; vsp++;
            break;
        case OP_get_arg:
            vstk[vsp].kind = VK_ARG; vstk[vsp].type = VTYPE_INT;
            vstk[vsp].idx = (int)get_u16(p + 1); vsp++;
            break;

        case OP_put_loc0: case OP_put_loc1: case OP_put_loc2: case OP_put_loc3:
        case OP_put_loc8: case OP_put_loc:
        case OP_put_loc_check: case OP_put_loc_check_init:
        case OP_set_loc0: case OP_set_loc1: case OP_set_loc2: case OP_set_loc3:
        case OP_set_loc8: case OP_set_loc: {
            int is_set = (op >= OP_set_loc0 && op <= OP_set_loc3) ||
                         op == OP_set_loc8 || op == OP_set_loc;
            int idx = (op == OP_put_loc8 || op == OP_set_loc8) ? p[1] :
                      (op == OP_put_loc || op == OP_set_loc ||
                       op == OP_put_loc_check || op == OP_put_loc_check_init) ? (int)get_u16(p + 1) :
                      (op >= OP_set_loc0 && op <= OP_set_loc3) ? (op - OP_set_loc0) :
                      (op - OP_put_loc0);
            struct JitVSlot val = vstk[vsp - 1];
            int pop_d = vsp - 1;
            if (!is_set) vsp--;
            jit_nat_materialize_alias(c, &L, vstk, vsp, VK_LOC, idx);
            sljit_s32 vr, sr; sljit_sw voff, sw;
            jit_nat_vslot_operand(&L, &val, pop_d, &sr, &sw);
            if (jit_nat_var_is_reg(&L, idx, &vr, &voff)) {
                sljit_emit_op1(c, SLJIT_MOV_S32, vr, 0, sr, sw);
            } else {
                if (sr == SLJIT_MEM1(SLJIT_SP)) {
                    sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, sr, sw);
                    sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_MEM1(vr), voff, SLJIT_R0, 0);
                } else {
                    sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_MEM1(vr), voff, sr, sw);
                }
            }
            if (is_set) {
                vstk[vsp - 1].kind = VK_LOC;
                vstk[vsp - 1].type = VTYPE_INT;
                vstk[vsp - 1].idx = idx;
            }
            break;
        }
        case OP_put_arg0: case OP_put_arg1: case OP_put_arg2: case OP_put_arg3:
        case OP_put_arg:
        case OP_set_arg0: case OP_set_arg1: case OP_set_arg2: case OP_set_arg3:
        case OP_set_arg: {
            int is_set = (op >= OP_set_arg0 && op <= OP_set_arg3) || op == OP_set_arg;
            int idx = (op == OP_put_arg || op == OP_set_arg) ? (int)get_u16(p + 1) :
                      (op >= OP_set_arg0 && op <= OP_set_arg3) ? (op - OP_set_arg0) :
                      (op - OP_put_arg0);
            struct JitVSlot val = vstk[vsp - 1];
            int pop_d = vsp - 1;
            if (!is_set) vsp--;
            jit_nat_materialize_alias(c, &L, vstk, vsp, VK_ARG, idx);
            sljit_s32 ar, sr; sljit_sw aoff, sw;
            jit_nat_arg_is_reg(&L, idx, &ar, &aoff);
            jit_nat_vslot_operand(&L, &val, pop_d, &sr, &sw);
            sljit_emit_op1(c, SLJIT_MOV_S32, ar, 0, sr, sw);
            if (is_set) {
                vstk[vsp - 1].kind = VK_ARG;
                vstk[vsp - 1].type = VTYPE_INT;
                vstk[vsp - 1].idx = idx;
            }
            break;
        }

        case OP_inc_loc: case OP_dec_loc: {
            int idx = p[1];
            sljit_s32 aop = (op == OP_inc_loc) ? SLJIT_ADD : SLJIT_SUB;
            jit_nat_materialize_alias(c, &L, vstk, vsp, VK_LOC, idx);
            sljit_s32 vr; sljit_sw voff;
            if (jit_nat_var_is_reg(&L, idx, &vr, &voff)) {
                sljit_emit_op2(c, aop, SLJIT_R0, 0, vr, 0, SLJIT_IMM, 1);
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_R0, 0);
                j = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_R1, 0);
                fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
                sljit_emit_op1(c, SLJIT_MOV_S32, vr, 0, SLJIT_R0, 0);
            } else {
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_MEM1(vr), voff);
                sljit_emit_op2(c, aop, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_IMM, 1);
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_R0, 0);
                j = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_R1, 0);
                fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_MEM1(vr), voff, SLJIT_R0, 0);
            }
            break;
        }
        case OP_add_loc: {
            int idx = p[1];
            struct JitVSlot rhs = vstk[--vsp];
            jit_nat_materialize_alias(c, &L, vstk, vsp, VK_LOC, idx);
            sljit_s32 vr, r2; sljit_sw voff, w2;
            jit_nat_vslot_operand(&L, &rhs, vsp, &r2, &w2);
            if (jit_nat_var_is_reg(&L, idx, &vr, &voff)) {
                sljit_emit_op2(c, SLJIT_ADD, SLJIT_R0, 0, vr, 0, r2, w2);
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_R0, 0);
                j = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_R1, 0);
                fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
                sljit_emit_op1(c, SLJIT_MOV_S32, vr, 0, SLJIT_R0, 0);
            } else {
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_MEM1(vr), voff);
                sljit_emit_op2(c, SLJIT_ADD, SLJIT_R0, 0, SLJIT_R0, 0, r2, w2);
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_R0, 0);
                j = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_R1, 0);
                fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_MEM1(vr), voff, SLJIT_R0, 0);
            }
            break;
        }

        case OP_dup:
            jit_nat_spill_slot(c, &L, &vstk[vsp - 1], vsp - 1);
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0,
                           SLJIT_MEM1(SLJIT_SP), jit_nat_stk_off(&L, vsp - 1));
            sljit_emit_op1(c, SLJIT_MOV_S32,
                           SLJIT_MEM1(SLJIT_SP), jit_nat_stk_off(&L, vsp), SLJIT_R0, 0);
            vstk[vsp] = vstk[vsp - 1];
            vsp++;
            break;
        case OP_drop:
            vsp--;
            break;
        case OP_swap: {
            jit_nat_spill_slot(c, &L, &vstk[vsp - 2], vsp - 2);
            jit_nat_spill_slot(c, &L, &vstk[vsp - 1], vsp - 1);
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0,
                           SLJIT_MEM1(SLJIT_SP), jit_nat_stk_off(&L, vsp - 2));
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0,
                           SLJIT_MEM1(SLJIT_SP), jit_nat_stk_off(&L, vsp - 1));
            sljit_emit_op1(c, SLJIT_MOV_S32,
                           SLJIT_MEM1(SLJIT_SP), jit_nat_stk_off(&L, vsp - 2), SLJIT_R1, 0);
            sljit_emit_op1(c, SLJIT_MOV_S32,
                           SLJIT_MEM1(SLJIT_SP), jit_nat_stk_off(&L, vsp - 1), SLJIT_R0, 0);
            break;
        }

        case OP_post_inc: case OP_post_dec: {
            int d = vsp - 1;
            jit_nat_spill_slot(c, &L, &vstk[d], d);
            jit_nat_load_reg(c, &L, &vstk[d], d, SLJIT_R0);
            sljit_emit_op2(c, (op == OP_post_inc) ? SLJIT_ADD : SLJIT_SUB,
                           SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_IMM, 1);
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_R0, 0);
            j = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_R1, 0);
            fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
            sljit_emit_op1(c, SLJIT_MOV_S32,
                           SLJIT_MEM1(SLJIT_SP), jit_nat_stk_off(&L, vsp), SLJIT_R0, 0);
            vstk[vsp].kind = VK_STK;
            vstk[vsp].type = VTYPE_INT;
            vsp++;
            break;
        }
        case OP_inc: case OP_dec: case OP_neg: case OP_plus: case OP_not: case OP_lnot: {
            int d = vsp - 1;
            jit_nat_load_reg(c, &L, &vstk[d], d, SLJIT_R0);
            if (op == OP_inc || op == OP_dec) {
                sljit_emit_op2(c, (op == OP_inc) ? SLJIT_ADD : SLJIT_SUB,
                               SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_IMM, 1);
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_R0, 0);
                j = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_R1, 0);
                fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
            } else if (op == OP_neg) {
                /* guard 0 (avoid -0.0) and INT32_MIN */
                j = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, 0);
                fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
                sljit_emit_op2(c, SLJIT_SUB, SLJIT_R0, 0, SLJIT_IMM, 0, SLJIT_R0, 0);
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_R0, 0);
                j = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_R1, 0);
                fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
            } else if (op == OP_not) {
                sljit_emit_op2(c, SLJIT_XOR, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_IMM, -1);
            } else if (op == OP_lnot) {
                sljit_emit_op2u(c, SLJIT_SUB | SLJIT_SET_Z, SLJIT_R0, 0, SLJIT_IMM, 0);
                sljit_emit_op_flags(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_EQUAL);
            }
            sljit_emit_op1(c, SLJIT_MOV_S32,
                           SLJIT_MEM1(SLJIT_SP), jit_nat_stk_off(&L, d), SLJIT_R0, 0);
            vstk[d].kind = VK_STK;
            vstk[d].type = (op == OP_lnot) ? VTYPE_BOOL : VTYPE_INT;
            break;
        }

        case OP_add: case OP_sub: case OP_mul: {
            int d1 = vsp - 2, d2 = vsp - 1;
            sljit_s32 r1, r2, aop = (op == OP_add) ? SLJIT_ADD : (op == OP_sub) ? SLJIT_SUB : SLJIT_MUL;
            sljit_sw w1, w2;
            jit_nat_vslot_operand(&L, &vstk[d1], d1, &r1, &w1);
            jit_nat_vslot_operand(&L, &vstk[d2], d2, &r2, &w2);
            if ((r1 & SLJIT_MEM) && (r2 & SLJIT_MEM)) {
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, r1, w1);
                r1 = SLJIT_R0; w1 = 0;
            }
            if (op == OP_mul)
                sljit_emit_op2(c, SLJIT_OR | SLJIT_32, SLJIT_R2, 0, r1, w1, r2, w2);
            sljit_emit_op2(c, aop, SLJIT_R0, 0, r1, w1, r2, w2);
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R1, 0, SLJIT_R0, 0);
            if (!(op != OP_mul && i + 2 < nops &&
                  bc[offs[i + 1]] == OP_push_0 && bc[offs[i + 2]] == OP_or)) {
                j = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_R1, 0);
                fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
            } else {
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_R1, 0);
            }
            if (op == OP_mul) {
                struct sljit_jump *j_nz = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, 0);
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R2, 0, SLJIT_R2, 0);
                j = sljit_emit_cmp(c, SLJIT_SIG_LESS, SLJIT_R2, 0, SLJIT_IMM, 0);
                fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
                sljit_set_label(j_nz, sljit_emit_label(c));
            }
            sljit_emit_op1(c, SLJIT_MOV_S32,
                           SLJIT_MEM1(SLJIT_SP), jit_nat_stk_off(&L, d1), SLJIT_R0, 0);
            vsp--;
            vstk[d1].kind = VK_STK;
            vstk[d1].type = VTYPE_INT;
            break;
        }

        case OP_div: case OP_mod: {
            int d1 = vsp - 2, d2 = vsp - 1;
            jit_nat_load_reg(c, &L, &vstk[d1], d1, SLJIT_R0);
            jit_nat_load_reg(c, &L, &vstk[d2], d2, SLJIT_R1);
            /* guard divisor == 0 */
            j = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, 0);
            fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
            /* guard dividend <= 0 for mod/div simplicity if INT32_MIN or -0 */
            if (op == OP_mod) {
                j = sljit_emit_cmp(c, SLJIT_SIG_LESS, SLJIT_R0, 0, SLJIT_IMM, 0);
                fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
                j = sljit_emit_cmp(c, SLJIT_SIG_LESS, SLJIT_R1, 0, SLJIT_IMM, 0);
                fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
            } else {
                /* guard INT32_MIN / -1 */
                j = sljit_emit_cmp(c, SLJIT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, (sljit_sw)INT32_MIN);
                fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
            }
            sljit_emit_op0(c, SLJIT_DIVMOD_S32);
            if (op == OP_div) {
                j = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, 0);
                fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
                sljit_emit_op1(c, SLJIT_MOV_S32,
                               SLJIT_MEM1(SLJIT_SP), jit_nat_stk_off(&L, d1), SLJIT_R0, 0);
            } else {
                sljit_emit_op1(c, SLJIT_MOV_S32,
                               SLJIT_MEM1(SLJIT_SP), jit_nat_stk_off(&L, d1), SLJIT_R1, 0);
            }
            vsp--;
            vstk[d1].kind = VK_STK;
            vstk[d1].type = VTYPE_INT;
            break;
        }

        case OP_shl: case OP_sar: case OP_shr:
        case OP_and: case OP_or: case OP_xor: {
            int d1 = vsp - 2, d2 = vsp - 1;
            sljit_s32 r1, r2; sljit_sw w1, w2;
            jit_nat_vslot_operand(&L, &vstk[d1], d1, &r1, &w1);
            jit_nat_vslot_operand(&L, &vstk[d2], d2, &r2, &w2);
            if ((r1 & SLJIT_MEM) && (r2 & SLJIT_MEM)) {
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, r1, w1);
                r1 = SLJIT_R0; w1 = 0;
            }
            if (op == OP_shl || op == OP_sar || op == OP_shr) {
                sljit_emit_op2(c, SLJIT_AND, SLJIT_R1, 0, r2, w2, SLJIT_IMM, 31);
                sljit_s32 sop = (op == OP_shl) ? (SLJIT_SHL | SLJIT_32) :
                                (op == OP_sar) ? (SLJIT_ASHR | SLJIT_32) :
                                                 (SLJIT_LSHR | SLJIT_32);
                sljit_emit_op2(c, sop, SLJIT_R0, 0, r1, w1, SLJIT_R1, 0);
                if (op == OP_shr) {
                    /* unsigned >= 0x80000000 doesn't fit in int32 */
                    sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_R0, 0);
                    j = sljit_emit_cmp(c, SLJIT_SIG_LESS, SLJIT_R0, 0, SLJIT_IMM, 0);
                    fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
                } else {
                    sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_R0, 0);
                }
            } else {
                sljit_s32 bop = (op == OP_and) ? SLJIT_AND : (op == OP_or) ? SLJIT_OR : SLJIT_XOR;
                sljit_emit_op2(c, bop, SLJIT_R0, 0, r1, w1, r2, w2);
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, SLJIT_R0, 0);
            }
            sljit_emit_op1(c, SLJIT_MOV_S32,
                           SLJIT_MEM1(SLJIT_SP), jit_nat_stk_off(&L, d1), SLJIT_R0, 0);
            vsp--;
            vstk[d1].kind = VK_STK;
            vstk[d1].type = VTYPE_INT;
            break;
        }

        case OP_lt: case OP_lte: case OP_gt: case OP_gte:
        case OP_eq: case OP_neq: case OP_strict_eq: case OP_strict_neq: {
            int d1 = vsp - 2, d2 = vsp - 1;
            sljit_s32 r1, r2; sljit_sw w1, w2;
            jit_nat_vslot_operand(&L, &vstk[d1], d1, &r1, &w1);
            jit_nat_vslot_operand(&L, &vstk[d2], d2, &r2, &w2);
            if ((r1 & SLJIT_MEM) && (r2 & SLJIT_MEM)) {
                sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R0, 0, r1, w1);
                r1 = SLJIT_R0; w1 = 0;
            }

            /* Peephole: fuse compare + immediate if_false/if_true into 1 cmp+jmp */
            if (i + 1 < nops && !is_target[i + 1]) {
                const uint8_t *np = bc + offs[i + 1];
                int nop = *np;
                if (nop == OP_if_false || nop == OP_if_false8 ||
                    nop == OP_if_true || nop == OP_if_true8) {
                    int rel = (nop == OP_if_true || nop == OP_if_false)
                              ? (int)(int32_t)get_u32(np + 1) : (int)(int8_t)np[1];
                    int target = offs[i + 1] + 1 + rel;
                    int invert = (nop == OP_if_false || nop == OP_if_false8);
                    vsp -= 2;
                    jit_nat_spill_all(c, &L, vstk, vsp);
                    j = sljit_emit_cmp(c, jit_nat_cmp_cond(op, invert), r1, w1, r2, w2);
                    fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BRANCH; fx[nfx].target_off = target; nfx++;
                    i++; /* skip fused OP_if_* */
                    break;
                }
            }

            /* Non-fused: materialize bool 0/1 */
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R2, 0, SLJIT_IMM, 1);
            j = sljit_emit_cmp(c, jit_nat_cmp_cond(op, 0), r1, w1, r2, w2);
            sljit_emit_op1(c, SLJIT_MOV_S32, SLJIT_R2, 0, SLJIT_IMM, 0);
            sljit_set_label(j, sljit_emit_label(c));
            sljit_emit_op1(c, SLJIT_MOV_S32,
                           SLJIT_MEM1(SLJIT_SP), jit_nat_stk_off(&L, d1), SLJIT_R2, 0);
            vsp--;
            vstk[d1].kind = VK_STK;
            vstk[d1].type = VTYPE_BOOL;
            break;
        }

        case OP_goto: case OP_goto8: case OP_goto16: {
            int rel = (op == OP_goto) ? (int)(int32_t)get_u32(p + 1) :
                      (op == OP_goto16) ? (int)(int16_t)get_u16(p + 1) : (int)(int8_t)p[1];
            int target = ooff + 1 + rel;
            jit_nat_spill_all(c, &L, vstk, vsp);
            if (target <= ooff) {
                /* Backward branch: inline interrupt_counter check */
                struct sljit_jump *jslow, *jcont;
                struct sljit_label *lslow;
                sljit_emit_op1(c, SLJIT_MOV_U32, SLJIT_R1, 0,
                               SLJIT_MEM1(SLJIT_S0),
                               (sljit_sw)offsetof(JSContext, interrupt_counter));
                sljit_emit_op2(c, SLJIT_SUB, SLJIT_R1, 0, SLJIT_R1, 0, SLJIT_IMM, 1);
                sljit_emit_op1(c, SLJIT_MOV32, SLJIT_MEM1(SLJIT_S0),
                               (sljit_sw)offsetof(JSContext, interrupt_counter),
                               SLJIT_R1, 0);
                jslow = sljit_emit_cmp(c, SLJIT_SIG_LESS_EQUAL, SLJIT_R1, 0, SLJIT_IMM, 0);
                j = sljit_emit_jump(c, SLJIT_JUMP);
                fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BRANCH; fx[nfx].target_off = target; nfx++;

                lslow = sljit_emit_label(c);
                sljit_set_label(jslow, lslow);
                sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_S0, 0);
                sljit_emit_icall(c, SLJIT_CALL, SLJIT_ARGS1(W, P),
                                 SLJIT_IMM, (sljit_sw)&jit_poll_slow);
                j = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_RETURN_REG, 0, SLJIT_IMM, 0);
                fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
                jcont = sljit_emit_jump(c, SLJIT_JUMP);
                fx[nfx].jmp = jcont; fx[nfx].kind = JIT_FX_BRANCH; fx[nfx].target_off = target; nfx++;
            } else {
                j = sljit_emit_jump(c, SLJIT_JUMP);
                fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BRANCH; fx[nfx].target_off = target; nfx++;
            }
            break;
        }

        case OP_if_true8: case OP_if_false8: case OP_if_true: case OP_if_false: {
            int rel = (op == OP_if_true || op == OP_if_false) ? (int)(int32_t)get_u32(p + 1) : (int)(int8_t)p[1];
            int target = ooff + 1 + rel;
            sljit_s32 ct = (op == OP_if_true || op == OP_if_true8) ? SLJIT_NOT_EQUAL : SLJIT_EQUAL;
            int pop_d = --vsp;
            sljit_s32 r1; sljit_sw w1;
            jit_nat_vslot_operand(&L, &vstk[pop_d], pop_d, &r1, &w1);
            jit_nat_spill_all(c, &L, vstk, vsp);
            j = sljit_emit_cmp(c, ct | SLJIT_32, r1, w1, SLJIT_IMM, 0);
            fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BRANCH; fx[nfx].target_off = target; nfx++;
            break;
        }

        case OP_get_var:
            vstk[vsp].kind = VK_SELF_FN;
            vstk[vsp].type = VTYPE_FN;
            vsp++;
            break;

        case OP_call0: case OP_call1: case OP_call2: case OP_call3: case OP_call: {
            int argc = (op == OP_call) ? (int)get_u16(p + 1) : (op - OP_call0);
            int fn_slot = vsp - 1 - argc;
            if (fn_slot < 0 || vstk[fn_slot].kind != VK_SELF_FN || argc > JIT_NAT_MAX_ARGS) {
                sljit_free_compiler(c);
                js_free(ctx, offs); js_free(ctx, stk_depth); js_free(ctx, is_target);
                js_free(ctx, labels); js_free(ctx, fx);
                return 0;
            }
            /* Spill any live stack items below this call */
            jit_nat_spill_all(c, &L, vstk, fn_slot);

            /* Interrupt/stack-depth guard before recursive call */
            struct sljit_jump *jslow, *jok;
            sljit_emit_op1(c, SLJIT_MOV_U32, SLJIT_R1, 0,
                           SLJIT_MEM1(SLJIT_S0),
                           (sljit_sw)offsetof(JSContext, interrupt_counter));
            sljit_emit_op2(c, SLJIT_SUB, SLJIT_R1, 0, SLJIT_R1, 0, SLJIT_IMM, 1);
            sljit_emit_op1(c, SLJIT_MOV32, SLJIT_MEM1(SLJIT_S0),
                           (sljit_sw)offsetof(JSContext, interrupt_counter),
                           SLJIT_R1, 0);
            jslow = sljit_emit_cmp(c, SLJIT_SIG_LESS_EQUAL, SLJIT_R1, 0, SLJIT_IMM, 0);
            jok = sljit_emit_jump(c, SLJIT_JUMP);
            sljit_set_label(jslow, sljit_emit_label(c));
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_S0, 0);
            sljit_emit_icall(c, SLJIT_CALL, SLJIT_ARGS1(W, P),
                             SLJIT_IMM, (sljit_sw)&jit_poll_slow);
            j = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_RETURN_REG, 0, SLJIT_IMM, 0);
            fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
            sljit_set_label(jok, sljit_emit_label(c));

            /* Load args into R1..R3, ctx into R0, and call worker_native via &e->code */
            for (int a = 0; a < argc; a++) {
                jit_nat_load_reg(c, &L, &vstk[fn_slot + 1 + a], fn_slot + 1 + a,
                                 (sljit_s32)(SLJIT_R1 + a));
            }
            sljit_emit_op1(c, SLJIT_MOV, SLJIT_R0, 0, SLJIT_S0, 0);
            sljit_emit_icall(c, SLJIT_CALL, SLJIT_ARGS4(W, P, W, W, W),
                             SLJIT_MEM0(), (sljit_sw)&e->code);
            /* Check bail bit (upper 32 bits != 0) */
            sljit_emit_op2(c, SLJIT_LSHR, SLJIT_R1, 0, SLJIT_RETURN_REG, 0, SLJIT_IMM, 32);
            j = sljit_emit_cmp(c, SLJIT_NOT_EQUAL, SLJIT_R1, 0, SLJIT_IMM, 0);
            fx[nfx].jmp = j; fx[nfx].kind = JIT_FX_BAIL; nfx++;
            sljit_emit_op1(c, SLJIT_MOV_S32,
                           SLJIT_MEM1(SLJIT_SP), jit_nat_stk_off(&L, fn_slot),
                           SLJIT_RETURN_REG, 0);
            vsp = fn_slot + 1;
            vstk[fn_slot].kind = VK_STK;
            vstk[fn_slot].type = VTYPE_INT;
            break;
        }

        case OP_return: {
            int d = --vsp;
            if (vstk[d].type != VTYPE_INT) {
                sljit_free_compiler(c);
                js_free(ctx, offs); js_free(ctx, stk_depth); js_free(ctx, is_target);
                js_free(ctx, labels); js_free(ctx, fx);
                return 0;
            }
            jit_nat_load_reg(c, &L, &vstk[d], d, SLJIT_R0);
            /* Zero-extend 32-bit int to 64-bit so upper 32 bits == 0 (not bailed) */
            sljit_emit_op1(c, SLJIT_MOV_U32, SLJIT_R0, 0, SLJIT_R0, 0);
            sljit_emit_return(c, SLJIT_MOV, SLJIT_R0, 0);
            break;
        }

        default:
            sljit_free_compiler(c);
            js_free(ctx, offs); js_free(ctx, stk_depth); js_free(ctx, is_target);
            js_free(ctx, labels); js_free(ctx, fx);
            return 0;
        }
    }

    bail_label = sljit_emit_label(c);
    sljit_emit_return(c, SLJIT_MOV, SLJIT_IMM, (sljit_sw)1 << 32);

    for (int k = 0; k < nfx; k++) {
        if (fx[k].kind == JIT_FX_BAIL) {
            sljit_set_label(fx[k].jmp, bail_label);
        } else if (fx[k].kind == JIT_FX_BRANCH) {
            int li = jit_find_label_idx(offs, nops, fx[k].target_off);
            if (li >= 0)
                sljit_set_label(fx[k].jmp, labels[li]);
            else
                sljit_set_label(fx[k].jmp, bail_label);
        }
    }

    if (sljit_get_compiler_error(c)) {
        sljit_free_compiler(c);
        js_free(ctx, offs); js_free(ctx, stk_depth); js_free(ctx, is_target);
        js_free(ctx, labels); js_free(ctx, fx);
        return 0;
    }

    void *code = sljit_generate_code(c, 0, NULL);
    sljit_free_compiler(c);
    js_free(ctx, offs); js_free(ctx, stk_depth); js_free(ctx, is_target);
    js_free(ctx, labels); js_free(ctx, fx);
    if (!code)
        return 0;

    e->code = code;
    e->self_recursive = self_rec;
    return 1;
}

#endif /* JS_JIT_CODEGEN */

#endif /* JS_JIT */
