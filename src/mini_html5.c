/*
 * mini_html5.c — WHATWG HTML5 parsing pipeline (tokenizer + tree construction).
 *
 * Implements, from scratch and faithful to the WHATWG HTML Standard:
 *   • spec tokenizer state machine (data / tag-open / tag-name / attribute
 *     states / comment / doctype / RAWTEXT / RCDATA / character-reference)
 *   • tree construction with insertion modes + the open-elements stack and
 *     the active-formatting-elements list
 *   • the Adoption Agency Algorithm for mis-nested formatting elements
 *   • implicit end-tag generation (auto-close) for p / li / dd / dt / option /
 *     tr / td / th / tbody / tfoot / thead / caption / colgroup / html / head
 *   • table foster-parenting (mis-placed table-internal nodes are fostered
 *     to the location just before the <table>)
 *   • streaming: the parser is fed chunks; partial tokens at a chunk boundary
 *     are carried over so the DOM is built incrementally as bytes arrive.
 *
 * Builds directly into a MiniDocument via the mini_node_* tree API so it
 * composes with the existing layout/render/CDP pipeline (no separate IR).
 *
 * Reference: https://html.spec.whatwg.org/ — §13.2.5 (tokenizer), §13.2.6
 * (tree construction), §13.2.6.4 (adoption agency).
 */
#include "mini_html5.h"
#include "mini_dom.h"
#include "mini_log.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <stdarg.h>

/* ================================================================== */
/* Element classification (spec "special" / "formatting" / void / text) */
/* ================================================================== */
/* Case-insensitive ASCII equal. */
static int h5_ci(const char *a, const char *b)
{
    if (!a || !b)
        return a == b;
    while (*a && *b)
    {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b))
            return 0;
        a++;
        b++;
    }
    return *a == *b;
}

/* The spec's "special" set — sorted for fast O(log N) binary search. */
static int h5_is_special(const char *t)
{
    static const char *const S[] = {
        "address", "applet", "area", "article", "aside", "base", "basefont", "bgsound",
        "blockquote", "body", "br", "button", "caption", "center", "col", "colgroup",
        "dd", "details", "dialog", "dir", "div", "dl", "dt", "embed", "fieldset", "figcaption",
        "figure", "footer", "form", "frame", "frameset", "h1", "h2", "h3", "h4", "h5",
        "h6", "head", "header", "hgroup", "hr", "html", "iframe", "img", "input",
        "isindex", "li", "link", "listing", "main", "marquee", "menu", "meta", "nav",
        "noembed", "noframes", "noscript", "object", "ol", "p", "param", "plaintext",
        "pre", "script", "search", "section", "select", "style", "summary", "table", "tbody",
        "td", "template", "textarea", "tfoot", "th", "thead", "title", "tr", "ul", "wbr",
        "xmp"
    };
    if (!t)
        return 0;
    int lo = 0, hi = (int)(sizeof(S) / sizeof(S[0])) - 1;
    while (lo <= hi)
    {
        int mid = (lo + hi) / 2;
        int cmp = _stricmp(t, S[mid]);
        if (cmp == 0)
            return 1;
        if (cmp < 0)
            hi = mid - 1;
        else
            lo = mid + 1;
    }
    return 0;
}
/* The adoption-agency formatting set (§13.2.6.4). */
static int h5_is_formatting(const char *t)
{
    static const char *const F[] = {"a", "b", "big", "code", "em", "font", "i", "nobr", "s",
                                    "small", "strike", "strong", "tt", "u", NULL};
    if (!t)
        return 0;
    for (int i = 0; F[i]; i++)
        if (h5_ci(t, F[i]))
            return 1;
    return 0;
}
/* RCDATA elements: text + entity decoding (title, textarea). */
static int h5_is_rcdata(const char *t)
{
    return t && (!strcmp(t, "title") || !strcmp(t, "textarea"));
}
/* RAWTEXT elements: verbatim text, no entity decode (script/style/xmp/...). */
static int h5_is_rawtext(const char *t)
{
    static const char *const R[] = {"style", "script", "xmp", "iframe", "noembed",
                                    "noframes", "noscript", "plaintext", NULL};
    if (!t)
        return 0;
    for (int i = 0; R[i]; i++)
        if (h5_ci(t, R[i]))
            return 1;
    return 0;
}

/* ---- Comprehensive 250+ named character-reference table (sorted) ------- */
typedef struct {
    const char *n;
    const char *u;
} H5Entity;

static const H5Entity H5_ENT[] = {
    {"AElig", "\xC3\x86"},
    {"Aacute", "\xC3\x81"},
    {"Acirc", "\xC3\x82"},
    {"Agrave", "\xC3\x80"},
    {"Alpha", "\xCE\x91"},
    {"Aring", "\xC3\x85"},
    {"Atilde", "\xC3\x83"},
    {"Auml", "\xC3\x84"},
    {"Beta", "\xCE\x92"},
    {"Ccedil", "\xC3\x87"},
    {"Chi", "\xCE\xA7"},
    {"Dagger", "\xE2\x80\xA1"},
    {"Delta", "\xCE\x94"},
    {"ETH", "\xC3\x90"},
    {"Eacute", "\xC3\x89"},
    {"Ecirc", "\xC3\x8A"},
    {"Egrave", "\xC3\x88"},
    {"Epsilon", "\xCE\x95"},
    {"Eta", "\xCE\x97"},
    {"Euml", "\xC3\x8B"},
    {"Gamma", "\xCE\x93"},
    {"Iacute", "\xC3\x8D"},
    {"Icirc", "\xC3\x8E"},
    {"Igrave", "\xC3\x8C"},
    {"Iota", "\xCE\x99"},
    {"Iuml", "\xC3\x8F"},
    {"Kappa", "\xCE\x9A"},
    {"Lambda", "\xCE\x9B"},
    {"Mu", "\xCE\x9C"},
    {"Ntilde", "\xC3\x91"},
    {"Nu", "\xCE\x9D"},
    {"OElig", "\xC5\x92"},
    {"Oacute", "\xC3\x93"},
    {"Ocirc", "\xC3\x94"},
    {"Ograve", "\xC3\x92"},
    {"Omega", "\xCE\xA9"},
    {"Omicron", "\xCE\x9F"},
    {"Oslash", "\xC3\x98"},
    {"Otilde", "\xC3\x95"},
    {"Ouml", "\xC3\x96"},
    {"Phi", "\xCE\xA6"},
    {"Pi", "\xCE\xA0"},
    {"Prime", "\xE2\x80\xB3"},
    {"Psi", "\xCE\xA8"},
    {"Rho", "\xCE\xA1"},
    {"Sigma", "\xCE\xA3"},
    {"THORN", "\xC3\x9E"},
    {"Tau", "\xCE\xA4"},
    {"Theta", "\xCE\x98"},
    {"Uacute", "\xC3\x9A"},
    {"Ucirc", "\xC3\x9B"},
    {"Ugrave", "\xC3\x99"},
    {"Upsilon", "\xCE\xA5"},
    {"Uuml", "\xC3\x9C"},
    {"Xi", "\xCE\x9E"},
    {"Yacute", "\xC3\x9D"},
    {"Yuml", "\xC5\xB8"},
    {"Zeta", "\xCE\xB6"},
    {"aacute", "\xC3\xA1"},
    {"acirc", "\xC3\xA2"},
    {"acute", "\xC2\xB4"},
    {"aelig", "\xC3\xA6"},
    {"agrave", "\xC3\xA0"},
    {"alefsym", "\xE2\x84\xB5"},
    {"alpha", "\xCE\xB1"},
    {"amp", "&"},
    {"and", "\xE2\x88\xA7"},
    {"ang", "\xE2\x88\xA0"},
    {"apos", "'"},
    {"aring", "\xC3\xA5"},
    {"asymp", "\xE2\x89\x88"},
    {"atilde", "\xC3\xA3"},
    {"auml", "\xC3\xA4"},
    {"bdquo", "\xE2\x80\x9E"},
    {"beta", "\xCE\xB2"},
    {"brvbar", "\xC2\xA6"},
    {"bull", "\xE2\x80\xA2"},
    {"cap", "\xE2\x88\xA9"},
    {"ccedil", "\xC3\xA7"},
    {"cedil", "\xC2\xB8"},
    {"cent", "\xC2\xA2"},
    {"check", "\xE2\x9C\x93"},
    {"chi", "\xCE\xB7"},
    {"circ", "\xCB\x86"},
    {"clubs", "\xE2\x99\xA3"},
    {"cong", "\xE2\x89\x85"},
    {"copy", "\xC2\xA9"},
    {"cup", "\xE2\x88\xAA"},
    {"curren", "\xC2\xA4"},
    {"dArr", "\xE2\x87\x93"},
    {"dagger", "\xE2\x80\xA0"},
    {"darr", "\xE2\x86\x93"},
    {"deg", "\xC2\xB0"},
    {"delta", "\xCE\xB4"},
    {"diams", "\xE2\x99\xA6"},
    {"divide", "\xC3\xB7"},
    {"eacute", "\xC3\xA9"},
    {"ecirc", "\xC3\xAA"},
    {"egrave", "\xC3\xA8"},
    {"empty", "\xE2\x88\x85"},
    {"emsp", "\xE2\x80\x83"},
    {"ensp", "\xE2\x80\x82"},
    {"epsilon", "\xCE\xB5"},
    {"equiv", "\xE2\x89\xA1"},
    {"eta", "\xCE\xB7"},
    {"eth", "\xC3\xB0"},
    {"euml", "\xC3\xAB"},
    {"euro", "\xE2\x82\xAC"},
    {"exist", "\xE2\x88\x83"},
    {"fnof", "\xC6\x92"},
    {"forall", "\xE2\x88\x80"},
    {"frac12", "\xC2\xBD"},
    {"frac14", "\xC2\xBC"},
    {"frac34", "\xC2\xBE"},
    {"gamma", "\xCE\xB3"},
    {"ge", "\xE2\x89\xA5"},
    {"gt", ">"},
    {"hArr", "\xE2\x87\x94"},
    {"harr", "\xE2\x86\x94"},
    {"hearts", "\xE2\x99\xA5"},
    {"hellip", "\xE2\x80\xA6"},
    {"iacute", "\xC3\xAD"},
    {"icirc", "\xC3\xAE"},
    {"iexcl", "\xC2\xA1"},
    {"igrave", "\xC3\xAC"},
    {"infin", "\xE2\x88\x9E"},
    {"int", "\xE2\x88\xAB"},
    {"iota", "\xCE\xB9"},
    {"iquest", "\xC2\xBF"},
    {"isin", "\xE2\x88\x88"},
    {"iuml", "\xC3\xAF"},
    {"kappa", "\xCE\xBA"},
    {"lArr", "\xE2\x87\x90"},
    {"lambda", "\xCE\xBB"},
    {"lang", "\xE2\x9F\xA8"},
    {"laquo", "\xC2\xAB"},
    {"larr", "\xE2\x86\x90"},
    {"lceil", "\xE2\x8C\x88"},
    {"ldquo", "\xE2\x80\x9C"},
    {"le", "\xE2\x89\xA4"},
    {"lfloor", "\xE2\x8C\x8A"},
    {"lowast", "\xE2\x88\x97"},
    {"loz", "\xE2\x97\x8A"},
    {"lrm", "\xE2\x80\x8E"},
    {"lsaquo", "\xE2\x80\xB9"},
    {"lsquo", "\xE2\x80\x98"},
    {"lt", "<"},
    {"macr", "\xC2\xAF"},
    {"mdash", "\xE2\x80\x94"},
    {"micro", "\xC2\xB5"},
    {"middot", "\xC2\xB7"},
    {"minus", "\xE2\x88\x92"},
    {"mu", "\xCE\xBC"},
    {"nabla", "\xE2\x88\x87"},
    {"nbsp", " "},
    {"ndash", "\xE2\x80\x93"},
    {"ne", "\xE2\x89\xA0"},
    {"ni", "\xE2\x88\x8B"},
    {"not", "\xC2\xAC"},
    {"notin", "\xE2\x88\x89"},
    {"nsub", "\xE2\x8A\x84"},
    {"ntilde", "\xC3\xB1"},
    {"nu", "\xCE\xBD"},
    {"oacute", "\xC3\xB3"},
    {"ocirc", "\xC3\xB4"},
    {"oelig", "\xC5\x93"},
    {"ograve", "\xC3\xB2"},
    {"oline", "\xE2\x80\xBE"},
    {"omega", "\xCE\xBF"},
    {"omicron", "\xCE\xBF"},
    {"oplus", "\xE2\x8A\x95"},
    {"or", "\xE2\x88\xA8"},
    {"ordf", "\xC2\xAA"},
    {"ordm", "\xC2\xBA"},
    {"oslash", "\xC3\xB8"},
    {"otilde", "\xC3\xB5"},
    {"otimes", "\xE2\x8A\x97"},
    {"ouml", "\xC3\xB6"},
    {"para", "\xC2\xB6"},
    {"part", "\xE2\x88\x82"},
    {"permil", "\xE2\x80\xB0"},
    {"perp", "\xE2\x8A\xA5"},
    {"phi", "\xCF\x86"},
    {"pi", "\xCF\x80"},
    {"piv", "\xCF\x96"},
    {"plusmn", "\xC2\xB1"},
    {"pound", "\xC2\xA3"},
    {"prime", "\xE2\x80\xB2"},
    {"prod", "\xE2\x88\x8F"},
    {"prop", "\xE2\x88\x9D"},
    {"psi", "\xCF\x88"},
    {"quot", "\""},
    {"rArr", "\xE2\x87\x92"},
    {"radic", "\xE2\x88\x9A"},
    {"rang", "\xE2\x9F\xA9"},
    {"raquo", "\xC2\xBB"},
    {"rarr", "\xE2\x86\x92"},
    {"rceil", "\xE2\x8C\x89"},
    {"rdquo", "\xE2\x80\x9D"},
    {"real", "\xE2\x84\x9C"},
    {"reg", "\xC2\xAE"},
    {"rfloor", "\xE2\x8C\x8B"},
    {"rho", "\xCF\x81"},
    {"rlm", "\xE2\x80\x8F"},
    {"rsaquo", "\xE2\x80\xBA"},
    {"rsquo", "\xE2\x80\x99"},
    {"sbquo", "\xE2\x80\x9A"},
    {"sdot", "\xE2\x8B\x85"},
    {"sect", "\xC2\xA7"},
    {"shy", "\xC2\xAD"},
    {"sigma", "\xCF\x83"},
    {"sigmaf", "\xCF\x82"},
    {"sim", "\xE2\x88\xBC"},
    {"spades", "\xE2\x99\xA0"},
    {"sub", "\xE2\x8A\x82"},
    {"sube", "\xE2\x8A\x86"},
    {"sum", "\xE2\x88\x91"},
    {"sup", "\xE2\x8A\x83"},
    {"sup1", "\xC2\xB9"},
    {"sup2", "\xC2\xB2"},
    {"sup3", "\xC2\xB3"},
    {"supe", "\xE2\x8A\x87"},
    {"szlig", "\xC3\x9F"},
    {"tau", "\xCF\x84"},
    {"there4", "\xE2\x88\xB4"},
    {"theta", "\xCE\xB8"},
    {"thetasym", "\xCF\x91"},
    {"thinsp", "\xE2\x80\x89"},
    {"thorn", "\xC3\xBE"},
    {"tilde", "\xCB\x9C"},
    {"times", "\xC3\x97"},
    {"trade", "\xE2\x84\xA2"},
    {"uArr", "\xE2\x87\x91"},
    {"uacute", "\xC3\xBA"},
    {"uarr", "\xE2\x86\x91"},
    {"ucirc", "\xC3\xBB"},
    {"ugrave", "\xC3\xB9"},
    {"uml", "\xC2\xA8"},
    {"upsih", "\xCF\x92"},
    {"upsilon", "\xCF\x85"},
    {"uuml", "\xC3\xBC"},
    {"weierp", "\xE2\x84\x98"},
    {"xi", "\xCE\xBE"},
    {"yacute", "\xC3\xBD"},
    {"yen", "\xC2\xA5"},
    {"yuml", "\xC3\xBF"},
    {"zeta", "\xCE\xB6"},
    {"zwj", "\xE2\x80\x8D"},
    {"zwnj", "\xE2\x80\x8C"}
};

static const H5Entity *h5_find_entity(const char *name)
{
    int lo = 0, hi = (int)(sizeof(H5_ENT) / sizeof(H5_ENT[0])) - 1;
    while (lo <= hi)
    {
        int mid = (lo + hi) / 2;
        int cmp = strcmp(name, H5_ENT[mid].n);
        if (cmp == 0)
            return &H5_ENT[mid];
        if (cmp < 0)
            hi = mid - 1;
        else
            lo = mid + 1;
    }
    return NULL;
}

/* Encode a single Unicode codepoint as UTF-8 into out (<=4 bytes). */
static int h5_utf8(unsigned cp, char *out)
{
    if (cp < 0x80)
    {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800)
    {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000)
    {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* Try to decode a character reference beginning at `s[0]` ('&'). On success
   writes UTF-8 to out and returns bytes consumed (incl. trailing ';'); on
   failure returns 0 (caller emits the literal '&'). */
static size_t h5_decode_ref_diag(struct MiniHtml5Parser *p, const char *s, size_t len, char *out, size_t cap);
#define h5_decode_ref(s, len, out, cap) h5_decode_ref_diag(NULL, s, len, out, cap)

/* ================================================================== */
/* Tokenizer                                                          */
/* ================================================================== */
typedef enum
{
    TS_DATA,
    TS_TAG_OPEN,
    TS_END_TAG_OPEN,
    TS_TAG_NAME,
    TS_BEFORE_ATTR_NAME,
    TS_ATTR_NAME,
    TS_AFTER_ATTR_NAME,
    TS_BEFORE_ATTR_VALUE,
    TS_ATTR_VALUE_DQ,
    TS_ATTR_VALUE_SQ,
    TS_ATTR_VALUE_UQ,
    TS_AFTER_ATTR_VALUE,
    TS_SELF_CLOSING,
    TS_BOGUS_COMMENT,
    TS_MARKUP_DECL_OPEN,
    TS_COMMENT,
    TS_DOCTYPE,
    TS_RAWTEXT,
    TS_RCDATA,
    TS_EOF
} H5TokState;

typedef enum
{
    TT_START,
    TT_END,
    TT_CHAR,
    TT_COMMENT,
    TT_DOCTYPE,
    TT_EOF
} H5TokType;

typedef struct
{
    H5TokType type;
    char *name; /* tag/comment/doctype name buffer */
    size_t name_cap;
    struct
    {
        char *n;
        char *v;
        size_t n_cap;
        size_t v_cap;
    } *attrs;
    int n_attr, cap_attr;
    int is_self_closing;
    int line, col;
} H5Token;

/* ================================================================== */
/* Tree-construction state                                            */
/* ================================================================== */
typedef enum
{
    M_INITIAL,
    M_BEFORE_HTML,
    M_BEFORE_HEAD,
    M_IN_HEAD,
    M_AFTER_HEAD,
    M_IN_BODY,
    M_IN_TABLE,
    M_IN_TABLE_BODY,
    M_IN_ROW,
    M_IN_CELL,
    M_IN_SELECT,
    M_AFTER_BODY,
    M_AFTER_AFTER_BODY
} H5Mode;

typedef struct
{
    int marker;
    struct MiniNode *node;
} AFEEntry;

struct MiniHtml5Parser
{
    struct MiniDocument *doc;
    struct MiniNode **open;
    int open_n, open_cap; /* open-elements stack */
    AFEEntry *afe;
    int afe_n, afe_cap; /* active formatting elts */
    H5Mode mode;
    struct MiniNode *head_element;      /* the <head> */
    struct MiniNode *form_element;      /* the <form> (AAA ctx) */
    struct MiniNode *table_foster;      /* current table for foster-parenting */
    struct MiniNode *pending_text_tail; /* last text node, for coalescing */
    size_t pending_text_len;            /* amortized length of pending text */
    size_t pending_text_cap;            /* amortized capacity of pending text */
    int fragment;                       /* setInnerHTML context */

    /* Line and column tracking for diagnostic reports */
    int line;
    int col;
    int cur_tok_line;
    int cur_tok_col;
    const char *src_input;
    MiniHtml5DiagCallback diag_cb;
    void *diag_ud;

    H5TokState ts;
    char *acc;
    int acc_n, acc_cap; /* token accumulator */
    char *carry;
    int carry_n;           /* partial-token carryover across feeds */
    H5Token cur;           /* current tag token being built */
    int reconsume;         /* 1 = re-feed carry next */
    int rawtext_is_rcdata; /* distinguishes RCDATA vs RAWTEXT */
    int eof;
};

static void h5_report_diag(struct MiniHtml5Parser *p, int sev, const char *rule, const char *fmt, ...)
{
    MiniParseDiagnostic diag;
    memset(&diag, 0, sizeof(diag));
    diag.line = (p && p->cur_tok_line > 0) ? p->cur_tok_line : (p && p->line > 0 ? p->line : 1);
    diag.col = (p && p->cur_tok_col > 0) ? p->cur_tok_col : (p && p->col > 0 ? p->col : 1);
    diag.severity = (MiniParseSeverity)sev;
    strncpy(diag.source, "html", sizeof(diag.source) - 1);
    if (rule)
        strncpy(diag.rule, rule, sizeof(diag.rule) - 1);

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(diag.message, sizeof(diag.message), fmt, ap);
    va_end(ap);

    const char *src = p ? p->src_input : NULL;
    mini_format_diagnostic_snippet(src, diag.line, diag.col, "HTML5", (MiniParseSeverity)sev,
                                   diag.rule, diag.message, diag.snippet, sizeof(diag.snippet));

    if (p && p->diag_cb)
        p->diag_cb(&diag, p->diag_ud);
    if (p && p->doc)
        mini_doc_add_diagnostic(p->doc, &diag);

    MiniLogLevel lvl = (sev == MINI_PARSE_SEV_ERROR) ? MINI_LOG_ERROR
                     : (sev == MINI_PARSE_SEV_WARN)  ? MINI_LOG_WARN
                                                     : MINI_LOG_INFO;
    mini_logf(lvl, "MiniHtml5", "\n%s", diag.snippet[0] ? diag.snippet : diag.message);
}

static size_t h5_decode_ref_diag(struct MiniHtml5Parser *p, const char *s, size_t len, char *out, size_t cap)
{
    if (!len || s[0] != '&')
        return 0;
    size_t i = 1;
    if (i < len && s[i] == '#')
    { /* numeric reference */
        i++;
        int hex = 0;
        unsigned int val = 0;
        if (i < len && (s[i] == 'x' || s[i] == 'X'))
        {
            hex = 1;
            i++;
        }
        int digits = 0;
        while (i < len)
        {
            char c = s[i];
            int d = (c >= '0' && c <= '9') ? c - '0' : (hex && c >= 'a' && c <= 'f') ? c - 'a' + 10
                                                   : (hex && c >= 'A' && c <= 'F')   ? c - 'A' + 10
                                                                                     : -1;
            if (d < 0)
                break;
            val = val * (hex ? 16 : 10) + (unsigned int)d;
            digits++;
            i++;
        }
        if (!digits)
            return 0;
        if (i < len && s[i] == ';')
            i++;
        else if (i == len)
            return 0; /* incomplete */

        /* Windows-1252 remap table for 0x80..0x9F per HTML5 spec */
        static const unsigned short WIN1252_REMAP[32] = {
            0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021,
            0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0x008D, 0x017D, 0x008F,
            0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
            0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x009D, 0x017E, 0x0178
        };
        if (val >= 0x80 && val <= 0x9F)
            val = WIN1252_REMAP[val - 0x80];
        else if (val == 0 || val > 0x10FFFF || (val >= 0xD800 && val <= 0xDFFF))
        {
            if (p)
                h5_report_diag(p, MINI_PARSE_SEV_WARN, "invalid-codepoint", "Character reference resolves to invalid Unicode code point U+%04X", val);
            val = 0xFFFD;
        }
        int w = h5_utf8(val, out);
        if ((size_t)w < cap)
            out[w] = 0;
        return i;
    }
    /* named reference: letters/digits up to ';' */
    size_t start = i;
    while (i < len)
    {
        char c = s[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))
            i++;
        else
            break;
    }
    size_t nl = i - start;
    int semi = (i < len && s[i] == ';');
    if (nl == 0)
        return 0;
    if (!semi && i == len)
        return 0; /* possibly incomplete: defer */
    char name[32];
    if (nl >= sizeof(name))
        nl = sizeof(name) - 1;
    memcpy(name, s + start, nl);
    name[nl] = 0;

    const H5Entity *ent = h5_find_entity(name);
    if (ent)
    {
        size_t w = 0;
        while (ent->u[w] && w + 1 < cap)
        {
            out[w] = ent->u[w];
            w++;
        }
        out[w] = 0;
        return semi ? i + 1 : i;
    }
    if (semi && p)
        h5_report_diag(p, MINI_PARSE_SEV_WARN, "unknown-named-entity", "Unknown named entity '&%s;'", name);
    return 0;
}

static void set_last_attr_value(H5Token *t, const char *v);
static void pop_until_one(struct MiniHtml5Parser *p, const char *a,
                          const char *b, const char *c);
static void maybe_enter_rawtext(struct MiniHtml5Parser *p);
static void process_token(struct MiniHtml5Parser *p, H5Token *t);
static void emit_current(struct MiniHtml5Parser *p);
static void emit_char(struct MiniHtml5Parser *p, const char *s);

/* ---- dynamic arrays ---- */
static int ptr_push(struct MiniNode ***arr, int *n, int *cap, struct MiniNode *v)
{
    if (*n == *cap)
    {
        int nc = *cap ? *cap * 2 : 16;
        struct MiniNode **na = realloc(*arr, nc * sizeof(void *));
        if (!na)
            return -1;
        *arr = na;
        *cap = nc;
    }
    (*arr)[(*n)++] = v;
    return 0;
}
static struct MiniNode *ptr_pop(struct MiniNode **arr, int *n)
{
    if (*n == 0)
        return NULL;
    return arr[--(*n)];
}

/* ---- string accumulator ---- */
static void acc_putc(struct MiniHtml5Parser *p, char c)
{
    if (p->acc_n + 2 > p->acc_cap)
    {
        int nc = p->acc_cap ? p->acc_cap * 2 : 64;
        char *na = realloc(p->acc, nc);
        if (!na)
            return;
        p->acc = na;
        p->acc_cap = nc;
    }
    p->acc[p->acc_n++] = c;
    p->acc[p->acc_n] = 0;
}
static void acc_puts(struct MiniHtml5Parser *p, const char *s)
{
    if (s)
        while (*s)
            acc_putc(p, *s++);
}
static void acc_reset(struct MiniHtml5Parser *p)
{
    p->acc_n = 0;
    if (p->acc)
        p->acc[0] = 0;
}
static char *acc_dup(struct MiniHtml5Parser *p)
{
    if (p->acc_n == 0)
        return strdup("");
    char *s = malloc(p->acc_n + 1);
    if (!s)
        return NULL;
    memcpy(s, p->acc, p->acc_n);
    s[p->acc_n] = 0;
    return s;
}

/* ---- token attr helpers with buffer reuse ---- */
static void tok_set_name(H5Token *t, const char *s)
{
    size_t len = s ? strlen(s) : 0;
    if (len + 1 > t->name_cap)
    {
        size_t ncap = (len + 1 < 32) ? 32 : (len + 1) * 2;
        char *nb = realloc(t->name, ncap);
        if (!nb)
            return;
        t->name = nb;
        t->name_cap = ncap;
    }
    if (s && len)
        memcpy(t->name, s, len);
    t->name[len] = 0;
}

static void tok_add_attr(H5Token *t, const char *name, const char *val)
{
    if (t->n_attr == t->cap_attr)
    {
        int nc = t->cap_attr ? t->cap_attr * 2 : 8;
        void *na = realloc(t->attrs, nc * sizeof(t->attrs[0]));
        if (!na)
            return;
        memset((char *)na + t->cap_attr * sizeof(t->attrs[0]), 0, (nc - t->cap_attr) * sizeof(t->attrs[0]));
        t->attrs = na;
        t->cap_attr = nc;
    }
    size_t nl = name ? strlen(name) : 0;
    size_t vl = val ? strlen(val) : 0;
    if (nl + 1 > t->attrs[t->n_attr].n_cap)
    {
        size_t ncap = (nl + 1 < 32) ? 32 : (nl + 1) * 2;
        char *nb = realloc(t->attrs[t->n_attr].n, ncap);
        if (nb)
        {
            t->attrs[t->n_attr].n = nb;
            t->attrs[t->n_attr].n_cap = ncap;
        }
    }
    if (vl + 1 > t->attrs[t->n_attr].v_cap)
    {
        size_t vcap = (vl + 1 < 32) ? 32 : (vl + 1) * 2;
        char *nb = realloc(t->attrs[t->n_attr].v, vcap);
        if (nb)
        {
            t->attrs[t->n_attr].v = nb;
            t->attrs[t->n_attr].v_cap = vcap;
        }
    }
    if (name && nl)
        memcpy(t->attrs[t->n_attr].n, name, nl);
    t->attrs[t->n_attr].n[nl] = 0;
    if (val && vl)
        memcpy(t->attrs[t->n_attr].v, val, vl);
    t->attrs[t->n_attr].v[vl] = 0;
    t->n_attr++;
}

static void tok_reset(H5Token *t)
{
    if (t->name)
        t->name[0] = 0;
    t->n_attr = 0;
    t->is_self_closing = 0;
    t->type = TT_START;
    t->line = 0;
    t->col = 0;
}

static void tok_destroy(H5Token *t)
{
    free(t->name);
    t->name = NULL;
    t->name_cap = 0;
    for (int i = 0; i < t->cap_attr; i++)
    {
        free(t->attrs[i].n);
        free(t->attrs[i].v);
    }
    free(t->attrs);
    t->attrs = NULL;
    t->n_attr = t->cap_attr = 0;
}

/* ---- open-elements stack helpers ---- */
static struct MiniNode *current_node(struct MiniHtml5Parser *p) { return p->open_n ? p->open[p->open_n - 1] : p->doc->root; }
static int open_index(struct MiniHtml5Parser *p, struct MiniNode *n)
{
    for (int i = p->open_n - 1; i >= 0; i--)
        if (p->open[i] == n)
            return i;
    return -1;
}
static void open_push(struct MiniHtml5Parser *p, struct MiniNode *n) { ptr_push(&p->open, &p->open_n, &p->open_cap, n); }
static struct MiniNode *open_pop(struct MiniHtml5Parser *p) { return ptr_pop(p->open, &p->open_n); }

static int pop_until(struct MiniHtml5Parser *p, const char *tag)
{
    int found = 0;
    for (int i = p->open_n - 1; i >= 0; i--)
        if (p->open[i]->tag && h5_ci(p->open[i]->tag, tag))
        {
            found = 1;
            break;
        }
    if (!found)
        return 0;
    while (p->open_n)
    {
        struct MiniNode *n = open_pop(p);
        if (n->tag && h5_ci(n->tag, tag))
            break;
    }
    return 1;
}

static int has_in_scope(struct MiniHtml5Parser *p, const char *tag, int table_scope)
{
    static const char *DEFAULT_SCOPE[] = {"applet", "caption", "html", "table", "td", "th", "marquee", "object", "template", NULL};
    static const char *TABLE_SCOPE[] = {"html", "table", NULL};
    const char **b = table_scope ? TABLE_SCOPE : DEFAULT_SCOPE;
    for (int i = p->open_n - 1; i >= 0; i--)
    {
        struct MiniNode *n = p->open[i];
        if (n->tag && h5_ci(n->tag, tag))
            return 1;
        if (n->tag)
        {
            for (int j = 0; b[j]; j++)
                if (h5_ci(n->tag, b[j]))
                    return 0;
        }
    }
    return 0;
}

static void open_remove(struct MiniHtml5Parser *p, struct MiniNode *n)
{
    int idx = open_index(p, n);
    if (idx < 0)
        return;
    for (int i = idx; i < p->open_n - 1; i++)
        p->open[i] = p->open[i + 1];
    p->open_n--;
}

/* ---- active-formatting-elements list ---- */
static void afe_push_typed(struct MiniHtml5Parser *p, int marker, struct MiniNode *node)
{
    if (p->afe_n == p->afe_cap)
    {
        int nc = p->afe_cap ? p->afe_cap * 2 : 16;
        AFEEntry *na = realloc(p->afe, nc * sizeof(AFEEntry));
        if (!na)
            return;
        p->afe = na;
        p->afe_cap = nc;
    }
    p->afe[p->afe_n].marker = marker;
    p->afe[p->afe_n].node = node;
    p->afe_n++;
}
static void afe_marker(struct MiniHtml5Parser *p) { afe_push_typed(p, 1, NULL); }
static int afe_index(struct MiniHtml5Parser *p, struct MiniNode *n)
{
    for (int i = p->afe_n - 1; i >= 0; i--)
        if (!p->afe[i].marker && p->afe[i].node == n)
            return i;
    return -1;
}
static void afe_remove(struct MiniHtml5Parser *p, struct MiniNode *n)
{
    int idx = afe_index(p, n);
    if (idx < 0)
        return;
    for (int i = idx; i < p->afe_n - 1; i++)
        p->afe[i] = p->afe[i + 1];
    p->afe_n--;
}

/* ================================================================== */
/* Insertion: create element, set attrs, attach to current node       */
/* ================================================================== */
static struct MiniNode *create_element_for_token(struct MiniHtml5Parser *p, H5Token *t)
{
    struct MiniNode *n = mini_node_create_element(t->name ? t->name : "div");
    if (!n)
        return NULL;
    for (int i = 0; i < t->n_attr; i++)
        mini_node_set_attribute(n, t->attrs[i].n, t->attrs[i].v);
    return n;
}

static struct MiniNode *insert_element(struct MiniHtml5Parser *p, struct MiniNode *parent, struct MiniNode *n)
{
    mini_node_append_child(parent, n);
    open_push(p, n);
    p->pending_text_tail = NULL;
    return n;
}

static void foster_insert(struct MiniHtml5Parser *p, struct MiniNode *n)
{
    struct MiniNode *table = p->table_foster;
    struct MiniNode *parent = table ? table->parent : current_node(p);
    if (table && parent)
    {
        if (mini_node_insert_before(parent, n, table) == 0)
        {
            open_push(p, n);
            p->pending_text_tail = NULL;
            return;
        }
    }
    insert_element(p, current_node(p), n);
}

static int is_all_ws(const char *s)
{
    if (!s)
        return 1;
    while (*s)
    {
        unsigned char c = (unsigned char)*s;
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r' && c != '\f')
            return 0;
        s++;
    }
    return 1;
}

/* Insert a character slice (text) with amortized capacity doubling.
   Avoids O(N^2) strlen + realloc re-scans for large text runs and inline scripts/styles. */
static void insert_char_len(struct MiniHtml5Parser *p, const char *utf8, size_t nl)
{
    if (!utf8 || !nl)
        return;
    struct MiniNode *cur = current_node(p);
    struct MiniNode *parent = cur;

    /* Rawtext / RCDATA 节点（style, script, title, textarea）直接追加到 cur->text */
    if (cur && cur->type == MN_ELEMENT_NODE && cur->tag &&
        (h5_is_rawtext(cur->tag) || h5_is_rcdata(cur->tag)))
    {
        if (p->pending_text_tail != cur)
        {
            p->pending_text_tail = cur;
            p->pending_text_len = cur->text ? strlen(cur->text) : 0;
            p->pending_text_cap = p->pending_text_len ? p->pending_text_len + 1 : 0;
        }
        if (p->pending_text_len + nl + 1 > p->pending_text_cap)
        {
            size_t ncap = (p->pending_text_len + nl + 1) * 2;
            if (ncap < 128)
                ncap = 128;
            char *nb = (char *)realloc(cur->text, ncap);
            if (nb)
            {
                cur->text = nb;
                p->pending_text_cap = ncap;
            }
        }
        if (cur->text)
        {
            memcpy(cur->text + p->pending_text_len, utf8, nl);
            p->pending_text_len += nl;
            cur->text[p->pending_text_len] = 0;
        }
        return;
    }

    if ((p->mode == M_IN_TABLE || p->mode == M_IN_TABLE_BODY || p->mode == M_IN_ROW) &&
        p->table_foster)
    {
        if (is_all_ws(utf8))
            return;
        parent = p->table_foster->parent ? p->table_foster->parent : cur;
        struct MiniNode *tn = NULL;
        if (p->table_foster->prev_sibling && p->table_foster->prev_sibling->type == MN_TEXT_NODE)
            tn = p->table_foster->prev_sibling;
        if (tn)
        {
            size_t ol = strlen(tn->text);
            char *nb = realloc(tn->text, ol + nl + 1);
            if (nb)
            {
                tn->text = nb;
                memcpy(tn->text + ol, utf8, nl);
                tn->text[ol + nl] = 0;
            }
        }
        else
        {
            char *tmp = malloc(nl + 1);
            if (tmp)
            {
                memcpy(tmp, utf8, nl);
                tmp[nl] = 0;
                tn = mini_node_create_text(tmp);
                free(tmp);
                mini_node_insert_before(parent, tn, p->table_foster);
                p->pending_text_tail = tn;
                p->pending_text_len = nl;
                p->pending_text_cap = nl + 1;
            }
        }
        return;
    }

    struct MiniNode *lc = parent ? parent->last_child : NULL;
    if (p->pending_text_tail && lc == p->pending_text_tail && lc && lc->type == MN_TEXT_NODE)
    {
        if (p->pending_text_len + nl + 1 > p->pending_text_cap)
        {
            size_t ncap = (p->pending_text_len + nl + 1) * 2;
            if (ncap < 64)
                ncap = 64;
            char *nb = realloc(lc->text, ncap);
            if (nb)
            {
                lc->text = nb;
                p->pending_text_cap = ncap;
            }
        }
        if (lc->text)
        {
            memcpy(lc->text + p->pending_text_len, utf8, nl);
            p->pending_text_len += nl;
            lc->text[p->pending_text_len] = 0;
        }
    }
    else
    {
        char *tmp = malloc(nl + 1);
        if (tmp)
        {
            memcpy(tmp, utf8, nl);
            tmp[nl] = 0;
            struct MiniNode *tn = mini_node_create_text(tmp);
            free(tmp);
            mini_node_append_child(parent, tn);
            p->pending_text_tail = tn;
            p->pending_text_len = nl;
            p->pending_text_cap = nl + 1;
        }
    }
}

static void insert_char(struct MiniHtml5Parser *p, const char *utf8)
{
    if (utf8)
        insert_char_len(p, utf8, strlen(utf8));
}

static void insert_comment(struct MiniHtml5Parser *p, struct MiniNode *parent, const char *data)
{
    struct MiniNode *c = mini_node_create_comment(data ? data : "");
    if (c)
        mini_node_append_child(parent, c);
    p->pending_text_tail = NULL;
}

static int is_implied_end(const char *t)
{
    static const char *I[] = {"dd", "dt", "li", "option", "optgroup", "p", "rb", "rp", "rt", "rtc", NULL};
    if (!t)
        return 0;
    for (int i = 0; I[i]; i++)
        if (h5_ci(t, I[i]))
            return 1;
    return 0;
}

static void gen_implied_end(struct MiniHtml5Parser *p, const char *except)
{
    while (p->open_n)
    {
        struct MiniNode *n = current_node(p);
        const char *t = n->tag;
        if (!t || !is_implied_end(t))
            break;
        if (except && h5_ci(t, except))
            break;
        open_pop(p);
    }
}

static int h5_closes_p(const char *t)
{
    static const char *C[] = {
        "address", "article", "aside", "blockquote", "center", "details", "dialog",
        "dir", "div", "dl", "fieldset", "figcaption", "figure", "footer", "form",
        "h1", "h2", "h3", "h4", "h5", "h6", "header", "hgroup", "hr", "main", "nav", "ol",
        "p", "pre", "section", "table", "ul", "listing", "summary", NULL};
    if (!t)
        return 0;
    for (int i = 0; C[i]; i++)
        if (h5_ci(t, C[i]))
            return 1;
    return 0;
}

static void close_p_if_open(struct MiniHtml5Parser *p)
{
    gen_implied_end(p, NULL);
    if (has_in_scope(p, "p", 0))
        pop_until(p, "p");
}

static void reconstruct_active_formatting(struct MiniHtml5Parser *p)
{
    if (p->afe_n == 0)
        return;
    int entry = p->afe_n - 1;
    AFEEntry *e = &p->afe[entry];
    if (e->marker)
        return;
    if (e->node && open_index(p, e->node) >= 0)
        return;
    int idx = entry;
    while (idx >= 0)
    {
        if (p->afe[idx].marker)
            break;
        if (p->afe[idx].node && open_index(p, p->afe[idx].node) >= 0)
            break;
        idx--;
    }
    for (int i = idx + 1; i < p->afe_n; i++)
    {
        struct MiniNode *orig = p->afe[i].node;
        if (!orig || p->afe[i].marker)
            continue;
        struct MiniNode *clone = mini_node_clone(orig, 0);
        if (!clone)
            continue;
        mini_node_append_child(current_node(p), clone);
        open_push(p, clone);
        p->afe[i].node = clone;
    }
}

static void adoption_agency(struct MiniHtml5Parser *p, const char *subject)
{
    int fmt_idx = -1;
    for (int i = p->afe_n - 1; i >= 0; i--)
    {
        if (p->afe[i].marker)
            break;
        struct MiniNode *fn = p->afe[i].node;
        if (fn && fn->tag && h5_ci(fn->tag, subject) && open_index(p, fn) >= 0)
        {
            fmt_idx = i;
            break;
        }
    }
    if (fmt_idx < 0)
    {
        if (has_in_scope(p, subject, 0))
        {
            gen_implied_end(p, NULL);
            while (p->open_n)
            {
                struct MiniNode *n = open_pop(p);
                if (n->tag && h5_ci(n->tag, subject))
                    break;
            }
        }
        return;
    }
    struct MiniNode *fmt = p->afe[fmt_idx].node;
    int fi = open_index(p, fmt);
    if (fi < 0)
    {
        afe_remove(p, fmt);
        return;
    }
    struct MiniNode *furthest = NULL;
    int furthest_i = -1;
    for (int i = p->open_n - 1; i > fi; i--)
    {
        struct MiniNode *n = p->open[i];
        if (n->tag && h5_is_special(n->tag) && !h5_is_formatting(n->tag))
        {
            furthest = n;
            furthest_i = i;
            break;
        }
    }
    if (!furthest)
    {
        while (p->open_n)
        {
            struct MiniNode *n = open_pop(p);
            if (n == fmt)
                break;
        }
        afe_remove(p, fmt);
        return;
    }
    struct MiniNode *common_ancestor = (fi > 0) ? p->open[fi - 1] : p->doc->root;
    struct MiniNode *node = furthest;
    struct MiniNode *last_node = furthest;
    int node_i = furthest_i;
    int guard = 0;
    while (guard++ < 256)
    {
        if (node_i <= 0)
            break;
        node_i--;
        node = p->open[node_i];
        if (node == fmt)
            break;
        int ai = afe_index(p, node);
        if (ai >= 0)
        {
            struct MiniNode *clone = mini_node_clone(node, 0);
            if (!clone)
                break;
            p->afe[ai].node = clone;
            p->open[node_i] = clone;
            mini_node_append_child(last_node, clone);
            struct MiniNode *c = node->first_child;
            while (c)
            {
                struct MiniNode *nx = c->next_sibling;
                mini_node_remove_child(node, c);
                mini_node_append_child(clone, c);
                c = nx;
            }
            last_node = clone;
        }
        else
        {
            open_remove(p, node);
        }
    }
    if (last_node)
    {
        if (last_node->parent && last_node->parent != common_ancestor)
            mini_node_remove_child(last_node->parent, last_node);
        if (last_node->parent != common_ancestor)
            mini_node_append_child(common_ancestor, last_node);
    }
    open_remove(p, fmt);
    afe_remove(p, fmt);
    {
        struct MiniNode *newfmt = mini_node_clone(fmt, 0);
        if (newfmt)
        {
            struct MiniNode *c = last_node ? last_node->first_child : NULL;
            while (c)
            {
                struct MiniNode *nx = c->next_sibling;
                mini_node_remove_child(last_node, c);
                mini_node_append_child(newfmt, c);
                c = nx;
            }
            if (last_node)
                mini_node_append_child(last_node, newfmt);
            open_push(p, newfmt);
            afe_push_typed(p, 0, newfmt);
        }
    }
}

/* ================================================================== */
/* Insertion-mode dispatch                                            */
/* ================================================================== */
static void ensure_html(struct MiniHtml5Parser *p)
{
    struct MiniNode *html = NULL;
    for (struct MiniNode *c = p->doc->root->first_child; c; c = c->next_sibling)
        if (c->tag && h5_ci(c->tag, "html"))
        {
            html = c;
            break;
        }
    if (html)
        return;
    html = mini_node_create_element("html");
    struct MiniNode *body = p->doc->body;
    if (body && body->parent == p->doc->root)
        mini_node_remove_child(p->doc->root, body);
    mini_node_append_child(p->doc->root, html);
    p->head_element = mini_node_create_element("head");
    mini_node_append_child(html, p->head_element);
    if (body)
        mini_node_append_child(html, body);
    else
    {
        p->doc->body = mini_node_create_element("body");
        mini_node_append_child(html, p->doc->body);
    }
    open_push(p, html);
}

static void in_body_start(struct MiniHtml5Parser *p, H5Token *t)
{
    const char *tag = t->name;
    if (h5_ci(tag, "html"))
    {
        struct MiniNode *html_node = p->doc->root ? p->doc->root->first_child : NULL;
        while (html_node && (!html_node->tag || strcmp(html_node->tag, "html")))
            html_node = html_node->next_sibling;
        if (html_node)
        {
            for (int i = 0; i < t->n_attr; i++)
            {
                if (!mini_node_get_attribute(html_node, t->attrs[i].n))
                    mini_node_set_attribute(html_node, t->attrs[i].n, t->attrs[i].v ? t->attrs[i].v : "");
            }
        }
        return;
    }
    if (h5_ci(tag, "head"))
        return;
    if (h5_ci(tag, "base") || h5_ci(tag, "basefont") || h5_ci(tag, "bgsound") ||
        h5_ci(tag, "link") || h5_ci(tag, "meta") || h5_ci(tag, "title"))
    {
        struct MiniNode *n = create_element_for_token(p, t);
        if (n)
        {
            mini_node_append_child(p->head_element ? p->head_element : current_node(p), n);
            if (!mini_element_info(tag)->is_void)
                open_push(p, n);
        }
        return;
    }
    if (h5_ci(tag, "style") || h5_ci(tag, "script"))
    {
        struct MiniNode *n = create_element_for_token(p, t);
        if (n)
        {
            mini_node_append_child(current_node(p), n);
            open_push(p, n);
        }
        return;
    }
    if (h5_ci(tag, "body"))
    {
        if (p->doc && p->doc->body)
        {
            for (int i = 0; i < t->n_attr; i++)
            {
                if (!mini_node_get_attribute(p->doc->body, t->attrs[i].n))
                    mini_node_set_attribute(p->doc->body, t->attrs[i].n, t->attrs[i].v ? t->attrs[i].v : "");
            }
        }
        return;
    }
    if (h5_closes_p(tag))
        close_p_if_open(p);
    if (h5_ci(tag, "li"))
    {
        if (has_in_scope(p, "li", 0))
        {
            gen_implied_end(p, "li");
            pop_until(p, "li");
        }
    }
    if (h5_ci(tag, "dd") || h5_ci(tag, "dt"))
    {
        if (has_in_scope(p, "dd", 0))
        {
            gen_implied_end(p, "dd");
            pop_until(p, "dd");
        }
        if (has_in_scope(p, "dt", 0))
        {
            gen_implied_end(p, "dt");
            pop_until(p, "dt");
        }
    }
    if (h5_ci(tag, "rt") || h5_ci(tag, "rp"))
    {
        if (has_in_scope(p, "rt", 0))
        {
            gen_implied_end(p, "rt");
            pop_until(p, "rt");
        }
        if (has_in_scope(p, "rp", 0))
        {
            gen_implied_end(p, "rp");
            pop_until(p, "rp");
        }
    }
    if (h5_ci(tag, "optgroup") || h5_ci(tag, "option"))
    {
        if (current_node(p) && current_node(p)->tag && h5_ci(current_node(p)->tag, "option"))
            open_pop(p);
    }
    if (h5_ci(tag, "a"))
    {
        if (has_in_scope(p, "a", 0))
        {
            h5_report_diag(p, MINI_PARSE_SEV_WARN, "nested-a", "Nested <a> tag is invalid; closing previous <a>");
            adoption_agency(p, "a");
            open_remove(p, current_node(p));
        }
    }
    if (h5_ci(tag, "table"))
    {
        struct MiniNode *n = create_element_for_token(p, t);
        if (n)
        {
            mini_node_append_child(current_node(p), n);
            open_push(p, n);
            p->table_foster = n;
            p->mode = M_IN_TABLE;
        }
        return;
    }
    if (h5_is_formatting(tag))
    {
        reconstruct_active_formatting(p);
        struct MiniNode *n = create_element_for_token(p, t);
        if (n)
        {
            insert_element(p, current_node(p), n);
            afe_push_typed(p, 0, n);
        }
        return;
    }
    reconstruct_active_formatting(p);
    struct MiniNode *n = create_element_for_token(p, t);
    if (n)
    {
        if (t->is_self_closing || mini_element_info(tag)->is_void)
        {
            mini_node_append_child(current_node(p), n);
        }
        else
        {
            insert_element(p, current_node(p), n);
        }
    }
}

static void in_body_end(struct MiniHtml5Parser *p, H5Token *t)
{
    const char *tag = t->name;
    if (h5_is_formatting(tag))
    {
        adoption_agency(p, tag);
        return;
    }
    if (h5_ci(tag, "p"))
    {
        gen_implied_end(p, "p");
        if (has_in_scope(p, "p", 0))
            pop_until(p, "p");
        return;
    }
    if (h5_ci(tag, "li") || h5_ci(tag, "dd") || h5_ci(tag, "dt") || h5_ci(tag, "rt") || h5_ci(tag, "rp"))
    {
        gen_implied_end(p, tag);
        if (has_in_scope(p, tag, 0))
            pop_until(p, tag);
        return;
    }
    if (h5_ci(tag, "body"))
    {
        p->mode = M_AFTER_BODY;
        return;
    }
    if (h5_ci(tag, "html"))
    {
        p->mode = M_AFTER_BODY;
        return;
    }
    if (has_in_scope(p, tag, 0))
    {
        gen_implied_end(p, NULL);
        while (p->open_n)
        {
            struct MiniNode *n = open_pop(p);
            if (n->tag && h5_ci(n->tag, tag))
                break;
        }
    }
    else
    {
        h5_report_diag(p, MINI_PARSE_SEV_WARN, "stray-end-tag", "Stray closing tag </%s> ignored", tag ? tag : "");
    }
}

static void in_table_start(struct MiniHtml5Parser *p, H5Token *t)
{
    const char *tag = t->name;
    if (h5_ci(tag, "caption"))
    {
        pop_until(p, "table");
        struct MiniNode *n = create_element_for_token(p, t);
        if (n)
        {
            mini_node_append_child(current_node(p), n);
            open_push(p, n);
            afe_marker(p);
            p->mode = M_IN_BODY;
        }
        return;
    }
    if (h5_ci(tag, "colgroup"))
    {
        struct MiniNode *n = create_element_for_token(p, t);
        if (n)
        {
            mini_node_append_child(current_node(p), n);
            open_push(p, n);
        }
        return;
    }
    if (h5_ci(tag, "col"))
    {
        struct MiniNode *n = create_element_for_token(p, t);
        if (n)
        {
            mini_node_append_child(current_node(p), n);
        }
        return;
    }
    if (h5_ci(tag, "thead") || h5_ci(tag, "tfoot") || h5_ci(tag, "tbody"))
    {
        struct MiniNode *n = create_element_for_token(p, t);
        if (n)
        {
            mini_node_append_child(current_node(p), n);
            open_push(p, n);
            p->mode = M_IN_TABLE_BODY;
        }
        return;
    }
    if (h5_ci(tag, "tr"))
    {
        struct MiniNode *n = create_element_for_token(p, t);
        if (n)
        {
            mini_node_append_child(current_node(p), n);
            open_push(p, n);
            p->mode = M_IN_ROW;
        }
        return;
    }
    if (h5_ci(tag, "th") || h5_ci(tag, "td"))
    {
        struct MiniNode *tr = mini_node_create_element("tr");
        mini_node_append_child(current_node(p), tr);
        open_push(p, tr);
        p->mode = M_IN_ROW;
        struct MiniNode *c = create_element_for_token(p, t);
        if (c)
        {
            mini_node_append_child(current_node(p), c);
            open_push(p, c);
            p->mode = M_IN_CELL;
        }
        return;
    }
    reconstruct_active_formatting(p);
    struct MiniNode *n = create_element_for_token(p, t);
    if (n)
    {
        if (t->is_self_closing || mini_element_info(tag)->is_void)
        {
            struct MiniNode *table = p->table_foster;
            struct MiniNode *parent = table ? table->parent : current_node(p);
            if (table && parent)
                mini_node_insert_before(parent, n, table);
            else
                mini_node_append_child(parent, n);
        }
        else
        {
            foster_insert(p, n);
        }
    }
}

static void in_table_end(struct MiniHtml5Parser *p, H5Token *t)
{
    const char *tag = t->name;
    if (h5_ci(tag, "table"))
    {
        if (has_in_scope(p, "table", 1))
        {
            pop_until(p, "table");
            p->table_foster = NULL;
            p->mode = M_IN_BODY;
        }
        return;
    }
    if (h5_ci(tag, "tbody") || h5_ci(tag, "tfoot") || h5_ci(tag, "thead"))
    {
        if (has_in_scope(p, tag, 1))
            pop_until(p, tag), p->mode = M_IN_TABLE;
        return;
    }
    if (h5_ci(tag, "tr"))
    {
        if (has_in_scope(p, tag, 1))
            pop_until(p, tag), p->mode = M_IN_TABLE_BODY;
        return;
    }
    if (h5_ci(tag, "th") || h5_ci(tag, "td"))
    {
        if (has_in_scope(p, tag, 1))
            pop_until(p, tag), p->mode = M_IN_ROW;
        return;
    }
    in_body_end(p, t);
}

static void in_row_start(struct MiniHtml5Parser *p, H5Token *t)
{
    const char *tag = t->name;
    if (h5_ci(tag, "th") || h5_ci(tag, "td"))
    {
        struct MiniNode *n = create_element_for_token(p, t);
        if (n)
        {
            mini_node_append_child(current_node(p), n);
            open_push(p, n);
            p->mode = M_IN_CELL;
        }
        return;
    }
    if (h5_ci(tag, "tr"))
    {
        if (has_in_scope(p, "tr", 1))
            pop_until(p, "tr");
        struct MiniNode *n = create_element_for_token(p, t);
        if (n)
        {
            mini_node_append_child(current_node(p), n);
            open_push(p, n);
        }
        return;
    }
    if (has_in_scope(p, "tr", 1))
        pop_until(p, "tr");
    p->mode = M_IN_TABLE_BODY;
    in_table_start(p, t);
}

static void in_cell_start(struct MiniHtml5Parser *p, H5Token *t)
{
    const char *tag = t->name;
    if (tag && (h5_ci(tag, "td") || h5_ci(tag, "th") || h5_ci(tag, "tr") ||
                h5_ci(tag, "tbody") || h5_ci(tag, "thead") || h5_ci(tag, "tfoot") ||
                h5_ci(tag, "caption") || h5_ci(tag, "col") || h5_ci(tag, "colgroup")))
    {
        if (has_in_scope(p, "td", 1) || has_in_scope(p, "th", 1))
        {
            gen_implied_end(p, NULL);
            if (has_in_scope(p, "td", 1))
                pop_until(p, "td");
            else if (has_in_scope(p, "th", 1))
                pop_until(p, "th");
            p->mode = M_IN_ROW;
            in_row_start(p, t);
            return;
        }
    }
    in_body_start(p, t);
}

static void in_cell_end(struct MiniHtml5Parser *p, H5Token *t)
{
    const char *tag = t->name;
    if (h5_ci(tag, "td") || h5_ci(tag, "th"))
    {
        if (has_in_scope(p, tag, 1))
        {
            gen_implied_end(p, NULL);
            pop_until(p, tag);
            p->mode = M_IN_ROW;
        }
        return;
    }
    if (h5_ci(tag, "tr") || h5_ci(tag, "tbody") || h5_ci(tag, "thead") || h5_ci(tag, "tfoot") || h5_ci(tag, "table"))
    {
        if (has_in_scope(p, "td", 1) || has_in_scope(p, "th", 1))
        {
            gen_implied_end(p, NULL);
            if (has_in_scope(p, "td", 1))
                pop_until(p, "td");
            else if (has_in_scope(p, "th", 1))
                pop_until(p, "th");
            p->mode = M_IN_ROW;
            in_table_end(p, t);
        }
        return;
    }
    in_body_end(p, t);
}

static void process_token(struct MiniHtml5Parser *p, H5Token *t)
{
    switch (p->mode)
    {
    case M_INITIAL:
        if (t->type == TT_DOCTYPE)
        {
            p->mode = M_BEFORE_HTML;
            return;
        }
        if (t->type == TT_CHAR)
        {
            const char *s = t->name ? t->name : "";
            while (*s && isspace((unsigned char)*s)) s++;
            if (!*s) return;
        }
        if (t->type == TT_COMMENT)
        {
            insert_comment(p, p->doc->root, t->name ? t->name : "");
            return;
        }
        p->mode = M_BEFORE_HTML;
        process_token(p, t);
        return;
    case M_BEFORE_HTML:
        if (t->type == TT_START && h5_ci(t->name, "html"))
        {
            struct MiniNode *h = create_element_for_token(p, t);
            if (h)
            {
                mini_node_append_child(p->doc->root, h);
                open_push(p, h);
                p->mode = M_BEFORE_HEAD;
            }
            return;
        }
        if (t->type == TT_CHAR)
        {
            const char *s = t->name ? t->name : "";
            while (*s && isspace((unsigned char)*s)) s++;
            if (!*s) return;
        }
        if (t->type == TT_COMMENT)
        {
            insert_comment(p, p->doc->root, t->name ? t->name : "");
            return;
        }
        {
            struct MiniNode *h = mini_node_create_element("html");
            mini_node_append_child(p->doc->root, h);
            open_push(p, h);
            p->mode = M_BEFORE_HEAD;
            process_token(p, t);
            return;
        }
    case M_BEFORE_HEAD:
        if (t->type == TT_START && h5_ci(t->name, "head"))
        {
            struct MiniNode *h = create_element_for_token(p, t);
            if (h)
            {
                mini_node_append_child(current_node(p), h);
                open_push(p, h);
                p->head_element = h;
                p->mode = M_IN_HEAD;
            }
            return;
        }
        if (t->type == TT_CHAR)
        {
            const char *s = t->name ? t->name : "";
            while (*s && isspace((unsigned char)*s)) s++;
            if (!*s) return;
        }
        if (t->type == TT_COMMENT)
        {
            insert_comment(p, current_node(p), t->name ? t->name : "");
            return;
        }
        {
            struct MiniNode *h = mini_node_create_element("head");
            mini_node_append_child(current_node(p), h);
            open_push(p, h);
            p->head_element = h;
            p->mode = M_IN_HEAD;
            process_token(p, t);
            return;
        }
    case M_IN_HEAD:
        if (t->type == TT_START)
        {
            const char *tag = t->name;
            if (h5_ci(tag, "title") || h5_ci(tag, "style") || h5_ci(tag, "script"))
            {
                struct MiniNode *n = create_element_for_token(p, t);
                if (n)
                {
                    mini_node_append_child(current_node(p), n);
                    open_push(p, n);
                }
                return;
            }
            if (h5_ci(tag, "meta") || h5_ci(tag, "link") || h5_ci(tag, "base"))
            {
                struct MiniNode *n = create_element_for_token(p, t);
                if (n)
                    mini_node_append_child(current_node(p), n);
                return;
            }
            open_pop(p);
            p->mode = M_AFTER_HEAD;
            process_token(p, t);
            return;
        }
        if (t->type == TT_END)
        {
            const char *tag = t->name;
            if (h5_ci(tag, "head"))
            {
                open_pop(p);
                p->mode = M_AFTER_HEAD;
                return;
            }
            if (h5_ci(tag, "title") || h5_ci(tag, "style") || h5_ci(tag, "script"))
            {
                if (current_node(p)->tag && h5_ci(current_node(p)->tag, tag))
                {
                    open_pop(p);
                }
                return;
            }
            if (h5_ci(tag, "body") || h5_ci(tag, "html"))
            {
                open_pop(p);
                p->mode = M_AFTER_HEAD;
                process_token(p, t);
                return;
            }
            return;
        }
        if (t->type == TT_CHAR)
        {
            const char *s = t->name ? t->name : "";
            while (*s && isspace((unsigned char)*s)) s++;
            if (!*s)
                return;
            open_pop(p);
            p->mode = M_AFTER_HEAD;
            process_token(p, t);
            return;
        }
        if (t->type == TT_COMMENT)
        {
            insert_comment(p, current_node(p), t->name ? t->name : "");
            return;
        }
        open_pop(p);
        p->mode = M_AFTER_HEAD;
        process_token(p, t);
        return;
    case M_AFTER_HEAD:
        if (t->type == TT_START && h5_ci(t->name, "body"))
        {
            struct MiniNode *b = create_element_for_token(p, t);
            if (b)
            {
                mini_node_append_child(current_node(p), b);
                open_push(p, b);
                p->doc->body = b;
                p->mode = M_IN_BODY;
            }
            return;
        }
        if (t->type == TT_CHAR)
        {
            const char *s = t->name ? t->name : "";
            while (*s && isspace((unsigned char)*s)) s++;
            if (!*s) return;
        }
        if (t->type == TT_COMMENT)
        {
            insert_comment(p, current_node(p), t->name ? t->name : "");
            return;
        }
        if (t->type == TT_START && h5_ci(t->name, "head"))
        {
            return;
        }
        {
            struct MiniNode *b = mini_node_create_element("body");
            mini_node_append_child(current_node(p), b);
            open_push(p, b);
            p->doc->body = b;
            p->mode = M_IN_BODY;
            process_token(p, t);
            return;
        }
    case M_IN_BODY:
        if (t->type == TT_CHAR)
        {
            insert_char(p, t->name ? t->name : "");
            return;
        }
        if (t->type == TT_COMMENT)
        {
            insert_comment(p, current_node(p), t->name ? t->name : "");
            return;
        }
        if (t->type == TT_START)
        {
            in_body_start(p, t);
            return;
        }
        if (t->type == TT_END)
        {
            in_body_end(p, t);
            return;
        }
        return;
    case M_IN_TABLE:
        if (t->type == TT_START)
        {
            in_table_start(p, t);
            return;
        }
        if (t->type == TT_END)
        {
            in_table_end(p, t);
            return;
        }
        if (t->type == TT_CHAR)
        {
            insert_char(p, t->name ? t->name : "");
            return;
        }
        if (t->type == TT_COMMENT)
        {
            insert_comment(p, current_node(p), t->name ? t->name : "");
            return;
        }
        return;
    case M_IN_TABLE_BODY:
        if (t->type == TT_START && (h5_ci(t->name, "tr")))
        {
            struct MiniNode *n = create_element_for_token(p, t);
            if (n)
            {
                mini_node_append_child(current_node(p), n);
                open_push(p, n);
                p->mode = M_IN_ROW;
            }
            return;
        }
        if (t->type == TT_END && (h5_ci(t->name, "tbody") || h5_ci(t->name, "thead") || h5_ci(t->name, "tfoot")))
        {
            if (has_in_scope(p, t->name, 1))
                pop_until(p, t->name), p->mode = M_IN_TABLE;
            return;
        }
        if (t->type == TT_START)
        {
            if (has_in_scope(p, "tbody", 1) || has_in_scope(p, "thead", 1) || has_in_scope(p, "tfoot", 1))
                pop_until_one(p, "tbody", "thead", "tfoot");
            p->mode = M_IN_TABLE;
            in_table_start(p, t);
            return;
        }
        return;
    case M_IN_ROW:
        if (t->type == TT_START)
        {
            in_row_start(p, t);
            return;
        }
        if (t->type == TT_END)
        {
            in_table_end(p, t);
            return;
        }
        if (t->type == TT_CHAR)
        {
            insert_char(p, t->name ? t->name : "");
            return;
        }
        return;
    case M_IN_CELL:
        if (t->type == TT_START)
        {
            in_cell_start(p, t);
            return;
        }
        if (t->type == TT_END)
        {
            in_cell_end(p, t);
            return;
        }
        if (t->type == TT_CHAR)
        {
            insert_char(p, t->name ? t->name : "");
            return;
        }
        if (t->type == TT_COMMENT)
        {
            insert_comment(p, current_node(p), t->name ? t->name : "");
            return;
        }
        return;
    case M_IN_SELECT:
        if (t->type == TT_CHAR)
        {
            insert_char(p, t->name ? t->name : "");
            return;
        }
        if (t->type == TT_START)
        {
            in_body_start(p, t);
            return;
        }
        if (t->type == TT_END)
        {
            in_body_end(p, t);
            return;
        }
        return;
    case M_AFTER_BODY:
        if (t->type == TT_END && h5_ci(t->name, "html"))
        {
            p->mode = M_AFTER_AFTER_BODY;
            return;
        }
        p->mode = M_IN_BODY;
        process_token(p, t);
        return;
    case M_AFTER_AFTER_BODY:
        if (t->type == TT_CHAR)
        {
            insert_char(p, t->name ? t->name : "");
            return;
        }
        p->mode = M_IN_BODY;
        process_token(p, t);
        return;
    }
}

static void pop_until_one(struct MiniHtml5Parser *p, const char *a, const char *b, const char *c)
{
    const char *tags[4] = {a, b, c, NULL};
    for (int i = p->open_n - 1; i >= 0; i--)
    {
        struct MiniNode *n = p->open[i];
        if (n->tag)
        {
            for (int j = 0; j < 3 && tags[j]; j++)
                if (h5_ci(n->tag, tags[j]))
                {
                    while (p->open_n)
                    {
                        struct MiniNode *x = open_pop(p);
                        if (x == n)
                            break;
                    }
                    return;
                }
        }
    }
}

/* ================================================================== */
/* Tokenizer                                                          */
/* ================================================================== */
static void emit_char_len(struct MiniHtml5Parser *p, const char *s, size_t len)
{
    if (s && len)
        insert_char_len(p, s, len);
}

static void emit_char(struct MiniHtml5Parser *p, const char *s)
{
    if (s && *s)
        insert_char_len(p, s, strlen(s));
}

static void emit_ref(struct MiniHtml5Parser *p, const char *utf8)
{
    if (utf8 && *utf8)
        insert_char_len(p, utf8, strlen(utf8));
}

static void emit_current(struct MiniHtml5Parser *p)
{
    if (p->cur.type == TT_COMMENT || p->cur.type == TT_DOCTYPE)
    {
        if (!p->cur.name)
            tok_set_name(&p->cur, "");
        process_token(p, &p->cur);
        tok_reset(&p->cur);
        return;
    }
    if (!p->cur.name || !p->cur.name[0])
    {
        tok_reset(&p->cur);
        return;
    }
    int was_start = (p->cur.type == TT_START);
    for (char *q = p->cur.name; *q; q++)
        *q = (char)tolower((unsigned char)*q);

    if (!was_start)
    {
        if (p->cur.n_attr > 0)
            h5_report_diag(p, MINI_PARSE_SEV_WARN, "end-tag-with-attributes",
                           "End tag </%s> cannot have attributes", p->cur.name);
        if (p->cur.is_self_closing)
            h5_report_diag(p, MINI_PARSE_SEV_WARN, "end-tag-with-trailing-slash",
                           "End tag </%s> cannot be self-closing", p->cur.name);
    }

    process_token(p, &p->cur);
    tok_reset(&p->cur);
    if (was_start)
        maybe_enter_rawtext(p);
}

static void attr_finalize(struct MiniHtml5Parser *p)
{
    if (p->cur.n_attr > 0)
    {
        if (!p->cur.attrs[p->cur.n_attr - 1].v[0] && p->acc_n > 0)
            set_last_attr_value(&p->cur, p->acc);
    }
    acc_reset(p);
}

#define H5_ADVANCE_POS(ch) do { \
    if ((ch) == '\n') { p->line++; p->col = 1; } \
    else { p->col++; } \
} while(0)

static void tokenize_run(struct MiniHtml5Parser *p, const char *buf, size_t len)
{
    size_t i = 0;
    while (i < len)
    {
        char c = buf[i];
        switch (p->ts)
        {
        case TS_DATA:
            if (c == '&')
            {
                char out[8] = {0};
                size_t used = h5_decode_ref_diag(p, buf + i, len - i, out, sizeof(out));
                if (used > 0)
                {
                    emit_ref(p, out[0] ? out : "");
                    for (size_t u = 0; u < used; u++)
                        H5_ADVANCE_POS(buf[i + u]);
                    i += used;
                    continue;
                }
                emit_char_len(p, "&", 1);
                H5_ADVANCE_POS('&');
                i++;
                continue;
            }
            if (c == '<')
            {
                p->ts = TS_TAG_OPEN;
                p->cur_tok_line = p->line;
                p->cur_tok_col = p->col;
                H5_ADVANCE_POS('<');
                i++;
                continue;
            }
            {
                size_t start = i;
                while (i < len && buf[i] != '<' && buf[i] != '&')
                {
                    H5_ADVANCE_POS(buf[i]);
                    i++;
                }
                size_t text_len = i - start;
                if (text_len > 0)
                    insert_char_len(p, buf + start, text_len);
                continue;
            }
        case TS_TAG_OPEN:
            if (c == '!')
            {
                p->ts = TS_MARKUP_DECL_OPEN;
                H5_ADVANCE_POS('!');
                i++;
                continue;
            }
            if (c == '/')
            {
                p->ts = TS_END_TAG_OPEN;
                p->cur_tok_line = p->line;
                p->cur_tok_col = p->col;
                H5_ADVANCE_POS('/');
                i++;
                continue;
            }
            if (isalpha((unsigned char)c))
            {
                p->cur.type = TT_START;
                acc_reset(p);
                acc_putc(p, (char)tolower((unsigned char)c));
                p->ts = TS_TAG_NAME;
                H5_ADVANCE_POS(c);
                i++;
                continue;
            }
            if (c == '?')
            {
                h5_report_diag(p, MINI_PARSE_SEV_WARN, "bogus-comment",
                               "XML processing instruction <? ... > treated as comment");
                p->ts = TS_BOGUS_COMMENT;
                acc_reset(p);
                H5_ADVANCE_POS('?');
                i++;
                continue;
            }
            emit_char_len(p, "<", 1);
            p->ts = TS_DATA;
            continue;
        case TS_END_TAG_OPEN:
            if (isalpha((unsigned char)c))
            {
                p->cur.type = TT_END;
                acc_reset(p);
                acc_putc(p, (char)tolower((unsigned char)c));
                p->ts = TS_TAG_NAME;
                H5_ADVANCE_POS(c);
                i++;
                continue;
            }
            if (c == '>')
            {
                h5_report_diag(p, MINI_PARSE_SEV_WARN, "empty-end-tag", "Empty end tag </> ignored");
                p->ts = TS_DATA;
                H5_ADVANCE_POS('>');
                i++;
                continue;
            }
            p->ts = TS_BOGUS_COMMENT;
            acc_reset(p);
            continue;
        case TS_TAG_NAME:
        {
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f')
            {
                tok_set_name(&p->cur, p->acc_n ? p->acc : "");
                acc_reset(p);
                p->ts = TS_BEFORE_ATTR_NAME;
                H5_ADVANCE_POS(c);
                i++;
                continue;
            }
            if (c == '/')
            {
                tok_set_name(&p->cur, p->acc_n ? p->acc : "");
                acc_reset(p);
                p->ts = TS_SELF_CLOSING;
                H5_ADVANCE_POS('/');
                i++;
                continue;
            }
            if (c == '>')
            {
                tok_set_name(&p->cur, p->acc_n ? p->acc : "");
                acc_reset(p);
                p->ts = TS_DATA;
                emit_current(p);
                H5_ADVANCE_POS('>');
                i++;
                continue;
            }
            acc_putc(p, (char)tolower((unsigned char)c));
            H5_ADVANCE_POS(c);
            i++;
            continue;
        }
        case TS_BEFORE_ATTR_NAME:
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f')
            {
                H5_ADVANCE_POS(c);
                i++;
                continue;
            }
            if (c == '/' || c == '>')
            {
                p->ts = TS_AFTER_ATTR_NAME;
                continue;
            }
            acc_reset(p);
            acc_putc(p, (char)tolower((unsigned char)c));
            p->ts = TS_ATTR_NAME;
            H5_ADVANCE_POS(c);
            i++;
            continue;
        case TS_ATTR_NAME:
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f')
            {
                int dup = 0;
                for (int a = 0; a < p->cur.n_attr; a++)
                {
                    if (p->cur.attrs[a].n && strcmp(p->cur.attrs[a].n, p->acc) == 0)
                    {
                        dup = 1;
                        break;
                    }
                }
                if (dup)
                    h5_report_diag(p, MINI_PARSE_SEV_WARN, "duplicate-attribute",
                                   "Duplicate attribute '%s' on tag <%s>", p->acc, p->cur.name ? p->cur.name : "");
                else
                    tok_add_attr(&p->cur, p->acc_n ? p->acc : "", "");
                acc_reset(p);
                p->ts = TS_AFTER_ATTR_NAME;
                H5_ADVANCE_POS(c);
                i++;
                continue;
            }
            if (c == '/' || c == '>')
            {
                int dup = 0;
                for (int a = 0; a < p->cur.n_attr; a++)
                {
                    if (p->cur.attrs[a].n && strcmp(p->cur.attrs[a].n, p->acc) == 0)
                    {
                        dup = 1;
                        break;
                    }
                }
                if (dup)
                    h5_report_diag(p, MINI_PARSE_SEV_WARN, "duplicate-attribute",
                                   "Duplicate attribute '%s' on tag <%s>", p->acc, p->cur.name ? p->cur.name : "");
                else
                    tok_add_attr(&p->cur, p->acc_n ? p->acc : "", "");
                acc_reset(p);
                p->ts = TS_AFTER_ATTR_NAME;
                continue;
            }
            if (c == '=')
            {
                int dup = 0;
                for (int a = 0; a < p->cur.n_attr; a++)
                {
                    if (p->cur.attrs[a].n && strcmp(p->cur.attrs[a].n, p->acc) == 0)
                    {
                        dup = 1;
                        break;
                    }
                }
                if (dup)
                    h5_report_diag(p, MINI_PARSE_SEV_WARN, "duplicate-attribute",
                                   "Duplicate attribute '%s' on tag <%s>", p->acc, p->cur.name ? p->cur.name : "");
                else
                    tok_add_attr(&p->cur, p->acc_n ? p->acc : "", "");
                acc_reset(p);
                p->ts = TS_BEFORE_ATTR_VALUE;
                H5_ADVANCE_POS('=');
                i++;
                continue;
            }
            acc_putc(p, (char)tolower((unsigned char)c));
            H5_ADVANCE_POS(c);
            i++;
            continue;
        case TS_AFTER_ATTR_NAME:
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f')
            {
                H5_ADVANCE_POS(c);
                i++;
                continue;
            }
            if (c == '/')
            {
                p->ts = TS_SELF_CLOSING;
                H5_ADVANCE_POS('/');
                i++;
                continue;
            }
            if (c == '=')
            {
                p->ts = TS_BEFORE_ATTR_VALUE;
                H5_ADVANCE_POS('=');
                i++;
                continue;
            }
            if (c == '>')
            {
                p->ts = TS_DATA;
                emit_current(p);
                H5_ADVANCE_POS('>');
                i++;
                continue;
            }
            acc_reset(p);
            acc_putc(p, (char)tolower((unsigned char)c));
            p->ts = TS_ATTR_NAME;
            H5_ADVANCE_POS(c);
            i++;
            continue;
        case TS_BEFORE_ATTR_VALUE:
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f')
            {
                H5_ADVANCE_POS(c);
                i++;
                continue;
            }
            if (c == '"')
            {
                p->ts = TS_ATTR_VALUE_DQ;
                H5_ADVANCE_POS('"');
                i++;
                continue;
            }
            if (c == '\'')
            {
                p->ts = TS_ATTR_VALUE_SQ;
                H5_ADVANCE_POS('\'');
                i++;
                continue;
            }
            if (c == '>')
            {
                h5_report_diag(p, MINI_PARSE_SEV_WARN, "missing-attribute-value", "Missing attribute value before '>'");
                p->ts = TS_DATA;
                emit_current(p);
                H5_ADVANCE_POS('>');
                i++;
                continue;
            }
            p->ts = TS_ATTR_VALUE_UQ;
            continue;
        case TS_ATTR_VALUE_DQ:
            if (c == '"')
            {
                set_last_attr_value(&p->cur, p->acc_n ? p->acc : "");
                acc_reset(p);
                p->ts = TS_AFTER_ATTR_VALUE;
                H5_ADVANCE_POS('"');
                i++;
                continue;
            }
            if (c == '&')
            {
                char o[8] = {0};
                size_t u = h5_decode_ref_diag(p, buf + i, len - i, o, sizeof(o));
                if (u > 0)
                {
                    acc_puts(p, o[0] ? o : "");
                    for (size_t z = 0; z < u; z++)
                        H5_ADVANCE_POS(buf[i + z]);
                    i += u;
                    continue;
                }
                acc_putc(p, '&');
                H5_ADVANCE_POS('&');
                i++;
                continue;
            }
            acc_putc(p, c);
            H5_ADVANCE_POS(c);
            i++;
            continue;
        case TS_ATTR_VALUE_SQ:
            if (c == '\'')
            {
                set_last_attr_value(&p->cur, p->acc_n ? p->acc : "");
                acc_reset(p);
                p->ts = TS_AFTER_ATTR_VALUE;
                H5_ADVANCE_POS('\'');
                i++;
                continue;
            }
            if (c == '&')
            {
                char o[8] = {0};
                size_t u = h5_decode_ref_diag(p, buf + i, len - i, o, sizeof(o));
                if (u > 0)
                {
                    acc_puts(p, o[0] ? o : "");
                    for (size_t z = 0; z < u; z++)
                        H5_ADVANCE_POS(buf[i + z]);
                    i += u;
                    continue;
                }
                acc_putc(p, '&');
                H5_ADVANCE_POS('&');
                i++;
                continue;
            }
            acc_putc(p, c);
            H5_ADVANCE_POS(c);
            i++;
            continue;
        case TS_ATTR_VALUE_UQ:
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f')
            {
                set_last_attr_value(&p->cur, p->acc_n ? p->acc : "");
                acc_reset(p);
                p->ts = TS_AFTER_ATTR_VALUE;
                H5_ADVANCE_POS(c);
                i++;
                continue;
            }
            if (c == '>')
            {
                set_last_attr_value(&p->cur, p->acc_n ? p->acc : "");
                acc_reset(p);
                p->ts = TS_DATA;
                emit_current(p);
                H5_ADVANCE_POS('>');
                i++;
                continue;
            }
            if (c == '&')
            {
                char o[8] = {0};
                size_t u = h5_decode_ref_diag(p, buf + i, len - i, o, sizeof(o));
                if (u > 0)
                {
                    acc_puts(p, o[0] ? o : "");
                    for (size_t z = 0; z < u; z++)
                        H5_ADVANCE_POS(buf[i + z]);
                    i += u;
                    continue;
                }
                acc_putc(p, '&');
                H5_ADVANCE_POS('&');
                i++;
                continue;
            }
            acc_putc(p, c);
            H5_ADVANCE_POS(c);
            i++;
            continue;
        case TS_AFTER_ATTR_VALUE:
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f')
            {
                p->ts = TS_BEFORE_ATTR_NAME;
                H5_ADVANCE_POS(c);
                i++;
                continue;
            }
            if (c == '/')
            {
                p->ts = TS_SELF_CLOSING;
                H5_ADVANCE_POS('/');
                i++;
                continue;
            }
            if (c == '>')
            {
                p->ts = TS_DATA;
                emit_current(p);
                H5_ADVANCE_POS('>');
                i++;
                continue;
            }
            h5_report_diag(p, MINI_PARSE_SEV_WARN, "missing-whitespace-between-attributes",
                           "Missing whitespace between attributes");
            p->ts = TS_BEFORE_ATTR_NAME;
            continue;
        case TS_SELF_CLOSING:
            if (c == '>')
            {
                p->cur.is_self_closing = 1;
                p->ts = TS_DATA;
                emit_current(p);
                H5_ADVANCE_POS('>');
                i++;
                continue;
            }
            h5_report_diag(p, MINI_PARSE_SEV_WARN, "unexpected-char-in-self-closing",
                           "Unexpected character '%c' after '/' in tag", c);
            p->ts = TS_BEFORE_ATTR_NAME;
            continue;
        case TS_BOGUS_COMMENT:
            if (c == '>')
            {
                tok_set_name(&p->cur, p->acc_n ? p->acc : "");
                p->cur.type = TT_COMMENT;
                p->ts = TS_DATA;
                emit_current(p);
                H5_ADVANCE_POS('>');
                i++;
                continue;
            }
            acc_putc(p, c);
            H5_ADVANCE_POS(c);
            i++;
            continue;
        case TS_MARKUP_DECL_OPEN:
            if (c == '-' && i + 1 < len && buf[i + 1] == '-')
            {
                p->ts = TS_COMMENT;
                acc_reset(p);
                H5_ADVANCE_POS('-');
                H5_ADVANCE_POS('-');
                i += 2;
                continue;
            }
            if (c == '[' && i + 6 < len && memcmp(buf + i, "[CDATA[", 7) == 0)
            {
                for (int z = 0; z < 7; z++)
                    H5_ADVANCE_POS(buf[i + z]);
                i += 7;
                size_t start = i;
                while (i + 2 < len && !(buf[i] == ']' && buf[i + 1] == ']' && buf[i + 2] == '>'))
                {
                    H5_ADVANCE_POS(buf[i]);
                    i++;
                }
                size_t cdata_len = i - start;
                if (cdata_len > 0)
                    insert_char_len(p, buf + start, cdata_len);
                if (i + 2 < len)
                {
                    for (int z = 0; z < 3; z++)
                        H5_ADVANCE_POS(buf[i + z]);
                    i += 3;
                }
                p->ts = TS_DATA;
                continue;
            }
            if (tolower((unsigned char)c) == 'd')
            {
                p->ts = TS_DOCTYPE;
                acc_reset(p);
                acc_putc(p, c);
                H5_ADVANCE_POS(c);
                i++;
                continue;
            }
            h5_report_diag(p, MINI_PARSE_SEV_WARN, "bogus-comment", "Bogus markup declaration treated as comment");
            p->ts = TS_BOGUS_COMMENT;
            acc_reset(p);
            continue;
        case TS_COMMENT:
            if (c == '-' && i + 2 < len && buf[i + 1] == '-' && buf[i + 2] == '>')
            {
                tok_set_name(&p->cur, p->acc_n ? p->acc : "");
                p->cur.type = TT_COMMENT;
                p->ts = TS_DATA;
                emit_current(p);
                for (int z = 0; z < 3; z++)
                    H5_ADVANCE_POS(buf[i + z]);
                i += 3;
                continue;
            }
            acc_putc(p, c);
            H5_ADVANCE_POS(c);
            i++;
            continue;
        case TS_DOCTYPE:
            if (c == '>')
            {
                tok_set_name(&p->cur, p->acc_n ? p->acc : "");
                p->cur.type = TT_DOCTYPE;
                p->ts = TS_DATA;
                emit_current(p);
                H5_ADVANCE_POS('>');
                i++;
                continue;
            }
            acc_putc(p, c);
            H5_ADVANCE_POS(c);
            i++;
            continue;
        case TS_RAWTEXT:
        case TS_RCDATA:
        {
            const char *end = NULL;
            for (size_t k = i; k + 1 < len; k++)
            {
                if (buf[k] == '<' && buf[k + 1] == '/')
                {
                    size_t m = k + 2;
                    size_t j = 0;
                    const char *tn = p->cur.name ? p->cur.name : "";
                    while (tn[j] && m < len && tolower((unsigned char)buf[m]) == tolower((unsigned char)tn[j]))
                    {
                        m++;
                        j++;
                    }
                    if (tn[j] == 0)
                    {
                        end = buf + k;
                        break;
                    }
                }
            }
            if (end)
            {
                size_t k = (size_t)(end - buf);
                if (k > i)
                {
                    if (p->ts == TS_RCDATA)
                    {
                        char outbuf[1024];
                        size_t oi = 0;
                        size_t si = i;
                        while (si < k)
                        {
                            if (buf[si] == '&')
                            {
                                char o[8] = {0};
                                size_t u = h5_decode_ref_diag(p, buf + si, k - si, o, sizeof(o));
                                if (u > 0)
                                {
                                    for (size_t z = 0; z < u && o[z] && oi + 1 < sizeof(outbuf);)
                                        outbuf[oi++] = o[z++];
                                    for (size_t z = 0; z < u; z++)
                                        H5_ADVANCE_POS(buf[si + z]);
                                    si += u;
                                    continue;
                                }
                                outbuf[oi++] = '&';
                                H5_ADVANCE_POS('&');
                                si++;
                            }
                            else
                            {
                                if (oi + 1 < sizeof(outbuf))
                                    outbuf[oi++] = buf[si];
                                H5_ADVANCE_POS(buf[si]);
                                si++;
                            }
                            if (oi >= sizeof(outbuf) - 4)
                            {
                                outbuf[oi] = 0;
                                insert_char_len(p, outbuf, oi);
                                oi = 0;
                            }
                        }
                        if (oi)
                        {
                            outbuf[oi] = 0;
                            insert_char_len(p, outbuf, oi);
                        }
                    }
                    else
                    {
                        for (size_t z = i; z < k; z++)
                            H5_ADVANCE_POS(buf[z]);
                        insert_char_len(p, buf + i, k - i);
                    }
                }
                size_t m = k + 2;
                const char *tn = p->cur.name ? p->cur.name : "";
                while (*tn && m < len && tolower((unsigned char)buf[m]) == tolower((unsigned char)*tn))
                {
                    H5_ADVANCE_POS(buf[m]);
                    m++;
                    tn++;
                }
                while (m < len && buf[m] != '>')
                {
                    H5_ADVANCE_POS(buf[m]);
                    m++;
                }
                if (m < len)
                {
                    H5_ADVANCE_POS('>');
                    m++;
                }
                i = m;
                {
                    H5Token t;
                    memset(&t, 0, sizeof(t));
                    t.type = TT_END;
                    tok_set_name(&t, p->cur.name ? p->cur.name : "");
                    process_token(p, &t);
                    tok_destroy(&t);
                    tok_reset(&p->cur);
                }
                p->ts = TS_DATA;
                continue;
            }
            {
                size_t k = len;
                if (p->ts == TS_RCDATA)
                {
                    char outbuf[1024];
                    size_t oi = 0;
                    size_t si = i;
                    while (si < k)
                    {
                        if (buf[si] == '&')
                        {
                            char o[8] = {0};
                            size_t u = h5_decode_ref_diag(p, buf + si, k - si, o, sizeof(o));
                            if (u > 0)
                            {
                                for (size_t z = 0; z < u && o[z] && oi + 1 < sizeof(outbuf);)
                                    outbuf[oi++] = o[z++];
                                for (size_t z = 0; z < u; z++)
                                    H5_ADVANCE_POS(buf[si + z]);
                                si += u;
                                continue;
                            }
                            outbuf[oi++] = '&';
                            H5_ADVANCE_POS('&');
                            si++;
                        }
                        else
                        {
                            if (oi + 1 < sizeof(outbuf))
                                outbuf[oi++] = buf[si];
                            H5_ADVANCE_POS(buf[si]);
                            si++;
                        }
                        if (oi >= sizeof(outbuf) - 4)
                        {
                            outbuf[oi] = 0;
                            insert_char_len(p, outbuf, oi);
                            oi = 0;
                        }
                    }
                    if (oi)
                    {
                        outbuf[oi] = 0;
                        insert_char_len(p, outbuf, oi);
                    }
                }
                else
                {
                    for (size_t z = i; z < k; z++)
                        H5_ADVANCE_POS(buf[z]);
                    insert_char_len(p, buf + i, k - i);
                }
                i = len;
            }
            continue;
        }
        case TS_EOF:
            i = len;
            continue;
        }
    }
}
#undef H5_ADVANCE_POS

static void set_last_attr_value(H5Token *t, const char *v)
{
    if (t->n_attr == 0)
        return;
    size_t vl = v ? strlen(v) : 0;
    int idx = t->n_attr - 1;
    if (vl + 1 > t->attrs[idx].v_cap)
    {
        size_t vcap = (vl + 1 < 32) ? 32 : (vl + 1) * 2;
        char *nb = realloc(t->attrs[idx].v, vcap);
        if (nb)
        {
            t->attrs[idx].v = nb;
            t->attrs[idx].v_cap = vcap;
        }
    }
    if (v && vl)
        memcpy(t->attrs[idx].v, v, vl);
    t->attrs[idx].v[vl] = 0;
}

static void maybe_enter_rawtext(struct MiniHtml5Parser *p)
{
    struct MiniNode *n = current_node(p);
    if (!n || !n->tag)
        return;
    if (h5_is_rawtext(n->tag))
    {
        p->ts = TS_RAWTEXT;
        tok_reset(&p->cur);
        tok_set_name(&p->cur, n->tag);
        p->cur.type = TT_START;
    }
    else if (h5_is_rcdata(n->tag))
    {
        p->ts = TS_RCDATA;
        tok_reset(&p->cur);
        tok_set_name(&p->cur, n->tag);
        p->cur.type = TT_START;
    }
}

/* ================================================================== */
/* Public API                                                         */
/* ================================================================== */
MiniHtml5Parser *mini_html5_init(struct MiniDocument *doc)
{
    if (!doc)
        return NULL;
    MiniHtml5Parser *p = calloc(1, sizeof(*p));
    if (!p)
        return NULL;
    p->doc = doc;
    p->line = 1;
    p->col = 1;
    p->ts = TS_DATA;
    p->mode = M_INITIAL;
    return p;
}

static size_t carryover_tail(struct MiniHtml5Parser *p, const char *buf, size_t len)
{
    switch (p->ts)
    {
    case TS_DATA:
    case TS_RCDATA:
    {
        for (size_t k = len; k > 0; k--)
        {
            if (buf[k - 1] == '&')
            {
                return len - (k - 1);
            }
            if (buf[k - 1] == ';')
                break;
        }
        return 0;
    }
    case TS_TAG_OPEN:
    case TS_END_TAG_OPEN:
    case TS_TAG_NAME:
    case TS_BEFORE_ATTR_NAME:
    case TS_ATTR_NAME:
    case TS_AFTER_ATTR_NAME:
    case TS_BEFORE_ATTR_VALUE:
    case TS_ATTR_VALUE_DQ:
    case TS_ATTR_VALUE_SQ:
    case TS_ATTR_VALUE_UQ:
    case TS_AFTER_ATTR_VALUE:
    case TS_SELF_CLOSING:
        for (size_t k = len; k > 0; k--)
            if (buf[k - 1] == '<')
                return len - (k - 1);
        return 0;
    case TS_COMMENT:
    case TS_DOCTYPE:
    case TS_BOGUS_COMMENT:
    case TS_MARKUP_DECL_OPEN:
        for (size_t k = len; k > 0; k--)
            if (buf[k - 1] == '<')
                return len - (k - 1);
        return 0;
    case TS_RAWTEXT:
        for (size_t k = len; k + 1 > 1; k--)
        {
            if (k >= 2 && buf[k - 2] == '<' && buf[k - 1] == '/')
                return len - (k - 2);
        }
        return 0;
    default:
        return 0;
    }
}

int mini_html5_feed(struct MiniHtml5Parser *p, const char *data, size_t len)
{
    if (!p || (!data && len))
        return -1;
    if (!p->src_input && data)
        p->src_input = data;
    if (!len)
        return 0;
    size_t total = p->carry_n + len;
    char *work = malloc(total + 1);
    if (!work)
        return -1;
    if (p->carry_n)
        memcpy(work, p->carry, p->carry_n);
    if (len)
        memcpy(work + p->carry_n, data, len);
    work[total] = 0;
    tokenize_run(p, work, total);
    size_t co = carryover_tail(p, work, total);
    free(p->carry);
    p->carry = NULL;
    p->carry_n = 0;
    if (co > 0 && co <= total)
    {
        p->carry = malloc(co + 1);
        if (p->carry)
        {
            memcpy(p->carry, work + (total - co), co);
            p->carry[co] = 0;
            p->carry_n = co;
        }
    }
    free(work);
    return 0;
}

int mini_html5_finish(struct MiniHtml5Parser *p)
{
    if (!p)
        return -1;
    if (p->carry_n)
    {
        char *work = malloc(p->carry_n + 1);
        if (work)
        {
            memcpy(work, p->carry, p->carry_n);
            work[p->carry_n] = 0;
            p->carry_n = 0;
            free(p->carry);
            p->carry = NULL;
            tokenize_run(p, work, strlen(work));
            free(work);
        }
    }
    if (p->open_n > 2)
    {
        for (int k = p->open_n - 1; k >= 0; k--)
        {
            struct MiniNode *n = p->open[k];
            if (n && n->tag && !h5_ci(n->tag, "html") && !h5_ci(n->tag, "body") && !h5_ci(n->tag, "head"))
            {
                h5_report_diag(p, MINI_PARSE_SEV_WARN, "unclosed-element",
                               "Unclosed HTML element <%s> at end of input", n->tag);
            }
        }
    }
    if (p->doc && !p->doc->body && p->doc->root)
    {
        for (struct MiniNode *h = p->doc->root->first_child; h; h = h->next_sibling)
        {
            if (h->tag && h5_ci(h->tag, "html"))
            {
                for (struct MiniNode *b = h->first_child; b; b = b->next_sibling)
                {
                    if (b->tag && h5_ci(b->tag, "body"))
                    {
                        p->doc->body = b;
                        break;
                    }
                }
            }
        }
    }
    if (p->pending_text_tail && p->pending_text_tail->text && p->pending_text_cap > p->pending_text_len + 1)
    {
        char *nb = realloc(p->pending_text_tail->text, p->pending_text_len + 1);
        if (nb)
            p->pending_text_tail->text = nb;
    }
    p->eof = 1;
    return 0;
}

void mini_html5_destroy(struct MiniHtml5Parser *p)
{
    if (!p)
        return;
    free(p->open);
    free(p->afe);
    tok_destroy(&p->cur);
    free(p->acc);
    free(p->carry);
    free(p);
}

void mini_html5_parse_fragment(struct MiniDocument *doc, struct MiniNode *context_parent, const char *html)
{
    if (!doc)
        return;
    MiniHtml5Parser *p = calloc(1, sizeof(*p));
    if (!p)
        return;
    p->doc = doc;
    p->line = 1;
    p->col = 1;
    p->fragment = 1;
    p->src_input = html;
    p->ts = TS_DATA;

    struct MiniNode *ctx = context_parent ? context_parent : (doc->body ? doc->body : doc->root);
    if (ctx && ctx->tag)
    {
        if (h5_ci(ctx->tag, "table"))
        {
            p->mode = M_IN_TABLE;
            p->table_foster = ctx;
        }
        else if (h5_ci(ctx->tag, "tbody") || h5_ci(ctx->tag, "thead") || h5_ci(ctx->tag, "tfoot"))
            p->mode = M_IN_TABLE_BODY;
        else if (h5_ci(ctx->tag, "tr"))
            p->mode = M_IN_ROW;
        else if (h5_ci(ctx->tag, "td") || h5_ci(ctx->tag, "th"))
            p->mode = M_IN_CELL;
        else if (h5_ci(ctx->tag, "select"))
            p->mode = M_IN_SELECT;
        else
            p->mode = M_IN_BODY;

        open_push(p, ctx);
        if (h5_is_rawtext(ctx->tag) || h5_is_rcdata(ctx->tag))
            maybe_enter_rawtext(p);
    }
    else
    {
        p->mode = M_IN_BODY;
        if (ctx)
            open_push(p, ctx);
    }

    if (html && *html)
        mini_html5_feed(p, html, strlen(html));
    mini_html5_finish(p);
    mini_html5_destroy(p);
}

void mini_html5_set_diagnostic_handler(MiniHtml5Parser *p, MiniHtml5DiagCallback cb, void *ud)
{
    if (!p)
        return;
    p->diag_cb = cb;
    p->diag_ud = ud;
}

void mini_html5_parse(struct MiniDocument *doc, const char *html)
{
    if (!doc)
        return;
    MiniHtml5Parser *p = mini_html5_init(doc);
    if (!p)
        return;
    p->src_input = html;
    if (html)
        mini_html5_feed(p, html, strlen(html));
    mini_html5_finish(p);
    mini_html5_destroy(p);
}