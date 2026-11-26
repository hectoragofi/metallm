// Qwen3 / Qwen3.5 inference engine: freestanding, platform-agnostic.
//
// ---------------------------------------------------------------------------------------------
// API (model file format v2). Changes since v1:
//  * .BIN format v2 (export.py). v1 files are rejected ("unsupported model file version").
//    Header: arch field (0 = Qwen3 dense, 1 = Qwen3.5 hybrid Gated-DeltaNet), per-matrix weight
//    type (int8 / int4 / 1-bit / ternary 2-bit), name still at bytes 192..255. Big models are
//    split into <stem>.BIN, <stem>.B01, <stem>.B02 ... (raw byte split); the platform joins the
//    parts in RAM before eng_init.
//  * eng_chat_turn gained a system prompt: (sys, sys_len) is used only when first=1 (sys_len 0 =
//    none). Tokens follow the model's HF chat template exactly (also for later turns: see
//    "chat history" below).
//  * eng_set_isa(flags): kernel selection (ISA_* bits below). eng_use_avx2(on) still works and
//    equals eng_set_isa(on ? ISA_AVX2 : 0). Required CPU state:
//      ISA_AVX2      CPUID AVX2+FMA+F16C, CR4.OSXSAVE=1, XCR0 bits 0,1,2 (x87, SSE, AVX) set.
//      ISA_AVX512    CPUID.7.EBX AVX512F(16) DQ(17) BW(30) VL(31), XCR0 bits 5,6,7 (opmask,
//                    ZMM_Hi256, Hi16_ZMM) set in addition -> XCR0 |= 0xE7.
//      ISA_VNNI512   + CPUID.7.ECX bit 11 (AVX512_VNNI). Same XCR0 as ISA_AVX512.
//      ISA_AVXVNNI   CPUID.(7,1).EAX bit 4 (AVX-VNNI, VEX-encoded); XCR0 as for AVX2.
//    eng_set_isa ignores bits whose prerequisites are missing in the flags (e.g. VNNI512
//    without AVX512) and returns the flags it actually uses.
//  * eng_forward(e, tok, pos, want_logits) unchanged. pos == 0 resets the recurrent state.
//    Rewinding (pos lower than the previous call's pos + 1, e.g. /retry) is supported: the
//    engine restores its checkpoint of the turn start, or replays the token history.
//  * eng_forward_batch(e, toks, n, pos, logits): n tokens at pos..pos+n-1 reading every weight
//    once. logits == NULL: returns the last token's logits (like eng_forward with
//    want_logits=1). Otherwise logits[n][vocab] receives every position's logits.
//  * eng_detect_isa() (CPUID + XGETBV) and eng_use_avx512(on) (= eng_set_isa(eng_detect_isa())
//    with or without the AVX-512 bits). The app enables XCR0 bits 5-7 on every core first.
//  * eng_spec_step: speculative decoding with a smaller draft model (same tokenizer).
//  * eng_mem_required / eng_init unchanged in signature. Memory now also covers the fp16 KV
//    cache (full-attention layers only), recurrent state, checkpoints and batch buffers.
//  * eng_encode_special: like eng_encode but text spelling an added token (<|im_start|>,
//    <think>, ...) becomes that token. eng_history: the physical token history (tests).
//
// Speculative decoding from a chat loop (draft = a smaller model with the same tokenizer:
// Qwen3.5-0.8B for Qwen3.5-2B/4B/9B, Qwen3-0.6B for Qwen3-1.7B):
//    load both files; Engine T, D; each gets its own arena of eng_mem_required(file, max_seq)
//    (same max_seq) and eng_init. Per user turn:
//      nt = eng_chat_turn(&T, sys, sl, msg, n, first, think, toks, max);
//           eng_chat_turn(&D, sys, sl, msg, n, first, think, toks, max);   // same tokens
//      lg = eng_forward_batch(&T, toks, nt, pos, 0);   // or eng_forward per token
//           eng_forward_batch(&D, toks, nt, pos, 0);
//      pos += nt; cur = eng_sample(&T, lg, temp, top_p, top_k, &rng);
//      while (cur != im_end && cur != eot) {
//          print(cur);
//          int out[16], n = eng_spec_step(&T, &D, &pos, cur, 2, temp, top_p, top_k, &rng, out);
//          // out[0..n-1] are the next tokens; out[n-1] becomes the next `cur` (not fed yet).
//          // If out[j] (j < n-1) is im_end/eot: print out[0..j-1], set pos -= n-1-j, stop.
//          for (j = 0; j < n-1; j++) print(out[j]);  cur = out[n-1];
//      }
//    pos then is the position after the last fed reply token, exactly as in the plain loop;
//    the next eng_chat_turn(first=0) + feed at pos works unchanged on both engines (each
//    discards anything fed past pos). /retry: same as without spec, on both engines.
//    k = 2 drafts per step (k <= 15); temp <= 0 or top_k == 1 gives exactly the target's greedy
//    output. Measured on the dev box (8 threads, Zen4, loaded): acceptance ~0.6 at k=2, speed
//    x0.95..x1.01 of plain decoding (verifying k+1 tokens costs ~1.5-2x one token because the
//    int8-activation kernels are compute bound), so the app does not use it.
//
// Chat history: the HF templates drop the reasoning of earlier turns (Qwen3, Qwen3.5) and
// re-tokenize the reply text. To match them exactly the engine rewrites the previous reply
// when the next turn starts (eng_chat_turn with first=0, then feeding the returned tokens at
// the caller's next pos). The caller keeps counting positions as before; the engine maps the
// caller's ("logical") positions to its own by an offset that is never negative, so logical
// pos >= physical pos and the caller's context-full checks stay safe.
// ---------------------------------------------------------------------------------------------
#pragma once
#include "common.h"

enum { ARCH_QWEN3 = 0, ARCH_QWEN35 = 1 };
enum { WT_Q8 = 0, WT_Q4 = 1, WT_Q1 = 2, WT_T2 = 3 };   // weight types
enum { MF_HAD = 1 };                                    // matrix input is block-Hadamard rotated
enum { HF_ATTN_GATE = 1, HF_V_TILED = 2, HF_HADAMARD = 4, HF_GDN_V_GROUPED = 8 };
enum { ISA_AVX2 = 1, ISA_AVX512 = 2, ISA_VNNI512 = 4, ISA_AVXVNNI = 8 };
enum { CS_QWEN3 = 0, CS_QWEN35 = 1, CS_NOTHINK = 2, CS_QWEN38 = 3 };   // chat template families

typedef struct __attribute__((packed)) {
    char magic[4];                 // "QWN3"
    u32 version;                   // 2
    u32 arch;
    u32 dim, hidden, n_layers, n_heads, n_kv, head_dim, vocab, rope_len, gs, rope_dim;
    u32 lin_nk, lin_nv, lin_dk, lin_dv, conv_k, flags;
    float rope_theta, eps;
    u64 full_mask;                 // bit l set: layer l is full attention (else linear)
    u64 off_rope, off_params, off_mats, off_tok, off_had, n_mats;
    u32 n_merges, tok_im_start, tok_im_end, tok_eot, tok_think, tok_think_end;
} ModelHeader;                     // <= 192 bytes; 192..255 = display name

typedef struct __attribute__((packed)) { u32 type, rows, cols, gs, flags, pad; u64 off_w, off_s, off_m; } MatDesc;

typedef struct {
    const u8* w; const u16* s; const u16* m;   // codes, fp16 group scales, fp16 group mins (Q4 asym) or NULL
    int type, rows, cols, gs, flags;
    const float* signs;                        // Hadamard signs for this input width (MF_HAD)
} QMat;

typedef struct {
    int full;                                  // 1 = full attention, 0 = Gated DeltaNet
    const float *attn_norm, *ffn_norm;
    // full attention
    const float *q_norm, *k_norm;
    QMat wq, wk, wv, wo;                       // wq rows: [q (nh*hd)] then [gate (nh*hd)] if gated
    // Gated DeltaNet
    const float *conv_w, *dt_bias, *A, *gnorm; // conv_w [conv_dim][conv_k], A = -exp(A_log)
    QMat in_qkv, in_z, in_a, in_b, out;
    QMat w1, w3, w2;
    int ci;                                    // index among full layers (KV) or linear layers (state)
} Layer;

// activations quantized to int8 in blocks of 32
// int8 activations, blocks of 32: q, scale s[b], sum[b] = s[b] * sum(q in block b); per 4 values
// (one 32-bit dot lane) nks[j] = -(q[4j] + .. + q[4j+3]) and xsx[j] = s[j / 8], for the kernels.
typedef struct { i8* q; float* s; float* sum; int* nks; float* xsx; } QAct;

typedef struct Engine {
    const ModelHeader* h;
    int arch, dim, hidden, L, nh, nkv, hd, V, max_seq, rope_dim, gated;
    int nk, nv, dk, dv, conv_k, conv_dim, n_full, n_lin, v_tiled, gdn_v_grouped;
    float eps;
    const float* rope;
    const float* final_norm;
    QMat embed, head;
    Layer* layers;
    int had_bs;                                // Hadamard block size (0 = none)
    // runtime buffers (B = max batch rows)
    int B;
    float *x, *xb, *q, *k, *v, *att_out, *hb, *hb2, *logits, *att, *tmp, *gate;
    float *qkv, *z, *ab, *bb;
    QAct act; float* hadbuf;
    u16 *key_cache, *val_cache;                // fp16 [n_full][max_seq][nkv*hd]
    float *conv_state, *rec_state;             // [n_lin][conv_k-1][conv_dim], [n_lin][nv][dk][dv]
    usize state_floats;
    // history / checkpoints / chat
    int* hist; int n_past, pos_off, last_logical;
    float* ckpt[3]; int ckpt_pos[3];           // 0 = turn start, 1 = assistant start, 2 = spare
    float* rec_log; int rec_n, rec_on;         // speculative rollback log
    float* conv_log;
    int chat_style, pend, pend_first, pend_asst, pend_content, asst_pos, content_pos;
    const u8* think_sys; int think_sys_len;
    u8* textbuf; int textcap; int* tokbuf;
    // tokenizer
    const u32 *vocab_offs, *merges, *bytetok, *uranges;
    const u8 *vocab_bytes, *uclass;
    int n_uranges, letter_m; u8 asc[128];
    const u32 *ccc, *decomp, *decomp_data, *comp;
    int n_ccc, n_decomp, n_comp;
    const u32* added; int n_added;             // [id, off, len] into vocab bytes (added tokens)
    u64* mkeys; u32* mvals; u32 mmask;
    int* bpe_scratch; int* bpe_rank; u32* cps; u8* nfcbuf;
} Engine;

// Memory needed for the runtime arena for a given context length.
usize eng_mem_required(const void* file, int max_seq);
// Returns 0 on success, or a static error string.
const char* eng_init(Engine* e, const void* file, u64 size, int max_seq, void* arena, usize arena_size);
// Run one token at position pos. Returns logits (or NULL if want_logits==0).
float* eng_forward(Engine* e, int token, int pos, int want_logits);
// Run n tokens at pos..pos+n-1. logits==NULL: return last logits; else fill logits[n][V], return it.
float* eng_forward_batch(Engine* e, const int* tokens, int n, int pos, float* logits);
// Sampling. temp<=0 -> greedy.
int eng_sample(Engine* e, const float* logits, float temp, float top_p, int top_k, u64* rng);
// BPE encode plain text (NFC + pre-tokenizer + BPE; no special tokens). Returns tokens written.
int eng_encode(Engine* e, const u8* text, int len, int* out, int max_out);
// Like eng_encode but added/special tokens written as text (e.g. "<think>") become their ids.
int eng_encode_special(Engine* e, const u8* text, int len, int* out, int max_out);
const u8* eng_token_bytes(Engine* e, int id, int* len);
// Tokens for one user turn (see "chat history" above). sys: optional system prompt (first=1 only).
int eng_chat_turn(Engine* e, const u8* sys, int sys_len, const u8* msg, int len, int first, int think, int* out, int max);
// Kernel selection: ISA_* flags (0 = portable C). Returns the flags in use.
int eng_set_isa(int flags);
// What this CPU + current XCR0 allow (CPUID + XGETBV; 0 if CR4.OSXSAVE is off). Pass to eng_set_isa.
int eng_detect_isa(void);
void eng_use_avx2(int on);
// on: every kernel this CPU + XCR0 allow (eng_detect_isa), i.e. AVX-512 VNNI on Zen 4; off: the
// same without AVX-512. Call after XCR0 |= 0xE7 on all cores that run engine work.
void eng_use_avx512(int on);
// Speculative decoding. *pos = tokens already fed to both engines, cur = last sampled token (not
// fed yet). Drafts up to k tokens (k <= 15) with `draft`, verifies with one batched pass of
// `target`, writes the produced tokens (accepted drafts + one token from the target) to out and
// returns their count; *pos advances by that count, and the last out token is the next `cur`.
// temp <= 0: exact greedy (identical to target-only greedy). temp > 0: speculative sampling
// (Leviathan et al.), output distribution equals the target's top_k/top_p/temp distribution.
int eng_spec_step(Engine* target, Engine* draft, int* pos, int cur, int k, float temp, float top_p, int top_k, u64* rng, int* out);
// Speculative-decoding rollback primitives (used by eng_spec_step, exposed for tests/tools):
void eng_record(Engine* e, int on);   // on: snapshot state now and log every token's state update
void eng_rollback(Engine* e, int pos);// back to pos (>= record start), using the log; stops recording
// Physical token history (after chat-history rewrites); returns its length. For tests.
int eng_history(Engine* e, const int** toks);

// kernels
typedef struct {
    void (*matmat)(float* out, int ldo, const QMat* m, const QAct* a, int n, int lda, int r0, int r1);
    float (*dot)(const float* a, const float* b, int n);
    void (*axpy)(float* y, float a, const float* x, int n);
    float (*dot_h)(const float* a, const u16* b, int n);       // fp32 . fp16
    void (*axpy_h)(float* y, float a, const u16* x, int n);
    void (*to_h)(u16* d, const float* s, int n);
} Kernels;
extern Kernels g_k;
void kernels_avx2(Kernels* k);
void kernels_avx512(Kernels* k, int vnni);
void kernels_avxvnni(Kernels* k);
float h2f(u16 h);
u16 f2h(float f);
