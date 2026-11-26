// User settings, stored as key=value lines in \SETTINGS.TXT.
#pragma once
#include "common.h"

typedef struct {
    int temp100;    // temperature x100, -1 = Qwen's recommended (0.7, or 0.6 when thinking)
    int topp100;    // top-p x100, -1 = recommended (0.8, or 0.95 when thinking)
    int topk;       // 1..100
    int think;      // step-by-step reasoning by default
    int max_reply;  // tokens per reply, 0 = until the context is full
    int memory;     // pass \MEMORY.TXT as the system prompt
    int autosave;   // write \CHATS\CHATnnnn.TXT after every reply
    int ctx;        // context tokens, 0 = largest that fits
    int threads;    // 0 = auto
    int boost;      // raise CPU clocks
    int font;       // -1 = auto, 0 small, 1 medium, 2 large
    int theme;      // THEME_*
    char model[32]; // last model file, e.g. QWEN3-06.BIN
} Settings;

void settings_default(Settings* s);
void settings_parse(Settings* s, const u8* text, u64 len);  // unknown keys and bad values are ignored
int settings_format(const Settings* s, char* out, int max); // returns length
