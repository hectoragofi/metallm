// Streaming markdown + LaTeX formatter for model output (feeds the console).
#pragma once
#include "common.h"

void md_reset(int base_color);  // start of a reply or a thinking section (flushes the previous one)
void md_putb(u8 b);             // next byte of model output (UTF-8)
void md_finish(void);           // end of the reply: flush anything pending, end on a fresh line
