// AP startup trampoline: copied to a page below 1 MiB, entered by SIPI in 16-bit real mode.
// Goes real mode -> 32-bit protected mode -> long mode, picks a stack, calls the C entry.
// The C side patches the absolute fields (tramp_gdtr_base, tramp_far32/64, tramp_*) first.
    .section .rdata,"dr"
    .globl ap_tramp_start, ap_tramp_end, tramp_gdt, tramp_pm32, tramp_lm64
    .globl tramp_gdtr_base, tramp_far32, tramp_far64, tramp_cr3, tramp_count
    .globl tramp_stack_base, tramp_stack_size, tramp_entry, tramp_max
    .p2align 12
    .code16
ap_tramp_start:
    cli
    cld
    mov     %cs, %ax
    mov     %ax, %ds
    xor     %esi, %esi
    mov     %cs, %si
    shl     $4, %esi                    // esi = physical base of this page
    lgdtl   (tramp_gdtr - ap_tramp_start)
    mov     %cr0, %eax
    and     $0x9FFFFFFB, %eax           // clear CD, NW (caches on) and EM
    or      $0x23, %eax                 // PE | MP | NE
    mov     %eax, %cr0
    ljmpl   *(tramp_far32 - ap_tramp_start)

    .code32
tramp_pm32:
    mov     $0x10, %ax
    mov     %ax, %ds
    mov     %ax, %es
    mov     %ax, %ss
    mov     %ax, %fs
    mov     %ax, %gs
    mov     %cr4, %eax
    or      $0x620, %eax                // PAE | OSFXSR | OSXMMEXCPT
    mov     %eax, %cr4
    mov     (tramp_cr3 - ap_tramp_start)(%esi), %eax
    mov     %eax, %cr3
    mov     $0xC0000080, %ecx           // EFER
    rdmsr
    or      $0x100, %eax                // LME
    wrmsr
    mov     %cr0, %eax
    or      $0x80000000, %eax           // PG
    mov     %eax, %cr0
    ljmp    *(tramp_far64 - ap_tramp_start)(%esi)

    .code64
tramp_lm64:
    mov     $0x10, %ax
    mov     %ax, %ds
    mov     %ax, %es
    mov     %ax, %ss
    mov     %esi, %esi
    mov     $1, %eax
    lock xaddl %eax, (tramp_count - ap_tramp_start)(%rsi)
    cmp     (tramp_max - ap_tramp_start)(%rsi), %rax
    jae     1f                          // more cores than stacks: park this one
    mov     %eax, %ecx                  // arg0 = AP index (0-based)
    lea     1(%rax), %rdx
    imul    (tramp_stack_size - ap_tramp_start)(%rsi), %rdx
    add     (tramp_stack_base - ap_tramp_start)(%rsi), %rdx
    mov     %rdx, %rsp
    and     $-16, %rsp
    sub     $32, %rsp                   // MS ABI shadow space
    mov     (tramp_entry - ap_tramp_start)(%rsi), %rax
    call    *%rax
1:  cli
    hlt
    jmp     1b

    .p2align 3
tramp_gdt:
    .quad 0
    .quad 0x00CF9A000000FFFF            // 0x08: 32-bit code
    .quad 0x00CF92000000FFFF            // 0x10: data
    .quad 0x00AF9A000000FFFF            // 0x18: 64-bit code
tramp_gdtr:
    .word 31
tramp_gdtr_base:
    .long 0
tramp_far32:
    .long 0
    .word 0x08
tramp_far64:
    .long 0
    .word 0x18
    .p2align 3
tramp_cr3:        .quad 0
tramp_count:      .quad 0
tramp_stack_base: .quad 0
tramp_stack_size: .quad 0
tramp_entry:      .quad 0
tramp_max:        .quad 0               // APs with index >= this park without using a stack
ap_tramp_end:
