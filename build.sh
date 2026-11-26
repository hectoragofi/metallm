#!/bin/sh
# Builds:
#   build/qwen.dll       - engine only, for testing against PyTorch from Python (ctypes)
#   build/BOOTX64.EFI    - the bare-metal UEFI chat app
set -e
cd "$(dirname "$0")"
LLVM="${LLVM:-/c/Program Files/LLVM/bin}"   # override: LLVM=/path/to/llvm/bin sh build.sh
CC="$LLVM/clang.exe"
LINK="$LLVM/lld-link.exe"
mkdir -p build/obj

CFLAGS="--target=x86_64-unknown-windows -O2 -ffreestanding -nostdlibinc -fno-stack-protector \
 -fno-stack-check -mno-stack-arg-probe -fshort-wchar -mno-red-zone -fno-math-errno \
 -fno-asynchronous-unwind-tables -Wall -Wno-unused-function -Isrc"

cc() { "$CC" $CFLAGS "$@"; }

cc -c src/libc.c         -o build/obj/libc.o
cc -c src/engine.c       -o build/obj/engine.o
cc -mavx2 -mfma -mf16c -c src/kernels_avx2.c -o build/obj/kernels_avx2.o
cc -mavx2 -mfma -mf16c -mavxvnni -DAVXVNNI -c src/kernels_avx2.c -o build/obj/kernels_avxvnni.o
cc -mavx2 -mfma -mf16c -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni \
  -c src/kernels_avx512.c -o build/obj/kernels_avx512.o

ENGINE_OBJS="build/obj/libc.o build/obj/engine.o build/obj/kernels_avx2.o build/obj/kernels_avxvnni.o build/obj/kernels_avx512.o"

# qwen.dll also carries the test-only kernel bench (tools/bench.c)
cc -c tools/bench.c -o build/obj/bench.o
"$LINK" -nologo -dll -noentry -nodefaultlib -out:build/qwen.dll $ENGINE_OBJS build/obj/bench.o \
  -export:eng_mem_required -export:eng_init -export:eng_forward -export:eng_sample \
  -export:eng_encode -export:eng_encode_special -export:eng_token_bytes -export:eng_use_avx2 -export:eng_use_avx512 \
  -export:eng_set_isa -export:eng_detect_isa -export:eng_chat_turn -export:eng_forward_batch \
  -export:eng_spec_step -export:eng_record -export:eng_rollback -export:eng_history \
  -export:g_parfor -export:bench_mem -export:bench_matmat

# test-only Windows thread pool for g_parfor (benchmarks from Python)
KLIB="${KLIB:-/c/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/um/x64/kernel32.lib}"
if [ -f "$KLIB" ]; then
  cc -c tools/parfor_win.c -o build/obj/parfor_win.o
  "$LINK" -nologo -dll -nodefaultlib -entry:_DllMainCRTStartup -out:build/parfor_win.dll build/obj/parfor_win.o "$KLIB"
fi

if [ -f src/efi_main.c ] && [ -f src/font.h ]; then
  cc -c src/console.c  -o build/obj/console.o
  cc -c src/cpu.c      -o build/obj/cpu.o
  cc -c src/fs.c       -o build/obj/fs.o
  cc -c src/md.c       -o build/obj/md.o
  cc -c src/settings.c -o build/obj/settings.o
  cc -c src/sbat.c     -o build/obj/sbat.o
  cc -c src/efi_main.c -o build/obj/efi_main.o
  cc -DFORCE_SIPI -c src/efi_main.c -o build/obj/efi_main_sipi.o
  cc -c src/ap_tramp.S -o build/obj/ap_tramp.o
  EFI_OBJS="$ENGINE_OBJS build/obj/console.o build/obj/cpu.o build/obj/fs.o build/obj/md.o build/obj/settings.o build/obj/sbat.o build/obj/ap_tramp.o"
  "$LINK" -nologo -subsystem:efi_application -entry:efi_main -nodefaultlib \
    -out:build/BOOTX64.EFI $EFI_OBJS build/obj/efi_main.o
  # test variant: skips firmware MP services and always wakes cores via APIC
  "$LINK" -nologo -subsystem:efi_application -entry:efi_main -nodefaultlib \
    -out:build/BOOTX64_SIPI.EFI $EFI_OBJS build/obj/efi_main_sipi.o
  ls -l build/BOOTX64.EFI
fi
ls -l build/qwen.dll
