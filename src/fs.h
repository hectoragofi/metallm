// Files on the boot stick (FAT32 through the firmware's file protocol). Paths are ASCII with
// backslashes, e.g. "\\CHATS\\CHAT0001.TXT". Every call returns 0 on success, else an error
// (the stick may be read-only); callers report it and carry on.
#pragma once
#include "uefi.h"

void fs_init(EFI_BOOT_SERVICES* bs, EFI_FILE* root);
int fs_ready(void);
// Reads a whole file into a pool buffer (NUL-terminated; free with fs_free). -1 = missing.
int fs_read(const char* path, u8** data, u64* len);
void fs_free(void* p);
int fs_write(const char* path, const void* data, u64 len);   // create or replace
int fs_append(const char* path, const void* data, u64 len);  // create or extend
int fs_delete(const char* path);
int fs_mkdir(const char* path);
// Next free name like \CHATS\CHAT0001.TXT: dir "\\CHATS", prefix "CHAT", ext ".TXT".
int fs_next_name(const char* dir, const char* prefix, const char* ext, char* out, int max);
const char* fs_error(void);  // short reason for the last failure
