// Minimal UEFI definitions (only what this app uses). Layouts follow the UEFI 2.x spec.
#pragma once
#include "common.h"

#define EFIAPI __attribute__((ms_abi))
typedef u64 EFI_STATUS;
typedef void* EFI_HANDLE;
typedef void* EFI_EVENT;
typedef u16 CHAR16;
typedef u64 UINTN;
typedef u8 BOOLEAN;
typedef u64 EFI_PHYSICAL_ADDRESS;

#define EFI_SUCCESS 0
#define EFI_ERR(x) (0x8000000000000000ull | (x))
#define EFI_NOT_READY EFI_ERR(6)
#define EFI_ERROR(s) ((i64)(s) < 0)

typedef struct { u32 d1; u16 d2, d3; u8 d4[8]; } EFI_GUID;

typedef struct { u64 Signature; u32 Revision, HeaderSize, CRC32, Reserved; } EFI_TABLE_HEADER;

// ---- text input / output ----
typedef struct { u16 ScanCode; CHAR16 UnicodeChar; } EFI_INPUT_KEY;
typedef struct EFI_SIMPLE_TEXT_INPUT {
    EFI_STATUS (EFIAPI *Reset)(struct EFI_SIMPLE_TEXT_INPUT*, BOOLEAN);
    EFI_STATUS (EFIAPI *ReadKeyStroke)(struct EFI_SIMPLE_TEXT_INPUT*, EFI_INPUT_KEY*);
    EFI_EVENT WaitForKey;
} EFI_SIMPLE_TEXT_INPUT;

typedef struct EFI_SIMPLE_TEXT_OUTPUT {
    EFI_STATUS (EFIAPI *Reset)(struct EFI_SIMPLE_TEXT_OUTPUT*, BOOLEAN);
    EFI_STATUS (EFIAPI *OutputString)(struct EFI_SIMPLE_TEXT_OUTPUT*, const CHAR16*);
    void *TestString, *QueryMode, *SetMode, *SetAttribute;
    EFI_STATUS (EFIAPI *ClearScreen)(struct EFI_SIMPLE_TEXT_OUTPUT*);
    void *SetCursorPosition, *EnableCursor, *Mode;
} EFI_SIMPLE_TEXT_OUTPUT;

// ---- boot services ----
typedef enum { AllocateAnyPages, AllocateMaxAddress, AllocateAddress } EFI_ALLOCATE_TYPE;
#define EfiLoaderData 2
#define ByProtocol 2

typedef struct {
    EFI_TABLE_HEADER Hdr;
    void *RaiseTPL, *RestoreTPL;
    EFI_STATUS (EFIAPI *AllocatePages)(EFI_ALLOCATE_TYPE, u32 MemType, UINTN Pages, EFI_PHYSICAL_ADDRESS*);
    EFI_STATUS (EFIAPI *FreePages)(EFI_PHYSICAL_ADDRESS, UINTN);
    EFI_STATUS (EFIAPI *GetMemoryMap)(UINTN* MapSize, void* Map, UINTN* MapKey, UINTN* DescSize, u32* DescVersion);
    EFI_STATUS (EFIAPI *AllocatePool)(u32 PoolType, UINTN Size, void** Buffer);
    EFI_STATUS (EFIAPI *FreePool)(void*);
    EFI_STATUS (EFIAPI *CreateEvent)(u32 Type, UINTN NotifyTpl, void* Fn, void* Ctx, EFI_EVENT* Event);
    void *SetTimer;
    EFI_STATUS (EFIAPI *WaitForEvent)(UINTN N, EFI_EVENT* Events, UINTN* Index);
    void *SignalEvent, *CloseEvent, *CheckEvent;
    void *InstallProtocolInterface, *ReinstallProtocolInterface, *UninstallProtocolInterface;
    EFI_STATUS (EFIAPI *HandleProtocol)(EFI_HANDLE, const EFI_GUID*, void**);
    void *Reserved, *RegisterProtocolNotify, *LocateHandle, *LocateDevicePath, *InstallConfigurationTable;
    void *LoadImage, *StartImage, *Exit, *UnloadImage, *ExitBootServices;
    void *GetNextMonotonicCount;
    EFI_STATUS (EFIAPI *Stall)(UINTN Microseconds);
    EFI_STATUS (EFIAPI *SetWatchdogTimer)(UINTN Timeout, u64 Code, UINTN DataSize, CHAR16* Data);
    void *ConnectController, *DisconnectController;
    void *OpenProtocol, *CloseProtocol, *OpenProtocolInformation;
    void *ProtocolsPerHandle;
    EFI_STATUS (EFIAPI *LocateHandleBuffer)(u32 SearchType, const EFI_GUID*, void* Key, UINTN* NoHandles, EFI_HANDLE** Buffer);
    EFI_STATUS (EFIAPI *LocateProtocol)(const EFI_GUID*, void* Registration, void** Interface);
    void *InstallMultipleProtocolInterfaces, *UninstallMultipleProtocolInterfaces;
    void *CalculateCrc32, *CopyMem, *SetMem, *CreateEventEx;
} EFI_BOOT_SERVICES;

typedef struct {
    EFI_TABLE_HEADER Hdr;
    CHAR16* FirmwareVendor;
    u32 FirmwareRevision;
    EFI_HANDLE ConsoleInHandle;
    EFI_SIMPLE_TEXT_INPUT* ConIn;
    EFI_HANDLE ConsoleOutHandle;
    EFI_SIMPLE_TEXT_OUTPUT* ConOut;
    EFI_HANDLE StandardErrorHandle;
    EFI_SIMPLE_TEXT_OUTPUT* StdErr;
    struct EFI_RUNTIME_SERVICES* RuntimeServices;
    EFI_BOOT_SERVICES* BootServices;
    UINTN NumberOfTableEntries;
    void* ConfigurationTable;
} EFI_SYSTEM_TABLE;

// ---- graphics output ----
typedef struct { u32 R, G, B, Reserved; } EFI_PIXEL_BITMASK;
typedef struct {
    u32 Version, HorizontalResolution, VerticalResolution;
    u32 PixelFormat;  // 0 RGBx, 1 BGRx, 2 bitmask, 3 blt-only
    EFI_PIXEL_BITMASK PixelInformation;
    u32 PixelsPerScanLine;
} EFI_GOP_MODE_INFO;
typedef struct {
    u32 MaxMode, Mode;
    EFI_GOP_MODE_INFO* Info;
    UINTN SizeOfInfo;
    EFI_PHYSICAL_ADDRESS FrameBufferBase;
    UINTN FrameBufferSize;
} EFI_GOP_MODE;
#define EfiBltBufferToVideo 2
typedef struct EFI_GOP {
    EFI_STATUS (EFIAPI *QueryMode)(struct EFI_GOP*, u32, UINTN*, EFI_GOP_MODE_INFO**);
    EFI_STATUS (EFIAPI *SetMode)(struct EFI_GOP*, u32);
    EFI_STATUS (EFIAPI *Blt)(struct EFI_GOP*, void* Buf, u32 Op, UINTN SX, UINTN SY, UINTN DX, UINTN DY, UINTN W, UINTN H, UINTN Delta);
    EFI_GOP_MODE* Mode;
} EFI_GOP;

// ---- files ----
typedef struct {
    u32 Revision; EFI_HANDLE ParentHandle; EFI_SYSTEM_TABLE* SystemTable;
    EFI_HANDLE DeviceHandle; void* FilePath; void* Reserved;
    u32 LoadOptionsSize; void* LoadOptions; void* ImageBase; u64 ImageSize;
    u32 ImageCodeType, ImageDataType; void* Unload;
} EFI_LOADED_IMAGE;

typedef struct EFI_FILE {
    u64 Revision;
    EFI_STATUS (EFIAPI *Open)(struct EFI_FILE*, struct EFI_FILE**, const CHAR16*, u64 Mode, u64 Attr);
    EFI_STATUS (EFIAPI *Close)(struct EFI_FILE*);
    EFI_STATUS (EFIAPI *Delete)(struct EFI_FILE*);
    EFI_STATUS (EFIAPI *Read)(struct EFI_FILE*, UINTN*, void*);
    EFI_STATUS (EFIAPI *Write)(struct EFI_FILE*, UINTN*, const void*);
    EFI_STATUS (EFIAPI *GetPosition)(struct EFI_FILE*, u64*);
    EFI_STATUS (EFIAPI *SetPosition)(struct EFI_FILE*, u64);
    EFI_STATUS (EFIAPI *GetInfo)(struct EFI_FILE*, const EFI_GUID*, UINTN*, void*);
    EFI_STATUS (EFIAPI *SetInfo)(struct EFI_FILE*, const EFI_GUID*, UINTN, void*);
    EFI_STATUS (EFIAPI *Flush)(struct EFI_FILE*);
} EFI_FILE;
typedef struct EFI_SFS {
    u64 Revision;
    EFI_STATUS (EFIAPI *OpenVolume)(struct EFI_SFS*, EFI_FILE**);
} EFI_SFS;
#define EFI_FILE_MODE_READ 1
#define EFI_FILE_MODE_WRITE 2
#define EFI_FILE_MODE_CREATE 0x8000000000000000ull
// EFI_FILE_INFO: u64 Size, FileSize, PhysicalSize; EFI_TIME x3; u64 Attribute; CHAR16 FileName[]
#define FILE_INFO_FILESIZE 8
#define FILE_INFO_ATTR 72
#define FILE_INFO_NAME 80
#define EFI_FILE_DIRECTORY 0x10

// ---- memory map ----
typedef struct { u32 Type; u32 Pad; u64 PhysicalStart, VirtualStart, NumberOfPages, Attribute; } EFI_MEMORY_DESCRIPTOR;
#define EfiConventionalMemory 7
#define EFI_BUFFER_TOO_SMALL EFI_ERR(5)

// ---- runtime services (GetTime, ResetSystem) ----
typedef struct { u16 Year; u8 Month, Day, Hour, Minute, Second, Pad1; u32 Nanosecond; short TimeZone; u8 Daylight, Pad2; } EFI_TIME;
typedef struct EFI_RUNTIME_SERVICES {
    EFI_TABLE_HEADER Hdr;
    EFI_STATUS (EFIAPI *GetTime)(EFI_TIME*, void* Capabilities);
    void *SetTime, *GetWakeupTime, *SetWakeupTime, *SetVirtualAddressMap, *ConvertPointer;
    void *GetVariable, *GetNextVariableName, *SetVariable, *GetNextHighMonotonicCount;
    void (EFIAPI *ResetSystem)(u32 Type, EFI_STATUS Status, UINTN DataSize, void* Data);
} EFI_RUNTIME_SERVICES;
#define EfiResetCold 0
#define EfiResetShutdown 2

// ---- pointer (mouse wheel = RelativeMovementZ) ----
typedef struct { i32 RelativeMovementX, RelativeMovementY, RelativeMovementZ; BOOLEAN LeftButton, RightButton; } EFI_SIMPLE_POINTER_STATE;
typedef struct { u64 ResolutionX, ResolutionY, ResolutionZ; BOOLEAN LeftButton, RightButton; } EFI_SIMPLE_POINTER_MODE;
typedef struct EFI_SIMPLE_POINTER {
    EFI_STATUS (EFIAPI *Reset)(struct EFI_SIMPLE_POINTER*, BOOLEAN);
    EFI_STATUS (EFIAPI *GetState)(struct EFI_SIMPLE_POINTER*, EFI_SIMPLE_POINTER_STATE*);
    EFI_EVENT WaitForInput;
    EFI_SIMPLE_POINTER_MODE* Mode;
} EFI_SIMPLE_POINTER;

// ---- multiprocessor services ----
typedef struct { u64 ProcessorId; u32 StatusFlag; u32 Package, Core, Thread; u8 ext[64]; } EFI_PROCESSOR_INFORMATION;
typedef void (EFIAPI *EFI_AP_PROCEDURE)(void*);
typedef struct EFI_MP {
    EFI_STATUS (EFIAPI *GetNumberOfProcessors)(struct EFI_MP*, UINTN* Total, UINTN* Enabled);
    EFI_STATUS (EFIAPI *GetProcessorInfo)(struct EFI_MP*, UINTN, EFI_PROCESSOR_INFORMATION*);
    void* StartupAllAPs;
    EFI_STATUS (EFIAPI *StartupThisAP)(struct EFI_MP*, EFI_AP_PROCEDURE, UINTN Cpu, EFI_EVENT WaitEvent, UINTN TimeoutUs, void* Arg, BOOLEAN* Finished);
    void *SwitchBSP, *EnableDisableAP;
    EFI_STATUS (EFIAPI *WhoAmI)(struct EFI_MP*, UINTN*);
} EFI_MP;

static const EFI_GUID GOP_GUID = {0x9042a9de, 0x23dc, 0x4a38, {0x96, 0xfb, 0x7a, 0xde, 0xd0, 0x80, 0x51, 0x6a}};
static const EFI_GUID LOADED_IMAGE_GUID = {0x5b1b31a1, 0x9562, 0x11d2, {0x8e, 0x3f, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b}};
static const EFI_GUID SFS_GUID = {0x964e5b22, 0x6459, 0x11d2, {0x8e, 0x39, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b}};
static const EFI_GUID POINTER_GUID = {0x31878c87, 0x0b75, 0x11d5, {0x9a, 0x4f, 0x00, 0x90, 0x27, 0x3f, 0xc1, 0x4d}};
static const EFI_GUID FILE_INFO_GUID = {0x09576e92, 0x6d3f, 0x11d2, {0x8e, 0x39, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b}};
static const EFI_GUID MP_GUID = {0x3fdda605, 0xa76e, 0x4f46, {0xad, 0x29, 0x12, 0xf4, 0x53, 0x1b, 0x3d, 0x08}};
