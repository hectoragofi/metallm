#!/usr/bin/env python3
"""Build a raw, bootable GPT + FAT32 (EFI System Partition) disk image.

Pure Python 3 stdlib.  Files are stream-copied and laid out contiguously;
zero regions are never written (the output is made sparse on NTFS, and on
other filesystems seeking over gaps yields holes naturally).

Build:
    python tools/make_disk.py OUT.img --file EFI/BOOT/BOOTX64.EFI=app.efi \
        --file QWEN.BIN=build/QWEN.BIN [--file ...] [--size-mb N] [--check]

Verify an existing image (independent reader: GPT -> FAT32 -> directory walk;
optional --file specs are compared byte-for-byte via SHA-256):
    python tools/make_disk.py --verify OUT.img [--file DEST=SRC ...]

Layout:
    LBA 0             protective MBR (one 0xEE partition)
    LBA 1             primary GPT header
    LBA 2..33         primary partition entry array (128 x 128 bytes)
    LBA 2048..        EFI System Partition, FAT32 (starts at 1 MiB, 1 MiB-aligned size)
    LBA N-33..N-2     backup partition entry array
    LBA N-1           backup GPT header
"""

import argparse
import datetime
import hashlib
import os
import struct
import sys
import uuid
import zlib

SECTOR = 512
MIB = 1024 * 1024
ESP_TYPE_GUID = uuid.UUID("C12A7328-F81F-11D2-BA4B-00A0C93EC93B")
PART_START_LBA = 2048                     # 1 MiB
GPT_ENTRIES = 128
GPT_ENTRY_SIZE = 128
GPT_ARRAY_SECTORS = GPT_ENTRIES * GPT_ENTRY_SIZE // SECTOR   # 32
FAT32_MIN_CLUSTERS = 65525
FAT32_SAFE_MIN_CLUSTERS = 65525 + 16      # stay clear of the FAT16/FAT32 boundary
FAT32_MAX_CLUSTERS = 0x0FFFFFF5 - 2
EOC = 0x0FFFFFFF
RESERVED_SECTORS_MIN = 32
BACKUP_BOOT_SECTOR = 6
FSINFO_SECTOR = 1
COPY_CHUNK = 8 * MIB
DEFAULT_SLACK_MIB = 64
VOLUME_LABEL = b"QWENBOOT   "
ATTR_READ_ONLY, ATTR_HIDDEN, ATTR_SYSTEM, ATTR_VOLUME_ID, ATTR_DIRECTORY, ATTR_ARCHIVE = 1, 2, 4, 8, 0x10, 0x20
ATTR_LFN = 0x0F
SHORT_NAME_CHARS = set(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&'()-@^_`{}~")


def die(msg):
    print("error: " + msg, file=sys.stderr)
    sys.exit(1)


def ceil_div(a, b):
    return -(-a // b)


# --------------------------------------------------------------------------
# Names / tree
# --------------------------------------------------------------------------

def short_name(component):
    """'BOOTX64.EFI' -> b'BOOTX64 EFI' (11 bytes). Validates 8.3."""
    comp = component.upper()
    if comp in (".", "..") or not comp:
        raise ValueError("invalid path component %r" % component)
    if comp.count(".") > 1:
        raise ValueError("not an 8.3 name: %r" % component)
    base, _, ext = comp.partition(".")
    if not (1 <= len(base) <= 8) or len(ext) > 3 or (_ and not ext):
        raise ValueError("not an 8.3 name: %r" % component)
    raw = (base.encode("ascii", "replace"), ext.encode("ascii", "replace"))
    for part in raw:
        for ch in part:
            if ch not in SHORT_NAME_CHARS:
                raise ValueError("invalid character in 8.3 name %r" % component)
    name = raw[0].ljust(8, b" ") + raw[1].ljust(3, b" ")
    if name[0] == 0xE5:
        name = b"\x05" + name[1:]
    return name


class Node:
    def __init__(self, name11, is_dir, src=None, size=0, parent=None):
        self.name11 = name11
        self.is_dir = is_dir
        self.src = src
        self.size = size
        self.parent = parent
        self.children = []          # ordered
        self.by_name = {}
        self.first_cluster = 0
        self.nclusters = 0


def build_tree(file_specs):
    root = Node(None, True)
    for dest, src in file_specs:
        parts = [p for p in dest.replace("\\", "/").split("/") if p]
        if not parts:
            die("empty destination path in --file %s=%s" % (dest, src))
        try:
            names = [short_name(p) for p in parts]
        except ValueError as e:
            die(str(e))
        if not os.path.isfile(src):
            die("source file not found: %s" % src)
        size = os.path.getsize(src)
        if size > 0xFFFFFFFF:
            die("%s is larger than the FAT32 file size limit (4 GiB - 1)" % src)
        cur = root
        for n in names[:-1]:
            child = cur.by_name.get(n)
            if child is None:
                child = Node(n, True, parent=cur)
                cur.children.append(child)
                cur.by_name[n] = child
            elif not child.is_dir:
                die("path conflict: %s is both a file and a directory" % dest)
            cur = child
        if names[-1] in cur.by_name:
            die("duplicate destination: %s" % dest)
        leaf = Node(names[-1], False, src=os.path.abspath(src), size=size, parent=cur)
        cur.children.append(leaf)
        cur.by_name[names[-1]] = leaf
    return root


def walk_dirs(root):
    """Directories in breadth-first order (root first)."""
    out, queue = [], [root]
    while queue:
        d = queue.pop(0)
        out.append(d)
        queue.extend(c for c in d.children if c.is_dir)
    return out


def walk_files(root):
    out = []

    def rec(d):
        for c in d.children:
            if c.is_dir:
                rec(c)
            else:
                out.append(c)
    rec(root)
    return out


def dir_entry_count(d, is_root):
    n = len(d.children)
    n += 1 if is_root else 2          # volume label entry, or "." and ".."
    return n


# --------------------------------------------------------------------------
# FAT32 geometry
# --------------------------------------------------------------------------

def default_sectors_per_cluster(part_sectors):
    # Microsoft fatgen103 table for FAT32 (512-byte sectors).
    if part_sectors <= 532480:        # <= 260 MB
        return 1
    if part_sectors <= 16777216:      # <= 8 GB
        return 8
    if part_sectors <= 33554432:      # <= 16 GB
        return 16
    if part_sectors <= 67108864:      # <= 32 GB
        return 32
    return 64


class Geometry:
    pass


def compute_geometry(total_mib, spc_override=None):
    g = Geometry()
    g.total_sectors = total_mib * MIB // SECTOR
    # Partition: from 1 MiB to 1 MiB before the end (tail holds the backup GPT).
    g.part_start = PART_START_LBA
    g.part_end = g.total_sectors - PART_START_LBA - 1           # inclusive
    g.part_sectors = g.part_end - g.part_start + 1
    if g.part_sectors <= 0:
        raise ValueError("image too small")
    spc = spc_override or default_sectors_per_cluster(g.part_sectors)
    g.spc = spc
    g.cluster_bytes = spc * SECTOR
    rsvd = RESERVED_SECTORS_MIN
    fatsz = 1
    while True:
        clusters = (g.part_sectors - rsvd - 2 * fatsz) // spc
        need = ceil_div((clusters + 2) * 4, SECTOR)
        if need <= fatsz:
            break
        fatsz = need
    # Align the data region (relative to the 1 MiB-aligned partition start)
    # to max(cluster, 4 KiB) by padding the reserved area.
    align = max(spc, 8)
    meta = rsvd + 2 * fatsz
    rsvd += (-meta) % align
    g.reserved = rsvd
    g.fat_sectors = fatsz
    g.data_start = rsvd + 2 * fatsz                       # relative to partition
    g.clusters = (g.part_sectors - g.data_start) // spc
    assert g.fat_sectors * SECTOR // 4 >= g.clusters + 2
    return g


def layout(root, g):
    """Assign contiguous clusters: directories first (BFS), then files."""
    nxt = 2
    for d in walk_dirs(root):
        is_root = d is root
        d.nclusters = max(1, ceil_div(dir_entry_count(d, is_root) * 32, g.cluster_bytes))
        d.first_cluster = nxt
        nxt += d.nclusters
    for f in walk_files(root):
        f.nclusters = ceil_div(f.size, g.cluster_bytes)
        f.first_cluster = nxt if f.nclusters else 0
        nxt += f.nclusters
    return nxt - 2            # clusters used


def clusters_needed(root, cluster_bytes):
    n = 0
    for d in walk_dirs(root):
        n += max(1, ceil_div(dir_entry_count(d, d is root) * 32, cluster_bytes))
    for f in walk_files(root):
        n += ceil_div(f.size, cluster_bytes)
    return n


def choose_size(root, size_mb, spc_override, slack_mib):
    def fits(mib, need_slack):
        try:
            g = compute_geometry(mib, spc_override)
        except ValueError:
            return None
        if g.clusters < FAT32_SAFE_MIN_CLUSTERS or g.clusters > FAT32_MAX_CLUSTERS:
            return None
        used = clusters_needed(root, g.cluster_bytes)
        free_bytes = (g.clusters - used) * g.cluster_bytes
        if used > g.clusters or free_bytes < need_slack:
            return None
        return g

    if size_mb is not None:
        g = fits(size_mb, 0)
        if g is None:
            try:
                g2 = compute_geometry(size_mb, spc_override)
                why = "%d clusters of %d bytes, files need %d" % (
                    g2.clusters, g2.cluster_bytes, clusters_needed(root, g2.cluster_bytes))
            except ValueError:
                why = "no room for a partition"
            die("--size-mb %d does not fit the files as valid FAT32 (%s; FAT32 needs >= %d clusters)"
                % (size_mb, why, FAT32_SAFE_MIN_CLUSTERS))
        return size_mb, g

    data = sum(f.size for f in walk_files(root))
    mib = max(40, ceil_div(data, MIB) + slack_mib + 4)
    for _ in range(100000):
        g = fits(mib, slack_mib * MIB)
        if g is not None:
            return mib, g
        mib += max(1, mib // 200)
    die("could not find a fitting image size")


# --------------------------------------------------------------------------
# Structures
# --------------------------------------------------------------------------

def dos_datetime(dt):
    d = ((dt.year - 1980) << 9) | (dt.month << 5) | dt.day
    t = (dt.hour << 11) | (dt.minute << 5) | (dt.second // 2)
    return d, t


def dirent(name11, attr, cluster, size, date, time):
    return struct.pack("<11sBBBHHHHHHHI", name11, attr, 0, 0, time, date, date,
                       (cluster >> 16) & 0xFFFF, time, date, cluster & 0xFFFF, size)


def boot_sector(g, vol_id):
    bs = bytearray(SECTOR)
    bs[0:3] = b"\xEB\x58\x90"
    bs[3:11] = b"MSWIN4.1"
    struct.pack_into("<HBHBHHBHHHII", bs, 11,
                     SECTOR, g.spc, g.reserved, 2, 0, 0, 0xF8, 0, 63, 255,
                     g.part_start, g.part_sectors)
    struct.pack_into("<IHHIHH12sBBBI11s8s", bs, 36,
                     g.fat_sectors, 0, 0, 2, FSINFO_SECTOR, BACKUP_BOOT_SECTOR, b"\0" * 12,
                     0x80, 0, 0x29, vol_id, VOLUME_LABEL, b"FAT32   ")
    # Boot code: not BIOS-bootable; just halt forever.
    bs[0x5A:0x5D] = b"\xF4\xEB\xFD"
    bs[510:512] = b"\x55\xAA"
    return bytes(bs)


def fsinfo_sector(free_count, next_free):
    s = bytearray(SECTOR)
    struct.pack_into("<I", s, 0, 0x41615252)
    struct.pack_into("<III", s, 484, 0x61417272, free_count, next_free)
    struct.pack_into("<I", s, 508, 0xAA550000)
    return bytes(s)


def third_boot_sector():
    s = bytearray(SECTOR)
    s[510:512] = b"\x55\xAA"
    return bytes(s)


def protective_mbr(total_sectors):
    mbr = bytearray(SECTOR)
    size = min(total_sectors - 1, 0xFFFFFFFF)
    mbr[446:462] = struct.pack("<B3sB3sII", 0x00, b"\x00\x02\x00", 0xEE, b"\xFF\xFF\xFF", 1, size)
    mbr[510:512] = b"\x55\xAA"
    return bytes(mbr)


def gpt_entries(g, part_guid):
    arr = bytearray(GPT_ENTRIES * GPT_ENTRY_SIZE)
    name = "EFI System Partition".encode("utf-16-le").ljust(72, b"\0")
    struct.pack_into("<16s16sQQQ72s", arr, 0, ESP_TYPE_GUID.bytes_le, part_guid.bytes_le,
                     g.part_start, g.part_end, 0, name)
    return bytes(arr)


def gpt_header(my_lba, alt_lba, entries_lba, g, disk_guid, entries_crc):
    first_usable = 2 + GPT_ARRAY_SECTORS
    last_usable = g.total_sectors - 2 - GPT_ARRAY_SECTORS
    hdr = bytearray(struct.pack("<8sIIIIQQQQ16sQIII", b"EFI PART", 0x00010000, 92, 0, 0,
                                my_lba, alt_lba, first_usable, last_usable, disk_guid.bytes_le,
                                entries_lba, GPT_ENTRIES, GPT_ENTRY_SIZE, entries_crc))
    struct.pack_into("<I", hdr, 16, zlib.crc32(hdr) & 0xFFFFFFFF)
    return bytes(hdr).ljust(SECTOR, b"\0")


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------

def try_make_sparse(f):
    if os.name != "nt":
        return False
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID,
                                        wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                        ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
        k32.DeviceIoControl.restype = wintypes.BOOL
        handle = msvcrt.get_osfhandle(f.fileno())
        ret = wintypes.DWORD(0)
        FSCTL_SET_SPARSE = 0x000900C4
        return bool(k32.DeviceIoControl(handle, FSCTL_SET_SPARSE, None, 0, None, 0,
                                        ctypes.byref(ret), None))
    except Exception:
        return False


def build(args):
    specs = []
    for s in args.file or []:
        if "=" not in s:
            die("--file expects DEST=SRC, got %r" % s)
        dest, src = s.split("=", 1)
        specs.append((dest, src))
    if not specs:
        die("no --file given")
    out_abs = os.path.abspath(args.image)
    for _, src in specs:
        if os.path.exists(out_abs) and os.path.samefile(src, out_abs):
            die("output image is also an input file")

    root = build_tree(specs)
    spc_override = None
    if args.cluster_kb:
        if args.cluster_kb not in (1, 2, 4, 8, 16, 32) and args.cluster_kb != 0.5:
            die("--cluster-kb must be one of 0.5,1,2,4,8,16,32")
        spc_override = int(args.cluster_kb * 1024) // SECTOR
    size_mb, g = choose_size(root, args.size_mb, spc_override, args.slack_mb)
    used = layout(root, g)
    free = g.clusters - used

    now = datetime.datetime.now()
    date, time = dos_datetime(now)
    vol_id = int.from_bytes(os.urandom(4), "little")
    disk_guid, part_guid = uuid.uuid4(), uuid.uuid4()

    # FAT (only the used prefix is non-zero).
    fat = bytearray((used + 2) * 4)
    struct.pack_into("<II", fat, 0, 0x0FFFFFF8, 0x0FFFFFFF)
    for node in walk_dirs(root) + walk_files(root):
        c0, n = node.first_cluster, node.nclusters
        for i in range(n):
            struct.pack_into("<I", fat, (c0 + i) * 4, (c0 + i + 1) if i < n - 1 else EOC)

    part_off = g.part_start * SECTOR
    fat_off = [part_off + (g.reserved + i * g.fat_sectors) * SECTOR for i in range(2)]
    data_off = part_off + g.data_start * SECTOR

    def cluster_off(c):
        return data_off + (c - 2) * g.cluster_bytes

    entries = gpt_entries(g, part_guid)
    ecrc = zlib.crc32(entries) & 0xFFFFFFFF
    last = g.total_sectors - 1
    backup_entries_lba = last - GPT_ARRAY_SECTORS

    bs = boot_sector(g, vol_id)
    fsi = fsinfo_sector(free, 2 + used if free else 0xFFFFFFFF)

    with open(args.image, "wb") as f:
        sparse = try_make_sparse(f)
        f.truncate(g.total_sectors * SECTOR)

        def put(off, data):
            f.seek(off)
            f.write(data)

        put(0, protective_mbr(g.total_sectors))
        put(SECTOR, gpt_header(1, last, 2, g, disk_guid, ecrc))
        put(2 * SECTOR, entries)
        # FAT32 reserved region: boot, FSInfo, 3rd boot sector, and backups at +6.
        for base in (0, BACKUP_BOOT_SECTOR):
            put(part_off + (base + 0) * SECTOR, bs + fsi + third_boot_sector())
        for off in fat_off:
            put(off, fat)
        # Directories.
        for d in walk_dirs(root):
            buf = bytearray()
            if d is root:
                buf += dirent(VOLUME_LABEL, ATTR_VOLUME_ID, 0, 0, date, time)
            else:
                parent_cl = 0 if d.parent is root else d.parent.first_cluster
                buf += dirent(b".          ", ATTR_DIRECTORY, d.first_cluster, 0, date, time)
                buf += dirent(b"..         ", ATTR_DIRECTORY, parent_cl, 0, date, time)
            for c in d.children:
                if c.is_dir:
                    buf += dirent(c.name11, ATTR_DIRECTORY, c.first_cluster, 0, date, time)
                else:
                    buf += dirent(c.name11, ATTR_ARCHIVE, c.first_cluster, c.size, date, time)
            put(cluster_off(d.first_cluster), bytes(buf))   # remainder of cluster stays zero
        # Files, streamed.
        for fnode in walk_files(root):
            if not fnode.size:
                continue
            f.seek(cluster_off(fnode.first_cluster))
            copied = 0
            with open(fnode.src, "rb") as src:
                while True:
                    chunk = src.read(COPY_CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    copied += len(chunk)
            if copied != fnode.size:
                die("%s changed size while copying" % fnode.src)
        # Backup GPT.
        put(backup_entries_lba * SECTOR, entries)
        put(last * SECTOR, gpt_header(last, 1, backup_entries_lba, g, disk_guid, ecrc))
        f.truncate(g.total_sectors * SECTOR)

    print("wrote %s: %d MiB (%d sectors), sparse=%s" % (args.image, size_mb, g.total_sectors, sparse))
    print("  ESP  LBA %d..%d (%d MiB), FAT32 %d B clusters x %d, FAT %d sectors x2, reserved %d"
          % (g.part_start, g.part_end, g.part_sectors * SECTOR // MIB, g.cluster_bytes,
             g.clusters, g.fat_sectors, g.reserved))
    print("  used %d clusters, free %d clusters (%.1f MiB)"
          % (used, free, free * g.cluster_bytes / MIB))
    for fnode in walk_files(root):
        print("  %-24s %12d bytes  clusters %d..%d" % (path_of(fnode), fnode.size,
              fnode.first_cluster, fnode.first_cluster + max(fnode.nclusters, 1) - 1))
    return specs


def path_of(node):
    parts = []
    while node.parent is not None:
        parts.append(display_name(node.name11))
        node = node.parent
    return "/".join(reversed(parts))


def display_name(name11):
    if name11[0] == 0x05:
        name11 = b"\xE5" + name11[1:]
    base = name11[:8].rstrip(b" ").decode("latin-1")
    ext = name11[8:].rstrip(b" ").decode("latin-1")
    return base + ("." + ext if ext else "")


# --------------------------------------------------------------------------
# Independent verifier
# --------------------------------------------------------------------------

class VerifyError(Exception):
    pass


def check(cond, msg):
    if not cond:
        raise VerifyError(msg)


def verify(image, specs, strict=False):
    """Parse the image from scratch and check every structure. Returns file map.

    FSInfo contents are advisory per the FAT spec and other FAT drivers (e.g.
    OVMF writing NvVars) update only the primary copy, so FSInfo mismatches are
    warnings unless strict (used for --check right after building).
    """
    warnings = []

    def soft(cond, msg):
        if not cond:
            if strict:
                raise VerifyError(msg)
            warnings.append(msg)
            print("WARN: " + msg)
    size = os.path.getsize(image)
    check(size % SECTOR == 0, "image size not a multiple of 512")
    nsec = size // SECTOR
    with open(image, "rb") as f:
        def rd(lba, count=1):
            f.seek(lba * SECTOR)
            data = f.read(count * SECTOR)
            check(len(data) == count * SECTOR, "short read at LBA %d" % lba)
            return data

        # --- protective MBR
        mbr = rd(0)
        check(mbr[510:512] == b"\x55\xAA", "MBR signature missing")
        parts = [mbr[446 + 16 * i: 462 + 16 * i] for i in range(4)]
        check(parts[0][4] == 0xEE, "MBR partition 1 is not type 0xEE")
        check(all(p == b"\0" * 16 for p in parts[1:]), "extra MBR partitions present")
        st, = struct.unpack_from("<I", parts[0], 8)
        ln, = struct.unpack_from("<I", parts[0], 12)
        check(st == 1 and ln == min(nsec - 1, 0xFFFFFFFF), "protective MBR range wrong")
        check(parts[0][0] == 0, "protective MBR entry marked bootable")
        print("MBR: protective 0xEE entry OK")

        # --- GPT headers
        def parse_hdr(lba):
            raw = rd(lba)
            (sig, rev, hsz, crc, _r, my, alt, fu, lu, dg, elba, ne, esz, ecrc) = \
                struct.unpack_from("<8sIIIIQQQQ16sQIII", raw)
            check(sig == b"EFI PART", "GPT signature missing at LBA %d" % lba)
            check(rev == 0x00010000 and hsz == 92, "GPT revision/size wrong at LBA %d" % lba)
            h = bytearray(raw[:hsz])
            h[16:20] = b"\0\0\0\0"
            check(zlib.crc32(h) & 0xFFFFFFFF == crc, "GPT header CRC mismatch at LBA %d" % lba)
            check(my == lba, "GPT MyLBA wrong at LBA %d" % lba)
            check(esz == 128 and ne * esz <= 0x100000, "GPT entry geometry odd")
            arr = rd(elba, ceil_div(ne * esz, SECTOR))[:ne * esz]
            check(zlib.crc32(arr) & 0xFFFFFFFF == ecrc, "GPT entry array CRC mismatch (hdr LBA %d)" % lba)
            return dict(alt=alt, fu=fu, lu=lu, dg=dg, elba=elba, ne=ne, esz=esz, arr=arr)

        p = parse_hdr(1)
        check(p["alt"] == nsec - 1, "primary AlternateLBA != last LBA")
        b = parse_hdr(nsec - 1)
        check(b["alt"] == 1, "backup AlternateLBA != 1")
        check(p["elba"] == 2, "primary entries not at LBA 2")
        check(b["elba"] + ceil_div(b["ne"] * b["esz"], SECTOR) == nsec - 1,
              "backup entries not directly before backup header")
        check(p["arr"] == b["arr"] and p["dg"] == b["dg"] and p["fu"] == b["fu"] and p["lu"] == b["lu"],
              "primary and backup GPT disagree")
        check(p["fu"] >= 2 + ceil_div(p["ne"] * p["esz"], SECTOR), "FirstUsableLBA overlaps entries")
        check(p["lu"] < b["elba"], "LastUsableLBA overlaps backup entries")
        print("GPT: primary + backup headers and entry arrays OK (CRC32s valid), disk GUID %s"
              % uuid.UUID(bytes_le=p["dg"]))
        esp = None
        for i in range(p["ne"]):
            e = p["arr"][i * p["esz"]:(i + 1) * p["esz"]]
            tg = uuid.UUID(bytes_le=e[:16])
            if tg.int == 0:
                continue
            s, en = struct.unpack_from("<QQ", e, 32)
            name = e[56:128].decode("utf-16-le").rstrip("\0")
            print("GPT: partition %d type %s LBA %d..%d %r" % (i + 1, str(tg).upper(), s, en, name))
            check(p["fu"] <= s <= en <= p["lu"], "partition outside usable range")
            if tg == ESP_TYPE_GUID:
                check(esp is None, "more than one ESP")
                esp = (s, en)
        check(esp is not None, "no EFI System Partition found")
        ps, pe = esp
        check(ps % 2048 == 0, "ESP not 1 MiB-aligned")

        # --- FAT32 BPB
        bs = rd(ps)
        check(bs[510:512] == b"\x55\xAA", "boot sector signature missing")
        check(bs[0] in (0xEB, 0xE9), "boot sector jump missing")
        (bps, spc, rsvd, nfats, rootent, tot16, media, fat16sz, _spt, _nh, hidd, tot32) = \
            struct.unpack_from("<HBHBHHBHHHII", bs, 11)
        (fatsz, extf, fsver, rootclus, fsinfo, bkboot) = struct.unpack_from("<IHHIHH", bs, 36)
        check(bps == 512, "BytsPerSec != 512")
        check(spc in (1, 2, 4, 8, 16, 32, 64, 128), "bad SecPerClus")
        check(nfats == 2, "NumFATs != 2")
        check(rootent == 0 and tot16 == 0 and fat16sz == 0, "FAT12/16 fields set")
        check(tot32 == pe - ps + 1, "TotSec32 (%d) != partition size (%d)" % (tot32, pe - ps + 1))
        check(hidd == ps, "HiddSec != partition start")
        check(fsver == 0 and extf == 0, "unexpected FSVer/ExtFlags")
        check(bs[66] == 0x29 and bs[82:90] == b"FAT32   ", "extended BPB missing")
        data_start = rsvd + nfats * fatsz
        clusters = (tot32 - data_start) // spc
        check(clusters >= FAT32_MIN_CLUSTERS, "only %d clusters: that is FAT16 by spec" % clusters)
        check(fatsz * bps // 4 >= clusters + 2, "FAT too small for cluster count")
        cb = spc * bps
        print("FAT32: %d clusters x %d B, reserved %d, FAT %d sectors x2, data at +%d sectors"
              % (clusters, cb, rsvd, fatsz, data_start))
        check(bkboot == 6 and rd(ps + 6) == bs, "backup boot sector at 6 missing/different")
        fsi = rd(ps + fsinfo)
        check(struct.unpack_from("<I", fsi, 0)[0] == 0x41615252 and
              struct.unpack_from("<I", fsi, 484)[0] == 0x61417272 and
              struct.unpack_from("<I", fsi, 508)[0] == 0xAA550000, "FSInfo signatures wrong")
        bfsi = rd(ps + 6 + fsinfo)
        check(bfsi[0:4] == fsi[0:4] and bfsi[484:488] == fsi[484:488] and bfsi[508:512] == fsi[508:512],
              "backup FSInfo signatures wrong")
        soft(bfsi == fsi, "backup FSInfo differs from primary (normal after another driver wrote)")

        # --- FATs
        fat_raw = [rd(ps + rsvd + i * fatsz, fatsz) for i in range(nfats)]
        check(fat_raw[0] == fat_raw[1], "FAT copies differ")
        fat = struct.unpack("<%dI" % (clusters + 2), fat_raw[0][:(clusters + 2) * 4])
        check(fat[0] & 0x0FFFFFFF == 0x0FFFFF00 | media and media == 0xF8, "FAT[0] media mismatch")
        check(fat[1] & 0x0C000000 == 0x0C000000, "FAT[1] dirty/error bits")
        free = sum(1 for v in fat[2:] if v & 0x0FFFFFFF == 0)
        fs_free, fs_next = struct.unpack_from("<II", fsi, 488)
        soft(fs_free in (free, 0xFFFFFFFF), "FSInfo free count %d != actual %d" % (fs_free, free))
        print("FAT: both copies identical, %d free clusters (FSInfo agrees), next-free hint %d"
              % (free, fs_next))

        owner = {}

        def chain(first, what):
            out = []
            c = first
            while True:
                check(2 <= c < clusters + 2, "%s: cluster %d out of range" % (what, c))
                check(c not in owner, "%s: cluster %d cross-linked with %s" % (what, c, owner.get(c)))
                owner[c] = what
                out.append(c)
                v = fat[c] & 0x0FFFFFFF
                if v >= 0x0FFFFFF8:
                    return out
                check(v not in (0, 1, 0x0FFFFFF7), "%s: bad FAT link %#x at %d" % (what, v, c))
                c = v

        def coff(c):
            return (ps + data_start + (c - 2) * spc) * SECTOR

        files = {}

        def walk(first, path, parent_first):
            cl = chain(first, path or "/")
            raw = b"".join(rd(ps + data_start + (c - 2) * spc, spc) for c in cl)
            ents = [raw[i:i + 32] for i in range(0, len(raw), 32)]
            idx = 0
            if path:  # non-root: '.' and '..' must lead
                check(len(ents) >= 2, "%s: too small" % path)
                d0, d1 = ents[0], ents[1]
                check(d0[:11] == b".          " and d0[11] & ATTR_DIRECTORY, "%s: missing '.'" % path)
                check(d1[:11] == b"..         " and d1[11] & ATTR_DIRECTORY, "%s: missing '..'" % path)
                c0 = struct.unpack_from("<H", d0, 20)[0] << 16 | struct.unpack_from("<H", d0, 26)[0]
                c1 = struct.unpack_from("<H", d1, 20)[0] << 16 | struct.unpack_from("<H", d1, 26)[0]
                check(c0 == first, "%s: '.' points to %d not %d" % (path, c0, first))
                check(c1 == parent_first, "%s: '..' points to %d not %d" % (path, c1, parent_first))
                idx = 2
            for e in ents[idx:]:
                if e[0] == 0:
                    break
                if e[0] == 0xE5 or e[11] == ATTR_LFN:
                    continue
                attr = e[11]
                if attr & ATTR_VOLUME_ID:
                    check(not path, "volume label outside root")
                    continue
                name = display_name(e[:11])
                full = (path + "/" + name) if path else name
                c = struct.unpack_from("<H", e, 20)[0] << 16 | struct.unpack_from("<H", e, 26)[0]
                fsize = struct.unpack_from("<I", e, 28)[0]
                if attr & ATTR_DIRECTORY:
                    check(name not in (".", ".."), "stray dot entry in %s" % (path or "/"))
                    walk(c, full, first if path else 0)
                else:
                    if fsize == 0:
                        check(c == 0, "%s: empty file with a cluster" % full)
                        files[full.upper()] = (0, hashlib.sha256().hexdigest(), True)
                        continue
                    cl2 = chain(c, full)
                    check(len(cl2) == ceil_div(fsize, cb),
                          "%s: chain has %d clusters, size needs %d" % (full, len(cl2), ceil_div(fsize, cb)))
                    contiguous = all(cl2[i + 1] == cl2[i] + 1 for i in range(len(cl2) - 1))
                    h = hashlib.sha256()
                    remaining = fsize
                    if contiguous:
                        f.seek(coff(cl2[0]))
                        while remaining:
                            chunk = f.read(min(COPY_CHUNK, remaining))
                            check(chunk, "%s: truncated image" % full)
                            h.update(chunk)
                            remaining -= len(chunk)
                    else:
                        for cc in cl2:
                            f.seek(coff(cc))
                            chunk = f.read(min(cb, remaining))
                            h.update(chunk)
                            remaining -= len(chunk)
                    files[full.upper()] = (fsize, h.hexdigest(), contiguous)

        walk(rootclus, "", 0)
        used = len(owner)
        check(used + free == clusters, "lost clusters: %d allocated in FAT but unreferenced"
              % (clusters - free - used))
        for name in sorted(files):
            sz, hx, contig = files[name]
            print("file: %-24s %12d bytes sha256 %s%s" % (name, sz, hx[:16],
                                                          "" if contig else " (FRAGMENTED)"))

    # --- compare against sources
    bad = 0
    for dest, src in specs:
        key = "/".join(p for p in dest.replace("\\", "/").upper().split("/") if p)
        if key not in files:
            print("MISSING in image: %s" % key)
            bad += 1
            continue
        h = hashlib.sha256()
        with open(src, "rb") as s:
            while True:
                chunk = s.read(COPY_CHUNK)
                if not chunk:
                    break
                h.update(chunk)
        sz, hx, _ = files[key]
        if hx != h.hexdigest() or sz != os.path.getsize(src):
            print("MISMATCH: %s vs %s" % (key, src))
            bad += 1
        else:
            print("match: %s == %s" % (key, src))
    if bad:
        raise VerifyError("%d file(s) missing or different" % bad)
    print("VERIFY OK" + (" (%d warning(s))" % len(warnings) if warnings else ""))
    return files


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Build a GPT + FAT32 ESP raw disk image.")
    ap.add_argument("image", help="output image (or image to check with --verify)")
    ap.add_argument("--file", action="append", metavar="DEST=SRC",
                    help="place SRC at DEST (8.3 path, e.g. EFI/BOOT/BOOTX64.EFI)")
    ap.add_argument("--size-mb", type=int, help="total image size in MiB (default: auto)")
    ap.add_argument("--slack-mb", type=int, default=DEFAULT_SLACK_MIB,
                    help="free space to leave when auto-sizing (default %d)" % DEFAULT_SLACK_MIB)
    ap.add_argument("--cluster-kb", type=float, help="force cluster size (0.5..32 KiB)")
    ap.add_argument("--verify", action="store_true",
                    help="do not build; verify IMAGE (against --file sources if given)")
    ap.add_argument("--check", action="store_true", help="verify right after building")
    args = ap.parse_args(argv)

    if args.verify:
        specs = [tuple(s.split("=", 1)) for s in (args.file or [])]
        for s in specs:
            if len(s) != 2:
                die("--file expects DEST=SRC")
        try:
            verify(args.image, specs)
        except VerifyError as e:
            die("VERIFY FAILED: %s" % e)
        return
    specs = build(args)
    if args.check:
        try:
            verify(args.image, specs, strict=True)
        except VerifyError as e:
            die("VERIFY FAILED: %s" % e)


if __name__ == "__main__":
    main()
