"""Boot metallm with Secure Boot ON, through shim, signed with your own key.

    python tools/secureboot.py keygen             # once: keys/MOK.key, MOK.crt, MOK.cer
    python tools/secureboot.py stick E:\\          # sign build/BOOTX64.EFI and set up the stick
    python tools/secureboot.py sign IN.EFI OUT.EFI # just sign a file

On the stick this puts:
    EFI\\BOOT\\BOOTX64.EFI   shim, signed by Microsoft (from Ubuntu's shim-signed package)
    EFI\\BOOT\\mmx64.efi     MokManager, signed by Canonical (same package)
    EFI\\BOOT\\grubx64.efi   metallm, signed with keys/MOK.key (shim starts this name)
    METALLM.CER             your public key, for MokManager to enroll

On the first boot shim can't verify metallm yet and opens MokManager: "Enroll key from disk",
pick METALLM.CER, confirm, reboot. From then on the key is in the machine's MOK list and every
build you sign with it boots. Keep keys/MOK.key private: anything signed with it boots on this
machine with Secure Boot on.

Signing is plain Python (only needs `cryptography`), so it works on Windows without sbsign.
"""
import argparse, datetime, hashlib, io, lzma, os, shutil, struct, sys, tarfile, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = os.path.join(ROOT, "keys")
SHIM_CACHE = os.path.join(ROOT, "build", "shim")
UBUNTU = "http://archive.ubuntu.com/ubuntu"
POCKETS = ["noble-updates", "noble"]   # Ubuntu 24.04; -updates has the newest shim if there is one


# ------------------------------------------------------------------ minimal DER encoder
def _len(n):
    if n < 0x80: return bytes([n])
    b = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(b)]) + b

def der(tag, body): return bytes([tag]) + _len(len(body)) + body
def seq(*xs): return der(0x30, b"".join(xs))
def set_of(*xs): return der(0x31, b"".join(sorted(xs)))   # DER SET OF: sorted encodings
def ctx(n, body, constructed=True): return der((0xA0 if constructed else 0x80) | n, body)
def integer(i): return der(0x02, i.to_bytes(i.bit_length() // 8 + 1, "big"))
def octets(b): return der(0x04, b)
NULL = b"\x05\x00"

def oid(s):
    a = [int(x) for x in s.split(".")]
    out = bytearray([40 * a[0] + a[1]])
    for v in a[2:]:
        enc = [v & 0x7F]
        while v > 0x7F:
            v >>= 7; enc.append(0x80 | (v & 0x7F))
        out += bytes(reversed(enc))
    return der(0x06, bytes(out))

def strip_tl(d):  # contents octets of a DER element (drop tag and length)
    n = d[1]
    return d[2:] if n < 0x80 else d[2 + (n & 0x7F):]

OID_SIGNED_DATA   = "1.2.840.113549.1.7.2"
OID_SHA256        = "2.16.840.1.101.3.4.2.1"
OID_RSA           = "1.2.840.113549.1.1.1"
OID_CONTENT_TYPE  = "1.2.840.113549.1.9.3"
OID_MESSAGE_DIGEST = "1.2.840.113549.1.9.4"
OID_SPC_INDIRECT  = "1.3.6.1.4.1.311.2.1.4"
OID_SPC_PE_IMAGE  = "1.3.6.1.4.1.311.2.1.15"
OID_SPC_OPUS_INFO = "1.3.6.1.4.1.311.2.1.12"


# ------------------------------------------------------------------ PE / Authenticode
class PE:
    def __init__(self, data):
        self.d = bytearray(data)
        d = self.d
        if d[:2] != b"MZ": raise ValueError("not a PE file (no MZ)")
        pe = struct.unpack_from("<I", d, 0x3C)[0]
        if d[pe:pe + 4] != b"PE\0\0": raise ValueError("not a PE file (no PE signature)")
        coff = pe + 4
        self.nsec, = struct.unpack_from("<H", d, coff + 2)
        opt_size, = struct.unpack_from("<H", d, coff + 16)
        opt = coff + 20
        magic, = struct.unpack_from("<H", d, opt)
        if magic != 0x20B: raise ValueError("expected a PE32+ (64-bit) image")
        self.checksum_off = opt + 64
        self.size_of_headers, = struct.unpack_from("<I", d, opt + 60)
        ndirs, = struct.unpack_from("<I", d, opt + 108)
        if ndirs < 5: raise ValueError("no certificate table directory entry")
        self.certdir_off = opt + 112 + 4 * 8
        self.sec_off = opt + opt_size

    def cert_table(self):
        return struct.unpack_from("<II", self.d, self.certdir_off)

    def sections(self):
        out = []
        for i in range(self.nsec):
            o = self.sec_off + 40 * i
            raw_size, raw_ptr = struct.unpack_from("<II", self.d, o + 16)
            if raw_size: out.append((raw_ptr, raw_size))
        return sorted(out)

    def authenticode_hash(self):
        """SHA-256 per the Authenticode PE spec (what firmware and shim compute)."""
        d, h = self.d, hashlib.sha256()
        h.update(d[:self.checksum_off])
        h.update(d[self.checksum_off + 4:self.certdir_off])
        h.update(d[self.certdir_off + 8:self.size_of_headers])
        done = self.size_of_headers
        for ptr, size in self.sections():
            h.update(d[ptr:ptr + size]); done += size
        _, cert_size = self.cert_table()
        end = len(d) - cert_size
        if end > done: h.update(d[done:end])
        return h.digest()

    def update_checksum(self):
        d = self.d
        struct.pack_into("<I", d, self.checksum_off, 0)
        if len(d) % 2: d += b"\0"
        s = 0
        for (w,) in struct.iter_unpack("<H", bytes(d)):
            s += w; s = (s & 0xFFFF) + (s >> 16)
        s = (s & 0xFFFF) + (s >> 16)
        struct.pack_into("<I", d, self.checksum_off, (s + len(d)) & 0xFFFFFFFF)


def sign_pe(data, key, cert):
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    pe = PE(data)
    if pe.cert_table() != (0, 0):
        raise ValueError("image is already signed (certificate table present)")
    while len(pe.d) % 8: pe.d += b"\0"     # the table must start 8-aligned; padding is hashed
    digest = pe.authenticode_hash()

    alg = seq(oid(OID_SHA256), NULL)
    obsolete = "<<<Obsolete>>>".encode("utf-16-be")
    pe_image_data = seq(der(0x03, b"\0"), ctx(0, ctx(2, ctx(0, obsolete, constructed=False))))
    indirect = seq(seq(oid(OID_SPC_PE_IMAGE), pe_image_data), seq(alg, octets(digest)))
    attrs = [
        seq(oid(OID_CONTENT_TYPE), set_of(oid(OID_SPC_INDIRECT))),
        seq(oid(OID_SPC_OPUS_INFO), set_of(seq())),
        # Authenticode hashes the SpcIndirectDataContent without its own tag and length
        seq(oid(OID_MESSAGE_DIGEST), set_of(octets(hashlib.sha256(strip_tl(indirect)).digest()))),
    ]
    signature = key.sign(set_of(*attrs), padding.PKCS1v15(), hashes.SHA256())
    cert_der = cert.public_bytes(serialization.Encoding.DER)
    signer = seq(integer(1),
                 seq(cert.issuer.public_bytes(), integer(cert.serial_number)),
                 alg,
                 ctx(0, b"".join(sorted(attrs))),
                 seq(oid(OID_RSA), NULL),
                 octets(signature))
    signed_data = seq(integer(1), set_of(alg),
                      seq(oid(OID_SPC_INDIRECT), ctx(0, indirect)),
                      ctx(0, cert_der),
                      set_of(signer))
    pkcs7 = seq(oid(OID_SIGNED_DATA), ctx(0, signed_data))

    win_cert = struct.pack("<IHH", 8 + len(pkcs7), 0x0200, 0x0002) + pkcs7  # WIN_CERT_TYPE_PKCS_SIGNED_DATA
    win_cert += b"\0" * (-len(win_cert) % 8)
    struct.pack_into("<II", pe.d, pe.certdir_off, len(pe.d), len(win_cert))
    pe.d += win_cert
    pe.update_checksum()
    return bytes(pe.d)


# ------------------------------------------------------------------ keys
def key_paths(kdir):
    return [os.path.join(kdir, n) for n in ("MOK.key", "MOK.crt", "MOK.cer")]

def load_keys(kdir):
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
    kp, cp, _ = key_paths(kdir)
    if not os.path.exists(kp):
        sys.exit(f"no key in {kdir}; run: python tools/secureboot.py keygen")
    with open(kp, "rb") as f: key = serialization.load_pem_private_key(f.read(), None)
    with open(cp, "rb") as f: cert = x509.load_pem_x509_certificate(f.read())
    return key, cert

def cmd_keygen(a):
    from cryptography import x509
    from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    paths = key_paths(a.keys)
    if any(os.path.exists(p) for p in paths) and not a.force:
        sys.exit(f"keys already exist in {a.keys} (use --force to replace them; builds signed with "
                 "the old key then stop booting until you enroll the new one)")
    os.makedirs(a.keys, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, a.name)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=365 * 30))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(True, False, False, False, False, False, False, False, False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CODE_SIGNING]), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .sign(key, hashes.SHA256()))
    kp, cp, dp = paths
    with open(kp, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()))
    with open(cp, "wb") as f: f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(dp, "wb") as f: f.write(cert.public_bytes(serialization.Encoding.DER))
    print(f"wrote {kp} (private, keep it), {cp}, {dp} (enroll this one)")

def cmd_sign(a):
    key, cert = load_keys(a.keys)
    with open(a.input, "rb") as f: data = f.read()
    out = sign_pe(data, key, cert)
    with open(a.output, "wb") as f: f.write(out)
    print(f"signed {a.input} -> {a.output}")


# ------------------------------------------------------------------ shim from Ubuntu
def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r: return r.read()

def _deb_files(deb, wanted):
    """Extract files from a .deb (ar archive holding data.tar.xz)."""
    if deb[:8] != b"!<arch>\n": raise ValueError("not a .deb")
    i, out = 8, {}
    while i < len(deb):
        name = deb[i:i + 16].decode().strip().rstrip("/"); size = int(deb[i + 48:i + 58])
        body = deb[i + 60:i + 60 + size]; i += 60 + size + (size & 1)
        if name.startswith("data.tar"):
            if not name.endswith(".xz"): raise ValueError(f"unsupported {name}")
            with tarfile.open(fileobj=io.BytesIO(lzma.decompress(body))) as t:
                for m in t.getmembers():
                    if m.name.lstrip("./") in wanted: out[m.name.lstrip("./")] = t.extractfile(m).read()
    return out

SHIM_IN_DEB = {"usr/lib/shim/shimx64.efi.signed.latest": "shimx64.efi",
               "usr/lib/shim/mmx64.efi": "mmx64.efi"}

def fetch_shim(refresh=False):
    paths = {v: os.path.join(SHIM_CACHE, v) for v in SHIM_IN_DEB.values()}
    if not refresh and all(os.path.exists(p) for p in paths.values()): return paths
    best = None
    for pocket in POCKETS:
        idx = lzma.decompress(_get(f"{UBUNTU}/dists/{pocket}/main/binary-amd64/Packages.xz")).decode()
        for para in idx.split("\n\n"):
            f = dict(l.split(": ", 1) for l in para.splitlines() if ": " in l and not l.startswith(" "))
            if f.get("Package") == "shim-signed":
                best = f; break
        if best: break
    if not best: sys.exit("shim-signed not found in the Ubuntu archive index")
    print(f"downloading shim-signed {best['Version']} from {UBUNTU}")
    deb = _get(f"{UBUNTU}/{best['Filename']}")
    if hashlib.sha256(deb).hexdigest() != best["SHA256"]: sys.exit("shim-signed download: SHA256 mismatch")
    files = _deb_files(deb, SHIM_IN_DEB)
    if len(files) != len(SHIM_IN_DEB): sys.exit(f"shim-signed package lacks {set(SHIM_IN_DEB) - set(files)}")
    os.makedirs(SHIM_CACHE, exist_ok=True)
    for src, dst in SHIM_IN_DEB.items():
        with open(paths[dst], "wb") as f: f.write(files[src])
    with open(os.path.join(SHIM_CACHE, "VERSION"), "w") as f: f.write(best["Version"] + "\n")
    return paths

def cmd_stick(a):
    key, cert = load_keys(a.keys)
    with open(a.efi, "rb") as f: app = f.read()
    if b".sbat" not in app[:4096]:
        sys.exit(f"{a.efi} has no .sbat section (shim won't start it); rebuild with the current build.sh")
    signed = sign_pe(app, key, cert)
    shim = fetch_shim(a.refresh_shim)
    boot = os.path.join(a.root, "EFI", "BOOT")
    os.makedirs(boot, exist_ok=True)
    shutil.copyfile(shim["shimx64.efi"], os.path.join(boot, "BOOTX64.EFI"))
    shutil.copyfile(shim["mmx64.efi"], os.path.join(boot, "mmx64.efi"))
    with open(os.path.join(boot, "grubx64.efi"), "wb") as f: f.write(signed)
    shutil.copyfile(key_paths(a.keys)[2], os.path.join(a.root, "METALLM.CER"))
    print(f"set up {a.root}: EFI\\BOOT\\BOOTX64.EFI (shim), mmx64.efi, grubx64.efi (metallm, signed), METALLM.CER")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keys", default=KEYS, help="key directory (default: keys/)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keygen", help="make a signing key and certificate")
    k.add_argument("--name", default="metallm Secure Boot (local key)")
    k.add_argument("--force", action="store_true")
    s = sub.add_parser("sign", help="sign a PE/EFI file")
    s.add_argument("input"); s.add_argument("output")
    t = sub.add_parser("stick", help="sign metallm and put it behind shim on a stick (or folder)")
    t.add_argument("root", help="the stick's root, e.g. E:\\")
    t.add_argument("--efi", default=os.path.join(ROOT, "build", "BOOTX64.EFI"))
    t.add_argument("--refresh-shim", action="store_true", help="download shim again")
    a = ap.parse_args()
    {"keygen": cmd_keygen, "sign": cmd_sign, "stick": cmd_stick}[a.cmd](a)

if __name__ == "__main__":
    main()
