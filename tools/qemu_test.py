"""Boot the disk image in QEMU+OVMF headlessly (no window), type into it, and take screenshots.

usage: python tools/qemu_test.py disk.img outdir [--smp N] [--accel whpx|tcg] [--type "message" | --type "@pgup end"]...
Screenshots are written as PNG via the QEMU monitor; keystrokes are sent with `sendkey`.
"""
import argparse, os, socket, subprocess, sys, time

QEMU = os.environ.get("QEMU", r"C:\Program Files\qemu\qemu-system-x86_64.exe")
OVMF = os.environ.get("OVMF", r"C:\Program Files\qemu\share\edk2-x86_64-code.fd")

ap = argparse.ArgumentParser()
ap.add_argument("disk"); ap.add_argument("outdir")
ap.add_argument("--smp", default="4")
ap.add_argument("--mem", default="4096")
ap.add_argument("--accel", default="tcg")
ap.add_argument("--cpu", default="max")
ap.add_argument("--type", action="append", default=[])
ap.add_argument("--boot-wait", type=float, default=60)
ap.add_argument("--reply-wait", type=float, default=120)
ap.add_argument("--res", default="1280x800")
ap.add_argument("--readonly", action="store_true", help="attach the disk read-only")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
with socket.socket() as _s:  # a free port, so two runs can go side by side
    _s.bind(("127.0.0.1", 0)); port = _s.getsockname()[1]
xres, yres = a.res.split("x")
cmd = [QEMU, "-machine", "q35", "-accel", a.accel, "-cpu", a.cpu, "-smp", a.smp, "-m", a.mem,
       "-drive", f"if=pflash,format=raw,readonly=on,file={OVMF}",
       "-drive", f"file={a.disk},format=raw,if=none,id=d0" + (",readonly=on" if a.readonly else ""),
       "-device", f"VGA,xres={xres},yres={yres}", "-display", "none",
       "-usb", "-device", "qemu-xhci,id=xhci", "-device", "usb-kbd",
       # read-only: a USB stick (NVMe can't be read-only); otherwise NVMe, which loads faster
       # bootindex: boot this disk even if NVVARS on it remembers the other kind of device
       "-device", "usb-storage,bus=xhci.0,drive=d0,bootindex=0" if a.readonly else "nvme,drive=d0,serial=qwen,bootindex=0",
       "-monitor", f"tcp:127.0.0.1:{port},server,nowait",
       "-serial", f"file:{os.path.join(a.outdir, 'serial.log')}", "-no-reboot"]
print(" ".join(cmd))
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
time.sleep(1.5)
s = None
for _ in range(50):
    try: s = socket.create_connection(("127.0.0.1", port)); break
    except OSError: time.sleep(0.2)
if s is None:
    print(p.communicate(timeout=5)[0].decode(errors="replace")); sys.exit(1)
s.settimeout(0.3)

def mon(c):
    s.sendall((c + "\n").encode()); time.sleep(0.15)
    try:
        while s.recv(65536): pass
    except socket.timeout: pass

n = 0
def shot(tag):
    global n; n += 1
    path = os.path.abspath(os.path.join(a.outdir, f"{n:02d}_{tag}.png")).replace("\\", "/")
    mon(f"screendump {path} -f png"); print("screenshot", path)

KEYS = {" ": "spc", "\n": "ret", ".": "dot", ",": "comma", "?": "shift-slash", "!": "shift-1", "'": "apostrophe",
        "/": "slash", "-": "minus", ":": "shift-semicolon", "+": "shift-equal", "=": "equal", "\b": "backspace"}
def keyname(ch):
    if ch in KEYS: return KEYS[ch]
    if ch.isupper(): return "shift-" + ch.lower()
    return ch

t0 = time.time()
while time.time() - t0 < a.boot_wait:
    time.sleep(5); shot("boot")
    if p.poll() is not None: break
for msg in [m.replace("\\b", "\b") for m in a.type]:
    if msg.startswith("@"):  # raw QEMU key names, e.g. "@pgup pgup" (no Enter)
        for k in msg[1:].split(): mon(f"sendkey {k}"); time.sleep(0.3)
        time.sleep(1); shot("keys"); continue
    for ch in msg: mon(f"sendkey {keyname(ch)}"); time.sleep(0.05)
    shot("typed")
    mon("sendkey ret")
    t1 = time.time()
    while time.time() - t1 < a.reply_wait:
        time.sleep(10); shot("reply")
try: mon("quit")
except OSError: pass
try: p.wait(timeout=10)
except subprocess.TimeoutExpired: p.kill()
print(p.stdout.read().decode(errors="replace")[-2000:])
