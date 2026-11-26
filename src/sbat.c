// SBAT metadata (https://github.com/rhboot/shim/blob/main/SBAT.md). shim 15.3 and newer refuse
// to start an image without a .sbat section, so this is what lets metallm boot behind shim with
// Secure Boot on (see tools/secureboot.py). Bump the generation number in the second line only
// if a released build ever has to be revoked.
#define SBAT_CSV \
    "sbat,1,SBAT Version,sbat,1,https://github.com/rhboot/shim/blob/main/SBAT.md\n" \
    "metallm,1,metallm,metallm,1,https://github.com/hectoragofi/metallm\n"

// no trailing NUL: shim parses the whole section as CSV
__attribute__((section(".sbat"), used)) const char sbat_data[sizeof SBAT_CSV - 1] = SBAT_CSV;
