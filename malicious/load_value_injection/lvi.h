// Cache line size used in gem5 (same as L1_BLOCK_SZ_BYTES)
#define L1_BLOCK_SZ_BYTES 64

// Set spacing between oracle entries to one full cache line (64 bytes)
#define ORACLE_STRIDE_BYTES L1_BLOCK_SZ_BYTES

// Flush cache: no-op for SE mode in gem5 (fence used as placeholder)
static inline void flushCache(uint64_t addr, uint64_t size) {
    asm volatile("fence.i" ::: "memory");
}

// Read cycle counter (used to measure cache access time)
static inline uint64_t rdcycle() {
    uint64_t cycle;
    asm volatile("rdcycle %0" : "=r"(cycle));
    return cycle;
}
