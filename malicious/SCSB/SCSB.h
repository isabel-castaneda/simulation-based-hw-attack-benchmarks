// Same cache line size as gem5 default L1
#define L1_BLOCK_SZ_BYTES 64

// Simulated cache flush (no-op in SE mode)
static inline void flushCache(uint64_t addr, uint64_t size) {
    asm volatile("fence.i" ::: "memory");
}

// Cycle counter for timing
static inline uint64_t rdcycle() {
    uint64_t cycle;
    asm volatile("rdcycle %0" : "=r"(cycle));
    return cycle;
}
