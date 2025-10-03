#define L1_BLOCK_SZ_BYTES 64

static inline void flushCache(uint64_t addr, uint64_t size) {
    // simple no-op implementation for SE mode
    asm volatile("fence.i" ::: "memory");
}

static inline uint64_t rdcycle() {
    uint64_t cycle;
    asm volatile("rdcycle %0" : "=r" (cycle));
    return cycle;
}
