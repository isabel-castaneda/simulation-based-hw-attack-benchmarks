#include <stdio.h>
#define N 10

int main() {
    unsigned int x = 0xF0F0F0F0;
    unsigned int y = 0x0F0F0F0F;
    unsigned int z = 0;

    for (int i = 0; i < N; ++i) {
        z = (x & y) | (x ^ y);
    }

    printf("Bitwise result: %u\n", z);
    return 0;
}
