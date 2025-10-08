#include <stdio.h>
#include <string.h>
#define N 16
#define SIZE N * N

int main() {
    char src[SIZE], dst[SIZE];
    memset(src, 'A', SIZE);

    for (int i = 0; i < 100; ++i) {
        memcpy(dst, src, SIZE);
    }

    printf("Copy done: %c\n", dst[0]);
    return 0;
}
