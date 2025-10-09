#include <stdio.h>
#include <string.h>
#define N 16
#define SIZE N * N
#define STRIDE 1

char buffer[SIZE];

int main() {
    memset(buffer, 1, SIZE);
    volatile int sum = 0;
    for (int i = 0; i < SIZE; i += STRIDE) {
        sum += buffer[i];
    }

    printf("Cache sum: %d\n", sum);
    return 0;
}
