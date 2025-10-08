#include <stdio.h>
#define N 10

int main() {
    float x = 1.5f;
    for (int i = 0; i < N; i++) {
        x = x * 1.0001f - 0.0001f;
    }
    printf("Result: %.4f\n", x);
    return 0;
}
