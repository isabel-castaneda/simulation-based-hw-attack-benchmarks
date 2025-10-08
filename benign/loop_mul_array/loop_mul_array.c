#include <stdio.h>
#define N 10

int main() {
    int a[N];
    for (int i = 0; i < N; i++) a[i] = i + 1;

    double prod = 1.0;
    for (int i = 0; i < N; i++) {
        prod *= a[i];
    }

    printf("Approximate product: %e\n", prod);
    return 0;
}
