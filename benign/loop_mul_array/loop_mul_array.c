#include <stdio.h>

int main() {
    int a[1000];
    for (int i = 0; i < 1000; i++) a[i] = i + 1;

    double prod = 1.0;
    for (int i = 0; i < 1000; i++) {
        prod *= a[i];
    }

    printf("Approximate product: %e\n", prod);
    return 0;
}
