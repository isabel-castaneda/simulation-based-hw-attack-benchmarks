#include <stdio.h>
#define N 10

int main() {
    int i, sum = 0;
    for (i = 0; i < N; ++i) {
        if (i % 2 == 0) {
            sum += i;
        } else {
            sum -= i;
        }
    }
    printf("Branch sum: %d\n", sum);
    return 0;
}
