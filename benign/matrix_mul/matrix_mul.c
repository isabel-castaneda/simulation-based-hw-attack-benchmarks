#include <stdio.h>
#define N 8

int A[N][N], B[N][N], C[N][N];

int main() {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            A[i][j] = 1;
            B[i][j] = 1;
        }

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                C[i][j] += A[i][k] * B[k][j];

    printf("Matrix[0][0] = %d\n", C[0][0]);
    return 0;
}
