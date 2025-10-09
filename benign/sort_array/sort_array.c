#include <stdio.h>
#define N 10

int main() {
    int array[N];
    for (int i = 0; i < N; ++i) {
        array[i] = N - i;
    }

    for (int i = 0; i < N - 1; ++i) {
        for (int j = 0; j < N - i - 1; ++j) {
            if (array[j] > array[j + 1]) {
                int tmp = array[j];
                array[j] = array[j + 1];
                array[j + 1] = tmp;
            }
        }
    }

    printf("Sorted first element: %d\n", array[0]);
    return 0;
}
