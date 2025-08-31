#include <iostream>
#include <cblas.h>
#include <chrono>
#include <vector>

using namespace std;
using namespace std::chrono;

int main() {
    int n;
    cout << "Enter the size of the matrices (n x n): ";
    cin >> n;

    // Create matrices A, B, and C
    vector<double> A(n * n), B(n * n), C(n * n);

    // Fill A and B with random values
    for (int i = 0; i < n * n; ++i) {
        A[i] = (rand() % 100) / 100.0; // Random numbers between 0 and 1
        B[i] = (rand() % 100) / 100.0;
    }

    // Start measuring time
    auto start = high_resolution_clock::now();

    // Matrix multiplication C = A * B using BLAS (cblas_dgemm)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                n, n, n, 
                1.0, 
                A.data(), n, 
                B.data(), n, 
                0.0, 
                C.data(), n);

    // Stop measuring time
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    // Output the result (first 5 elements of the product matrix for brevity)
    cout << "Product matrix (first 5x5 block):" << endl;
    for (int i = 0; i < min(5, n); ++i) {
        for (int j = 0; j < min(5, n); ++j) {
            cout << C[i * n + j] << " ";
        }
        cout << endl;
    }

    // Output time taken
    cout << "Time taken for multiplication: " << duration.count() << " microseconds" << endl;

    return 0;
}
