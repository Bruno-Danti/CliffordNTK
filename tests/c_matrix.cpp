#include <iostream>
#include <Eigen/Dense>
#include <chrono>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

int main() {
    int n;
    cout << "Enter the size of the matrices (n x n): ";
    cin >> n;

    // Generate two random matrices of size n x n
    MatrixXd A = MatrixXd::Random(n, n);
    MatrixXd B = MatrixXd::Random(n, n);

    // Start measuring time
    auto start = high_resolution_clock::now();

    // Multiply the matrices
    MatrixXd C = A * B;

    // Stop measuring time
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    // Output the result (first 5 elements of the product matrix for brevity)
    cout << "Product matrix (first 5x5 block):" << endl;
    cout << C.topLeftCorner(5, 5) << endl;

    // Output time taken
    cout << "Time taken for multiplication: " << duration.count() << " microseconds" << endl;

    return 0;
}
