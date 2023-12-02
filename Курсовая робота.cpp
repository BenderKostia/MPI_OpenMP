/*
///////////////////////// MPI /////////////////////////
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

using namespace std;

// Функція для множення двох блоків матриць
void multiplyBlocks(const vector<double>& A, const vector<double>& B, vector<double>& C, int blockSize) {
    for (int i = 0; i < blockSize; ++i) {
        for (int j = 0; j < blockSize; ++j) {
            double sum = 0.0;
            for (int k = 0; k < blockSize; ++k) {
                sum += A[i * blockSize + k] * B[k * blockSize + j];
            }
            C[i * blockSize + j] += sum;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 3000; // Визначаємо N як розмір кожної матриці

    // Перевірка, чи кількість процесів є квадратним числом
    double temp = sqrt(static_cast<double>(size));
    int sqrtP = static_cast<int>(round(temp));
    if (sqrtP * sqrtP != size) {
        if (rank == 0) {
            cerr << "The number of processes must be a square number." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int blockSize = N / sqrtP; // Визначення розміру блоку для кожного процесу

    // Ініціалізація блоків матриць A і B випадковими значеннями
    srand(static_cast<unsigned int>(time(NULL)) + rank); // Ініціалізація генератора випадкових чисел
    vector<double> A(blockSize * blockSize);
    vector<double> B(blockSize * blockSize);
    for (int i = 0; i < blockSize * blockSize; ++i) {
        A[i] = rand() % 10; // Випадкові числа від 0 до 9
        B[i] = rand() % 10;
    }
    vector<double> C(blockSize * blockSize, 0.0); // Ініціалізація нулями для матриці результату

    double startTime = MPI_Wtime(); // Запуск вимірювання часу

    // Основний цикл алгоритму Фокса
    for (int stage = 0; stage < sqrtP; ++stage) {
        // Визначення блоку для множення
        int bcastRoot = (rank / sqrtP + stage) % sqrtP;
        if (bcastRoot == rank % sqrtP) {
            multiplyBlocks(A, B, C, blockSize);
        }

        // Ротація блоків матриці B
        int sendRank = (rank / sqrtP) * sqrtP + (rank + 1) % sqrtP;
        int recvRank = (rank / sqrtP) * sqrtP + (rank - 1 + sqrtP) % sqrtP;
        MPI_Sendrecv_replace(B.data(), blockSize * blockSize, MPI_DOUBLE, sendRank, 0, recvRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Збір результатів у матрицю C
    vector<double> finalMatrix(N * N, 0.0);
    MPI_Gather(C.data(), blockSize * blockSize, MPI_DOUBLE, finalMatrix.data(), blockSize * blockSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double endTime = MPI_Wtime(); // Вимірювання часу завершення

    // На процесі з рангом 0 виводимо кінцеву матрицю та час виконання
    if (rank == 0) { */
  /*      cout << "Final Matrix:" << endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                cout << finalMatrix[i * N + j] << " ";
            }
            cout << endl;
        } */ /*
        cout << "Execution Time: " << endTime - startTime << " seconds." << endl;
    }

    MPI_Finalize();
    return 0;
}
*/

///////////////////////// OpenMP /////////////////////////
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

// Функція для ініціалізації матриць випадковими значеннями
void initializeMatrix(vector<vector<double>>& matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = rand() % 10;  // Значення від 0 до 9
        }
    }
}

// Функція для множення двох матриць
void multiplyMatrixBlocks(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C,
    int blockSize,
    int blockRow,
    int blockCol,
    int N) {

    for (int i = 0; i < blockSize; ++i) {
        for (int j = 0; j < blockSize; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[blockRow * blockSize + i][k] * B[k][blockCol * blockSize + j];
            }
            C[blockRow * blockSize + i][blockCol * blockSize + j] += sum;
        }
    }
}

int main() {
    const int N = 3000;  // Повний розмір матриць
    const int sqrtP = omp_get_max_threads(); // Вважаємо, що кількість потоків є квадратним числом
    const int blockSize = N / sqrtP; // Розмір блоку

    srand(static_cast<unsigned int>(time(nullptr))); // Ініціалізація генератора випадкових чисел

    vector<vector<double>> A(N, vector<double>(N));
    vector<vector<double>> B(N, vector<double>(N));
    vector<vector<double>> C(N, vector<double>(N, 0.0));

    initializeMatrix(A, N);
    initializeMatrix(B, N);

    double startTime = omp_get_wtime();

    // Паралельний блочний алгоритм множення матриць за алгоритмом Фокса
#pragma omp parallel shared(A, B, C)
    {
        int thread_id = omp_get_thread_num();
        int blockRow = thread_id / sqrtP;
        int blockCol = thread_id % sqrtP;

        for (int stage = 0; stage < sqrtP; ++stage) {
            int root = (blockRow + stage) % sqrtP;
            multiplyMatrixBlocks(A, B, C, blockSize, blockRow, root, N);
        }
    }

    double endTime = omp_get_wtime();

    cout << "Execution Time: " << endTime - startTime << " seconds." << endl;

    return 0;
}
