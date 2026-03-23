

#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>

#define MAX 1024

double A[MAX][MAX];
double x[MAX];
double y[MAX];

// Inicializa la matriz A y el vector x con valores aleatorios
void initialize() {
    for (int i = 0; i < MAX; i++) {
        x[i] = (double)rand() / RAND_MAX;
        for (int j = 0; j < MAX; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

// Acceso a A[i][j]: secuencial en memoria → CACHE FRIENDLY
void loop_ij() {
    memset(y, 0, sizeof(y));
    for (int i = 0; i < MAX; i++)
        for (int j = 0; j < MAX; j++)
            y[i] += A[i][j] * x[j];
}

// Acceso a A[i][j] variando i en el bucle interno → CACHE UNFRIENDLY
void loop_ji() {
    memset(y, 0, sizeof(y));
    for (int j = 0; j < MAX; j++)
        for (int i = 0; i < MAX; i++)
            y[i] += A[i][j] * x[j];
}

// Ejecuta un benchmark: corre la función N veces y retorna promedio en ms
double benchmark(void (*func)(), int runs = 5) {
    double total = 0.0;
    for (int r = 0; r < runs; r++) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end   = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<double, std::milli>(end - start).count();
    }
    return total / runs;
}

int main() {
    srand(42);
    initialize();

    std::cout << "==========================================\n";
    std::cout << "  Benchmark C++  |  MAX = " << MAX << "  |  double\n";
    std::cout << "==========================================\n\n";

    double time_ij = benchmark(loop_ij);
    double time_ji = benchmark(loop_ji);
    double ratio   = time_ji / time_ij;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Bucle ij (filas, cache-friendly)  : " << time_ij << " ms\n";
    std::cout << "Bucle ji (columnas, cache-unfriend): " << time_ji << " ms\n";
    std::cout << "\nRatio ji/ij = " << ratio << "x\n";
    std::cout << "  → El bucle ji es ~" << ratio << " veces MÁS LENTO\n\n";

    std::cout << "Explicación:\n";
    std::cout << "  C++ almacena matrices en row-major order.\n";
    std::cout << "  El bucle ij accede a A[i][j] de forma contigua (stride-1).\n";
    std::cout << "  El bucle ji accede a A[i][j] con stride=" << MAX
              << " → muchos cache misses.\n";

    return 0;
}
