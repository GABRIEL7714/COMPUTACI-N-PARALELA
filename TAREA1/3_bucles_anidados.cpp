#include <iostream>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <vector>
#include <string>

// ─── Tamaños a evaluar ───────────────────────────────────────────
static const int SIZES[] = {64, 128, 256, 512, 1024};
static const int NSIZES  = 5;

// ─── Matrices estáticas máximas (evita malloc en el benchmark) ───
static const int MAXN = 1024;
static double A[MAXN][MAXN], B[MAXN][MAXN], C[MAXN][MAXN];

// ─── Inicialización ──────────────────────────────────────────────
void init(int N) {
    srand(42);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
            B[i][j] = (double)rand() / RAND_MAX;
        }
}

void zero_C(int N) {
    for (int i = 0; i < N; i++)
        memset(C[i], 0, N * sizeof(double));
}

// ─── Los 6 órdenes de bucle ──────────────────────────────────────
// Mismo resultado matemático, distinto patrón de acceso a caché.

// ijk: el clásico "natural". B[k][j] tiene stride-N (columnar) → malo
void matmul_ijk(int N) {
    zero_C(N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

// ikj: el mejor orden. j varía en el inner loop → A[i][k] escalar,
//       B[k][j] y C[i][j] ambos stride-1 ✓
void matmul_ikj(int N) {
    zero_C(N);
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            double r = A[i][k];          // escalar: sólo 1 carga
            for (int j = 0; j < N; j++)
                C[i][j] += r * B[k][j]; // ambas matrices stride-1
        }
}

// jik
void matmul_jik(int N) {
    zero_C(N);
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

// jki
void matmul_jki(int N) {
    zero_C(N);
    for (int j = 0; j < N; j++)
        for (int k = 0; k < N; k++) {
            double r = B[k][j];
            for (int i = 0; i < N; i++)
                C[i][j] += A[i][k] * r;
}
}

// kij
void matmul_kij(int N) {
    zero_C(N);
    for (int k = 0; k < N; k++)
        for (int i = 0; i < N; i++) {
            double r = A[i][k];
            for (int j = 0; j < N; j++)
                C[i][j] += r * B[k][j];
        }
}

// kji
void matmul_kji(int N) {
    zero_C(N);
    for (int k = 0; k < N; k++)
        for (int j = 0; j < N; j++) {
            double r = B[k][j];
            for (int i = 0; i < N; i++)
                C[i][j] += A[i][k] * r;
        }
}

// ─── Benchmark ───────────────────────────────────────────────────
typedef void (*MatmulFn)(int);

double measure(MatmulFn fn, int N, int runs = 3) {
    double total = 0;
    for (int r = 0; r < runs; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn(N);
        auto t1 = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return total / runs;
}

int main() {
    struct Order { const char* name; MatmulFn fn; };
    Order orders[] = {
        {"ijk", matmul_ijk},
        {"ikj", matmul_ikj},
        {"jik", matmul_jik},
        {"jki", matmul_jki},
        {"kij", matmul_kij},
        {"kji", matmul_kji},
    };
    const int NORDERS = 6;

    std::cout << "=========================================================\n";
    std::cout << "  Multiplicación de matrices C++ — orden de bucles\n";
    std::cout << "=========================================================\n\n";

    // Encabezado
    std::cout << std::left << std::setw(6) << "Orden";
    for (int s = 0; s < NSIZES; s++)
        std::cout << std::right << std::setw(12) << ("N=" + std::to_string(SIZES[s]));
    std::cout << "\n" << std::string(6 + NSIZES * 12, '-') << "\n";

    for (int o = 0; o < NORDERS; o++) {
        std::cout << std::left << std::setw(6) << orders[o].name;
        for (int s = 0; s < NSIZES; s++) {
            int N = SIZES[s];
            if (N > 512 && (strcmp(orders[o].name,"kji")==0 || strcmp(orders[o].name,"jki")==0)) {
                // estos órdenes son muy lentos para N=1024 en demo rápido
                init(N);
                double ms = measure(orders[o].fn, N, 1);
                std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(1) << ms << "ms";
            } else {
                init(N);
                double ms = measure(orders[o].fn, N);
                std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(1) << ms << "ms";
            }
        }
        std::cout << "\n";
    }

    std::cout << "\n";
    std::cout << "Leyenda de acceso a caché:\n";
    std::cout << "  ikj / kij — stride-1 en j (inner loop)  → MEJOR\n";
    std::cout << "  ijk / jik — stride-N en B[k][j]         → MEDIO\n";
    std::cout << "  jki / kji — stride-N en A[i][k] y C[i][j] → PEOR\n";
    std::cout << "\nComplejidad: O(N³) = " << (long long)1024*1024*1024 << " ops para N=1024\n";
    return 0;
}
