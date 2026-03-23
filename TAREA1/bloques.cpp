

#include <iostream>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <string>

// ── Tamaños a evaluar ──────────────────────────────────────────
static const int SIZES[]  = {64, 128, 256, 512, 1024};
static const int NSIZES   = 5;
static const int RUNS     = 3;   // repeticiones por medición

// ── Tamaños de bloque a comparar ──────────────────────────────
// El óptimo depende del procesador; típicamente 16–64 para L1.
static const int BLOCKS[] = {16, 32, 64};
static const int NBLOCKS  = 3;

// ── Matrices estáticas máximas ─────────────────────────────────
static const int MAXN = 1024;
static double A[MAXN][MAXN], B[MAXN][MAXN], C[MAXN][MAXN];

// ── Inicialización ─────────────────────────────────────────────
void init(int N) {
    srand(42);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
            B[i][j] = (double)rand() / RAND_MAX;
            C[i][j] = 0.0;
        }
}

void zero_C(int N) {
    for (int i = 0; i < N; i++)
        memset(C[i], 0, N * sizeof(double));
}

// ══════════════════════════════════════════════════════════════
//  VERSIÓN 1 — Clásica ijk (3 bucles, referencia)
//  B[k][j] tiene stride-N en el bucle interno → cache-unfriendly
// ══════════════════════════════════════════════════════════════
void matmul_classic_ijk(int N) {
    zero_C(N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

// ══════════════════════════════════════════════════════════════
//  VERSIÓN 2 — Clásica ikj (3 bucles, mejor orden de referencia)
//  j varía en el bucle interno → B[k][j] y C[i][j] stride-1 ✓
// ══════════════════════════════════════════════════════════════
void matmul_classic_ikj(int N) {
    zero_C(N);
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            double r = A[i][k];
            for (int j = 0; j < N; j++)
                C[i][j] += r * B[k][j];
        }
}

// ══════════════════════════════════════════════════════════════
//  VERSIÓN 3 — Por bloques / Cache Blocking (6 bucles)
//  BLOCK: tamaño del tile (cuadrado BLOCK×BLOCK)
//
//  Memoria activa por iteración del bloque:
//    bloque A: BLOCK*BLOCK * 8 bytes
//    bloque B: BLOCK*BLOCK * 8 bytes
//    bloque C: BLOCK*BLOCK * 8 bytes
//    Total   : 3 * BLOCK² * 8 bytes
//  Para BLOCK=32: 3 * 1024 * 8 = 24 KB  → cabe en L1 (32 KB típico)
//  Para BLOCK=64: 3 * 4096 * 8 = 96 KB  → cabe en L2 (256 KB típico)
// ══════════════════════════════════════════════════════════════
void matmul_blocked(int N, int BLOCK) {
    zero_C(N);
    // Bucles EXTERNOS: recorren en pasos de BLOCK
    for (int ii = 0; ii < N; ii += BLOCK)
      for (int kk = 0; kk < N; kk += BLOCK)
        for (int jj = 0; jj < N; jj += BLOCK) {
            // Límites del bloque actual (maneja el caso N % BLOCK != 0)
            int i_end = std::min(ii + BLOCK, N);
            int k_end = std::min(kk + BLOCK, N);
            int j_end = std::min(jj + BLOCK, N);
            // Bucles INTERNOS: mini-multiplicación dentro del bloque
            // Orden ikj para máxima localidad dentro del bloque
            for (int i = ii; i < i_end; i++)
              for (int k = kk; k < k_end; k++) {
                  double r = A[i][k];       // escalar: 1 carga
                  for (int j = jj; j < j_end; j++)
                      C[i][j] += r * B[k][j]; // stride-1 ✓
              }
        }
}

// ── Benchmark ──────────────────────────────────────────────────
using Ms = std::chrono::duration<double, std::milli>;

template<typename Fn>
double measure(Fn fn, int N, int runs = RUNS) {
    double total = 0;
    for (int r = 0; r < runs; r++) {
        init(N);
        auto t0 = std::chrono::high_resolution_clock::now();
        fn(N);
        auto t1 = std::chrono::high_resolution_clock::now();
        total += Ms(t1 - t0).count();
    }
    return total / runs;
}

double measure_blocked(int N, int BLOCK, int runs = RUNS) {
    double total = 0;
    for (int r = 0; r < runs; r++) {
        init(N);
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_blocked(N, BLOCK);
        auto t1 = std::chrono::high_resolution_clock::now();
        total += Ms(t1 - t0).count();
    }
    return total / runs;
}

// ── Imprime una fila con tiempo y speedup ──────────────────────
void print_row(const std::string& label, double ms, double ref_ms) {
    std::cout << std::left  << std::setw(22) << label
              << std::right << std::setw(10) << std::fixed
              << std::setprecision(1) << ms << " ms"
              << std::setw(10) << std::setprecision(2)
              << (ref_ms / ms) << "×\n";
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  Multiplicación de matrices C++ — Clásica vs Por Bloques\n";
    std::cout << "=================================================================\n";
    std::cout << "  • Clásica ijk : 3 bucles, stride-N en B → referencia lenta\n";
    std::cout << "  • Clásica ikj : 3 bucles, stride-1 → mejor orden\n";
    std::cout << "  • Bloques B=16: 6 bucles, tile 16×16 (≈6 KB → L1)\n";
    std::cout << "  • Bloques B=32: 6 bucles, tile 32×32 (≈24 KB → L1)\n";
    std::cout << "  • Bloques B=64: 6 bucles, tile 64×64 (≈96 KB → L2)\n";
    std::cout << "  Speedup medido respecto a clásica ijk\n";
    std::cout << "=================================================================\n\n";

    for (int s = 0; s < NSIZES; s++) {
        int N = SIZES[s];
        long long ops = (long long)N * N * N * 2; // 2 flops por fma

        std::cout << "─── N = " << N << "  ("
                  << std::fixed << std::setprecision(1)
                  << ops / 1e6 << " Mops) ─────────────────────────────\n";
        std::cout << std::left  << std::setw(22) << "Método"
                  << std::right << std::setw(12) << "Tiempo"
                  << std::setw(10) << "Speedup\n";
        std::cout << std::string(44, '-') << "\n";

        double ref_ijk = measure([](int n){ matmul_classic_ijk(n); }, N);
        print_row("Clásica ijk",  ref_ijk, ref_ijk);

        double ref_ikj = measure([](int n){ matmul_classic_ikj(n); }, N);
        print_row("Clásica ikj",  ref_ikj, ref_ijk);

        for (int b = 0; b < NBLOCKS; b++) {
            int BS = BLOCKS[b];
            if (BS > N) continue;  // bloque mayor que la matriz: sin sentido
            double t = measure_blocked(N, BS);
            print_row("Bloques B=" + std::to_string(BS), t, ref_ijk);
        }
        std::cout << "\n";
    }

    // ── Tabla de uso de caché por tamaño de bloque ────────────
    std::cout << "=================================================================\n";
    std::cout << "  Memoria activa por bloque (3×BLOCK²×8 bytes)\n";
    std::cout << "=================================================================\n";
    std::cout << std::left << std::setw(12) << "BLOCK"
              << std::setw(16) << "Mem. bloque"
              << "Cabe en\n";
    std::cout << std::string(40, '-') << "\n";
    for (int b = 0; b < NBLOCKS; b++) {
        int BS = BLOCKS[b];
        int kb = 3 * BS * BS * 8 / 1024;
        std::string level = (kb <= 32) ? "L1 caché (≈32 KB)" :
                            (kb <= 256) ? "L2 caché (≈256 KB)" : "L3 caché";
        std::cout << std::left << std::setw(12) << BS
                  << std::setw(16) << (std::to_string(kb) + " KB")
                  << level << "\n";
    }
    std::cout << "\n";
    return 0;
}
