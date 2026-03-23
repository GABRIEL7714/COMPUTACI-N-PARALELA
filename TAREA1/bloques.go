

package main

import (
	"fmt"
	"math/rand"
	"strings"
	"time"
)

const MAXN = 1024

// Array plano row-major para máxima contigüidad (igual que C/C++)
// A[i][j] = data[i*N + j]
type FlatMatrix [MAXN * MAXN]float64

var A, B, C FlatMatrix

func initMats(N int) {
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			A[i*N+j] = rng.Float64()
			B[i*N+j] = rng.Float64()
			C[i*N+j] = 0
		}
	}
}

func zeroC(N int) {
	for i := 0; i < N*N; i++ {
		C[i] = 0
	}
}

// ══════════════════════════════════════════════════════════════
//  VERSIÓN 1 — Clásica ijk (3 bucles, referencia lenta)
//  B[k][j] accede con stride-N en el inner loop
// ══════════════════════════════════════════════════════════════
func matmulClassicIJK(N int) {
	zeroC(N)
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			for k := 0; k < N; k++ {
				C[i*N+j] += A[i*N+k] * B[k*N+j]
			}
		}
	}
}

// ══════════════════════════════════════════════════════════════
//  VERSIÓN 2 — Clásica ikj (3 bucles, mejor orden de referencia)
//  j varía en el inner loop: B[k][j] y C[i][j] stride-1 ✓
// ══════════════════════════════════════════════════════════════
func matmulClassicIKJ(N int) {
	zeroC(N)
	for i := 0; i < N; i++ {
		baseC := i * N
		for k := 0; k < N; k++ {
			r := A[i*N+k] // escalar: 1 carga
			baseB := k * N
			for j := 0; j < N; j++ {
				C[baseC+j] += r * B[baseB+j] // ambas stride-1 ✓
			}
		}
	}
}

// ══════════════════════════════════════════════════════════════
//  VERSIÓN 3 — Por bloques / Cache Blocking (6 bucles)
//
//  BLOCK: tamaño del tile cuadrado.
//  Memoria activa: 3 × BLOCK² × 8 bytes
//    BLOCK=16 → 6 KB  → cabe holgado en L1 (≈32 KB)
//    BLOCK=32 → 24 KB → cabe justo en L1
//    BLOCK=64 → 96 KB → cabe en L2 (≈256 KB)
//
//  Maneja el caso en que N no sea múltiplo de BLOCK
//  usando min() en los límites de los bucles internos.
// ══════════════════════════════════════════════════════════════
func matmulBlocked(N, BLOCK int) {
	zeroC(N)
	// Bucles EXTERNOS: iteran sobre bloques
	for ii := 0; ii < N; ii += BLOCK {
		for kk := 0; kk < N; kk += BLOCK {
			for jj := 0; jj < N; jj += BLOCK {
				// Límites del bloque actual (edge case: N % BLOCK != 0)
				iEnd := min(ii+BLOCK, N)
				kEnd := min(kk+BLOCK, N)
				jEnd := min(jj+BLOCK, N)
				// Bucles INTERNOS: mini-multiplicación ikj dentro del bloque
				for i := ii; i < iEnd; i++ {
					baseC := i * N
					for k := kk; k < kEnd; k++ {
						r := A[i*N+k] // escalar: 1 carga de memoria
						baseB := k * N
						for j := jj; j < jEnd; j++ {
							C[baseC+j] += r * B[baseB+j] // stride-1 ✓
						}
					}
				}
			}
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ── Benchmark ──────────────────────────────────────────────────
func measureFn(fn func(int), N, runs int) float64 {
	var total time.Duration
	for r := 0; r < runs; r++ {
		initMats(N)
		start := time.Now()
		fn(N)
		total += time.Since(start)
	}
	return float64(total.Milliseconds()) / float64(runs)
}

func measureBlocked(N, BLOCK, runs int) float64 {
	var total time.Duration
	for r := 0; r < runs; r++ {
		initMats(N)
		start := time.Now()
		matmulBlocked(N, BLOCK)
		total += time.Since(start)
	}
	return float64(total.Milliseconds()) / float64(runs)
}

func printRow(label string, ms, refMs float64) {
	speedup := refMs / ms
	fmt.Printf("%-22s %8.1f ms   %6.2f×\n", label, ms, speedup)
}

func main() {
	sizes  := []int{64, 128, 256, 512, 1024}
	blocks := []int{16, 32, 64}
	runs   := 3

	fmt.Println("=================================================================")
	fmt.Println("  Multiplicación de matrices Go — Clásica vs Por Bloques")
	fmt.Println("=================================================================")
	fmt.Println("  • Clásica ijk  : 3 bucles, stride-N en B → referencia lenta")
	fmt.Println("  • Clásica ikj  : 3 bucles, stride-1 → mejor orden")
	fmt.Println("  • Bloques B=16 : 6 bucles, tile 16×16 (≈6 KB → L1)")
	fmt.Println("  • Bloques B=32 : 6 bucles, tile 32×32 (≈24 KB → L1)")
	fmt.Println("  • Bloques B=64 : 6 bucles, tile 64×64 (≈96 KB → L2)")
	fmt.Println("  Speedup medido respecto a clásica ijk")
	fmt.Println("=================================================================\n")

	for _, N := range sizes {
		ops := float64(N) * float64(N) * float64(N) * 2
		fmt.Printf("─── N = %d  (%.1f Mops) ─────────────────────────────\n",
			N, ops/1e6)
		fmt.Printf("%-22s %10s   %8s\n", "Método", "Tiempo", "Speedup")
		fmt.Println(strings.Repeat("-", 44))

		refIJK := measureFn(matmulClassicIJK, N, runs)
		printRow("Clásica ijk", refIJK, refIJK)

		refIKJ := measureFn(matmulClassicIKJ, N, runs)
		printRow("Clásica ikj", refIKJ, refIJK)

		for _, BS := range blocks {
			if BS > N {
				continue
			}
			r := runs
			if N == 1024 {
				r = 2 // N=1024 tarda más; 2 corridas bastan
			}
			t := measureBlocked(N, BS, r)
			printRow(fmt.Sprintf("Bloques B=%d", BS), t, refIJK)
		}
		fmt.Println()
	}

	// Tabla de uso de caché
	fmt.Println("=================================================================")
	fmt.Println("  Memoria activa por bloque (3 × BLOCK² × 8 bytes)")
	fmt.Println("=================================================================")
	fmt.Printf("%-12s %-16s %s\n", "BLOCK", "Mem. bloque", "Cabe en")
	fmt.Println(strings.Repeat("-", 42))
	bsizes := []struct {
		bs    int
		kb    int
		level string
	}{
		{16, 6,  "L1 caché (≈32 KB)"},
		{32, 24, "L1 caché (≈32 KB) — justo"},
		{64, 96, "L2 caché (≈256 KB)"},
	}
	for _, b := range bsizes {
		fmt.Printf("%-12d %-16s %s\n", b.bs, fmt.Sprintf("%d KB", b.kb), b.level)
	}
}
