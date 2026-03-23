package main

import (
	"fmt"
	"math/rand"
	"strings"
	"time"
)

const MAXN = 1024

// Matrices como arrays planos (row-major) para garantizar contigüidad en memoria
// igual que en C/C++. A[i][j] = data[i*N + j]
type Matrix struct {
	data [MAXN * MAXN]float64
	n    int
}

func (m *Matrix) get(i, j int) float64        { return m.data[i*m.n+j] }
func (m *Matrix) set(i, j int, v float64)     { m.data[i*m.n+j] = v }
func (m *Matrix) add(i, j int, v float64)     { m.data[i*m.n+j] += v }

var A, B, C Matrix

func init_matrices(N int) {
	rng := rand.New(rand.NewSource(42))
	A.n, B.n, C.n = N, N, N
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			A.set(i, j, rng.Float64())
			B.set(i, j, rng.Float64())
		}
	}
}

func zero_C(N int) {
	for i := 0; i < N*N; i++ {
		C.data[i] = 0
	}
}

// ─── Los 6 órdenes de bucle ──────────────────────────────────────

// ijk — clásico. B[k][j] con stride-N en inner loop → medio
func matmul_ijk(N int) {
	zero_C(N)
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			for k := 0; k < N; k++ {
				C.add(i, j, A.get(i, k)*B.get(k, j))
			}
		}
	}
}

// ikj — MEJOR orden. j varía en inner loop: A[i][k] escalar,
//        B[k][j] y C[i][j] ambos stride-1 ✓
func matmul_ikj(N int) {
	zero_C(N)
	for i := 0; i < N; i++ {
		for k := 0; k < N; k++ {
			r := A.get(i, k) // escalar: sólo 1 carga de memoria
			base_b := k * N
			base_c := i * N
			for j := 0; j < N; j++ {
				C.data[base_c+j] += r * B.data[base_b+j] // stride-1 ✓
			}
		}
	}
}

// jik
func matmul_jik(N int) {
	zero_C(N)
	for j := 0; j < N; j++ {
		for i := 0; i < N; i++ {
			for k := 0; k < N; k++ {
				C.add(i, j, A.get(i, k)*B.get(k, j))
			}
		}
	}
}

// jki
func matmul_jki(N int) {
	zero_C(N)
	for j := 0; j < N; j++ {
		for k := 0; k < N; k++ {
			r := B.get(k, j)
			for i := 0; i < N; i++ {
				C.add(i, j, A.get(i, k)*r)
			}
		}
	}
}

// kij — también muy bueno, similar a ikj
func matmul_kij(N int) {
	zero_C(N)
	for k := 0; k < N; k++ {
		for i := 0; i < N; i++ {
			r := A.get(i, k)
			base_b := k * N
			base_c := i * N
			for j := 0; j < N; j++ {
				C.data[base_c+j] += r * B.data[base_b+j]
			}
		}
	}
}

// kji — PEOR. Ambos A y C tienen stride-N en el inner loop
func matmul_kji(N int) {
	zero_C(N)
	for k := 0; k < N; k++ {
		for j := 0; j < N; j++ {
			r := B.get(k, j)
			for i := 0; i < N; i++ {
				C.add(i, j, A.get(i, k)*r)
			}
		}
	}
}

// ─── Benchmark ───────────────────────────────────────────────────
type MatmulFn func(int)

func measure(fn MatmulFn, N, runs int) float64 {
	var total time.Duration
	for r := 0; r < runs; r++ {
		init_matrices(N)
		start := time.Now()
		fn(N)
		total += time.Since(start)
	}
	return float64(total.Milliseconds()) / float64(runs)
}

func main() {
	sizes := []int{64, 128, 256, 512, 1024}

	type Order struct {
		name string
		fn   MatmulFn
	}
	orders := []Order{
		{"ijk", matmul_ijk},
		{"ikj", matmul_ikj},
		{"jik", matmul_jik},
		{"jki", matmul_jki},
		{"kij", matmul_kij},
		{"kji", matmul_kji},
	}

	fmt.Println("=========================================================")
	fmt.Println("  Multiplicación de matrices Go — orden de bucles")
	fmt.Println("=========================================================\n")

	// Encabezado
	fmt.Printf("%-6s", "Orden")
	for _, n := range sizes {
		fmt.Printf("%12s", fmt.Sprintf("N=%d", n))
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 6+len(sizes)*12))

	for _, ord := range orders {
		fmt.Printf("%-6s", ord.name)
		for _, N := range sizes {
			runs := 3
			if N == 1024 && (ord.name == "jki" || ord.name == "kji") {
				runs = 1 // muy lento, 1 corrida basta para demo
			}
			ms := measure(ord.fn, N, runs)
			fmt.Printf("%10.1fms", ms)
		}
		fmt.Println()
	}

	fmt.Println()
	fmt.Println("Leyenda de acceso a caché:")
	fmt.Println("  ikj / kij — stride-1 en j (inner loop)     → MEJOR")
	fmt.Println("  ijk / jik — stride-N en B[k][j]            → MEDIO")
	fmt.Println("  jki / kji — stride-N en A y C inner loop   → PEOR")
	fmt.Printf("\nComplejidad: O(N³) = %d ops para N=1024\n", 1024*1024*1024)
}
