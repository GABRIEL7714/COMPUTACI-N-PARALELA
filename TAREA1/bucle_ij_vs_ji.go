package main

import (
	"fmt"
	"math/rand"
	"time"
)

const MAX = 1024

// Go usa slices 2D (slice de slices) o un slice plano con indexado manual.
// Usamos un slice plano para garantizar contigüidad en memoria (como C/C++).
type Matrix [MAX * MAX]float64

var (
	A Matrix
	x [MAX]float64
	y [MAX]float64
)

// Acceso a la matriz: A[i][j] → A[i*MAX + j]  (row-major, igual que C)
func matGet(m *Matrix, i, j int) float64 { return m[i*MAX+j] }
func matSet(m *Matrix, i, j int, v float64) { m[i*MAX+j] = v }

// Inicializa A y x con valores aleatorios
func initialize() {
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < MAX; i++ {
		x[i] = rng.Float64()
		for j := 0; j < MAX; j++ {
			matSet(&A, i, j, rng.Float64())
		}
	}
}

// Acceso contiguo en memoria → CACHE FRIENDLY
func loopIJ() {
	for i := range y {
		y[i] = 0
	}
	for i := 0; i < MAX; i++ {
		for j := 0; j < MAX; j++ {
			y[i] += matGet(&A, i, j) * x[j]
		}
	}
}


// Acceso con stride=MAX → CACHE UNFRIENDLY
func loopJI() {
	for i := range y {
		y[i] = 0
	}
	for j := 0; j < MAX; j++ {
		for i := 0; i < MAX; i++ {
			y[i] += matGet(&A, i, j) * x[j]
		}
	}
}

// Ejecuta la función 'runs' veces y retorna el tiempo promedio en ms
func benchmark(fn func(), runs int) float64 {
	var total time.Duration
	for r := 0; r < runs; r++ {
		start := time.Now()
		fn()
		total += time.Since(start)
	}
	return float64(total.Milliseconds()) / float64(runs)
}

func main() {
	initialize()

	runs := 5
	fmt.Println("==========================================")
	fmt.Printf("  Benchmark Go   |  MAX = %d  |  float64\n", MAX)
	fmt.Println("==========================================\n")

	timeIJ := benchmark(loopIJ, runs)
	timeJI := benchmark(loopJI, runs)
	ratio  := timeJI / timeIJ

	fmt.Printf("Bucle ij (filas, cache-friendly)  : %.3f ms\n", timeIJ)
	fmt.Printf("Bucle ji (columnas, cache-unfriend): %.3f ms\n", timeJI)
	fmt.Printf("\nRatio ji/ij = %.2fx\n", ratio)
	fmt.Printf("  → El bucle ji es ~%.2f veces MÁS LENTO\n\n", ratio)

	fmt.Println("Explicación:")
	fmt.Println("  Go (como C++) almacena arrays en row-major order.")
	fmt.Printf("  El bucle ij accede a A[i*MAX+j] de forma contigua (stride-1).\n")
	fmt.Printf("  El bucle ji accede a A[i*MAX+j] con stride=%d → muchos cache misses.\n", MAX)
}
