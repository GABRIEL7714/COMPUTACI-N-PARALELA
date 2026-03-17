#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include <fstream>
using namespace std;

int main() {
    vector<int> dimensiones = { 100, 1000, 10000 };
    ofstream archivo_salida("resultados_rendimiento.txt");
    if (!archivo_salida.is_open()) {
        cerr << "Error al abrir el archivo resultados_rendimiento.txt" << endl;
        return EXIT_FAILURE;
    }

    archivo_salida << "Dimension,Bucle1 (ns),Bucle2 (ns)" << endl;

    cout << "Iniciando pruebas de rendimiento..." << endl;

    for (int N : dimensiones) {
        cout << "Procesando matriz de tamaño: " << N << "x" << N << endl;

        double** matriz = new double* [N];
        if (N > 0) {
            matriz[0] = new double[N * N];
            for (int fila = 1; fila < N; fila++) {
                matriz[fila] = matriz[fila - 1] + N;
            }
            memset(matriz[0], 0.0, N * N * sizeof(double));
        }

        double* vec_entrada = new double[N];
        double* vec_salida = new double[N];

        std::random_device dispositivo;
        std::default_random_engine motor(dispositivo());
        std::uniform_real_distribution<double> distribucion(0.0, 10.0);

        for (int fila = 0; fila < N; fila++) {
            vec_entrada[fila] = distribucion(motor);
            vec_salida[fila] = 0.0;
        }

        for (int fila = 0; fila < N; fila++) {
            for (int col = 0; col < N; col++) {
                matriz[fila][col] = distribucion(motor);
            }
        }

        cout << "  Datos inicializados." << endl;

        auto t_inicio = chrono::steady_clock::now();
        for (int fila = 0; fila < N; fila++) {
            for (int col = 0; col < N; col++) {
                vec_salida[fila] += matriz[fila][col] * vec_entrada[col];
            }
        }
        auto t_fin = chrono::steady_clock::now();
        auto duracion_bucle1 = chrono::duration_cast<chrono::nanoseconds>(t_fin - t_inicio).count();

        cout << "  El primer bucle duró: " << duracion_bucle1 << " ns" << endl;

        for (int fila = 0; fila < N; fila++) {
            vec_salida[fila] = 0.0;
        }

        t_inicio = chrono::steady_clock::now();
        for (int col = 0; col < N; col++) {
            for (int fila = 0; fila < N; fila++) {
                vec_salida[fila] += matriz[fila][col] * vec_entrada[col];
            }
        }
        t_fin = chrono::steady_clock::now();
        auto duracion_bucle2 = chrono::duration_cast<chrono::nanoseconds>(t_fin - t_inicio).count();

        cout << "  El segundo bucle duró: " << duracion_bucle2 << " ns" << endl;

        archivo_salida << N << "," << duracion_bucle1 << "," << duracion_bucle2 << endl;

        if (N > 0) {
            delete[] matriz[0];
        }
        delete[] matriz;
        delete[] vec_entrada;
        delete[] vec_salida;

        cout << "------------------------------------" << endl;
    }

    archivo_salida.close();
    cout << "Resultados guardados en resultados_rendimiento.txt" << endl;
    return 0;
}
