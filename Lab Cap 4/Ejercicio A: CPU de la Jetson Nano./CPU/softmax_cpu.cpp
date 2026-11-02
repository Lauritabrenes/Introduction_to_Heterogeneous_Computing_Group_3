#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <random>
#include <cstdint>

// ======================================================
// SOFTMAX FP32
// ======================================================

std::vector<float> softmax_fp32(const std::vector<float>& entrada) {

    float maximo = *std::max_element(entrada.begin(), entrada.end());

    std::vector<float> salida(entrada.size());
    float suma = 0.0f;

    for (size_t i = 0; i < entrada.size(); i++) {
        salida[i] = std::exp(entrada[i] - maximo);
        suma += salida[i];
    }

    for (size_t i = 0; i < salida.size(); i++) {
        salida[i] /= suma;
    }

    return salida;
}

// ======================================================
// SOFTMAX FP16
// ======================================================

std::vector<_Float16> softmax_fp16(const std::vector<_Float16>& entrada) {

    // Buscar el maximo utilizando FP32 para el calculo
    float maximo = static_cast<float>(entrada[0]);

    for (size_t i = 1; i < entrada.size(); i++) {
        float valor = static_cast<float>(entrada[i]);

        if (valor > maximo) {
            maximo = valor;
        }
    }

    // Calcular exponenciales y suma utilizando FP32
    std::vector<float> exponenciales(entrada.size());
    float suma = 0.0f;

    for (size_t i = 0; i < entrada.size(); i++) {
        float valor = static_cast<float>(entrada[i]);

        exponenciales[i] = std::exp(valor - maximo);
        suma += exponenciales[i];
    }

    // Normalizar y convertir el resultado a FP16
    std::vector<_Float16> salida(entrada.size());

    for (size_t i = 0; i < entrada.size(); i++) {
        salida[i] = static_cast<_Float16>(exponenciales[i] / suma);
    }

    return salida;
}


// ======================================================
// SOFTMAX FP64 - REFERENCIA DE EXACTITUD
// ======================================================

std::vector<double> softmax_fp64(const std::vector<double>& entrada){

    double maximo = *std::max_element(entrada.begin(), entrada.end());

    std::vector<double> salida(entrada.size());
    double suma = 0.0;

    for (size_t i = 0; i < entrada.size(); i++) {
        salida[i] = std::exp(entrada[i] - maximo);
        suma += salida[i];
    }

    for (size_t i = 0; i < salida.size(); i++) {
        salida[i] /= suma;
    }

    return salida;
}

// ======================================================
// GENERACION DE VECTORES PARA LAS PRUEBAS
// ======================================================

std::vector<double> generar_vector(int n) {

    // Semilla fija para reproducibilidad
    std::mt19937 generador(12345);

    // Valores entre -4 y 4
    std::uniform_real_distribution<double> distribucion(-4.0, 4.0);

    std::vector<double> entrada(n);

    for (int i = 0; i < n; i++) {
        entrada[i] = distribucion(generador);
    }

    return entrada;
}

// ======================================================
// PRUEBAS CON DIFERENTES TAMANOS DE VECTORES
// ======================================================

void probar_tamanos() {

    std::vector<int> tamanos = {10, 100, 500, 1000};

    std::cout << "\n===== PRUEBAS CON DIFERENTES TAMANOS =====\n";

    for (int n : tamanos) {

        // Generar los mismos datos de entrada
        std::vector<double> entrada = generar_vector(n);

        // Preparar los formatos FP32 y FP16
        std::vector<float> entrada32(n);
        std::vector<_Float16> entrada16(n);

        for (int i = 0; i < n; i++) {
            entrada32[i] = static_cast<float>(entrada[i]);
            entrada16[i] = static_cast<_Float16>(entrada[i]);
        }

        // Calcular Softmax
        std::vector<double> referencia = softmax_fp64(entrada);
        std::vector<float> salida32 = softmax_fp32(entrada32);
        std::vector<_Float16> salida16 = softmax_fp16(entrada16);

        // Inicializar resultados
    
        double suma32 = 0.0;
        double suma16 = 0.0;

        double error32 = 0.0;
        double error16 = 0.0;

        double error_relativo32 = 0.0;
        double error_relativo16 = 0.0;

        // ======================================================
        // CALCULAR SUMAS Y ERRORES
        // ======================================================

        for (int i = 0; i < n; i++) {

            double valor32 = static_cast<double>(salida32[i]);
            double valor16 = static_cast<double>(salida16[i]);

            // Calcular las sumas
            suma32 += valor32;
            suma16 += valor16;

            // Calcular errores absolutos
            double absoluto32 = std::abs(referencia[i] - valor32);
            double absoluto16 = std::abs(referencia[i] - valor16);

            error32 = std::max(error32, absoluto32);
            error16 = std::max(error16, absoluto16);

            // Calcular errores relativos respecto a FP64
            if (referencia[i] > 0.0) {

                double relativo32 = absoluto32 / referencia[i];
                double relativo16 = absoluto16 / referencia[i];

                error_relativo32 = std::max(error_relativo32, relativo32);
                error_relativo16 = std::max(error_relativo16, relativo16);
            }
        }

        // ======================================================
        // MOSTRAR RESULTADOS
        // ======================================================

        std::cout << "\nN = " << n << "\n";

        std::cout << std::fixed << std::setprecision(6);

        std::cout << "Suma FP32: "
                  << suma32 << "\n";

        std::cout << "Suma FP16: "
                  << suma16 << "\n";

        // Mostrar errores absolutos
        std::cout << std::scientific << std::setprecision(8);

        std::cout << "Error absoluto maximo FP32: "
                  << error32 << "\n";

        std::cout << "Error absoluto maximo FP16: "
                  << error16 << "\n";

        // Mostrar errores relativos
        std::cout << "Error relativo maximo FP32: "
                  << error_relativo32 << "\n";

        std::cout << "Error relativo maximo FP16: "
                  << error_relativo16 << "\n";

        // Restaurar formato de salida
        std::cout << std::fixed << std::setprecision(6);

    } // Cierra el ciclo de los tamanos

} // Cierra la funcion probar_tamanos()

// ======================================================
// MEDICION DE TIEMPOS DE SOFTMAX
// ======================================================

void medir_tiempos() {

    std::vector<int> tamanos = {10, 100, 500, 1000};

    const int repeticiones = 10000;

    std::cout << "\n===== MEDICION DE TIEMPOS CPU =====\n";

    for (int n : tamanos) {

        std::vector<double> entrada = generar_vector(n);

        std::vector<float> entrada32(n);
        std::vector<_Float16> entrada16(n);

        for (int i = 0; i < n; i++) {
            entrada32[i] = static_cast<float>(entrada[i]);
            entrada16[i] = static_cast<_Float16>(entrada[i]);
        }

        // Calentamiento
        auto calentamiento32 = softmax_fp32(entrada32);
        auto calentamiento16 = softmax_fp16(entrada16);

        // Evitar variables sin utilizar
        (void)calentamiento32;
        (void)calentamiento16;

        // Medir FP32
        auto inicio32 = std::chrono::steady_clock::now();

        for (int r = 0; r < repeticiones; r++) {
            auto resultado = softmax_fp32(entrada32);
            (void)resultado;
        }

        auto fin32 = std::chrono::steady_clock::now();

        // Medir FP16
        auto inicio16 = std::chrono::steady_clock::now();

        for (int r = 0; r < repeticiones; r++) {
            auto resultado = softmax_fp16(entrada16);
            (void)resultado;
        }

        auto fin16 = std::chrono::steady_clock::now();

        // Duracion total en microsegundos
        double tiempo32 =
            std::chrono::duration<double, std::micro>(
                fin32 - inicio32
            ).count();

        double tiempo16 =
            std::chrono::duration<double, std::micro>(
                fin16 - inicio16
            ).count();

        // Tiempo promedio por llamada
        tiempo32 /= repeticiones;
        tiempo16 /= repeticiones;

        std::cout << std::fixed << std::setprecision(6);

        std::cout << "\nN = " << n << "\n";
        std::cout << "Tiempo promedio FP32 (us): "
                  << tiempo32 << "\n";
        std::cout << "Tiempo promedio FP16 (us): "
                  << tiempo16 << "\n";
    }
}


// ======================================================
// PROGRAMA PRINCIPAL
// ======================================================

int main() {

    // Vector inicial para comprobar ambas precisiones
    std::vector<float> entrada_fp32 = {1.0f, 2.0f, 3.0f, 4.0f};

    std::vector<_Float16> entrada_fp16 = {
        static_cast<_Float16>(1.0f),
        static_cast<_Float16>(2.0f),
        static_cast<_Float16>(3.0f),
        static_cast<_Float16>(4.0f)
    };

    // Ejecutar las dos versiones
    std::vector<float> resultado_fp32 = softmax_fp32(entrada_fp32);
    std::vector<_Float16> resultado_fp16 = softmax_fp16(entrada_fp16);

    std::cout << std::fixed << std::setprecision(6);

    // Mostrar resultados FP32
    std::cout << "===== SOFTMAX FP32 =====\n";

    float suma_fp32 = 0.0f;

    for (float valor : resultado_fp32) {
        std::cout << valor << "\n";
        suma_fp32 += valor;
    }

    std::cout << "Suma FP32: " << suma_fp32 << "\n\n";

    // Mostrar resultados FP16
    std::cout << "===== SOFTMAX FP16 =====\n";

    float suma_fp16 = 0.0f;

    for (_Float16 valor : resultado_fp16) {
        float convertido = static_cast<float>(valor);

        std::cout << convertido << "\n";
        suma_fp16 += convertido;
    }

std::cout << "Suma FP16: " << suma_fp16 << "\n\n";

// ======================================================
// COMPARACION CONTRA FP64
// ======================================================

// Utilizar los mismos valores de entrada
std::vector<double> entrada_fp64 = {1.0, 2.0, 3.0, 4.0};

// Calcular referencia
std::vector<double> resultado_fp64 = softmax_fp64(entrada_fp64);

std::cout << "===== COMPARACION CONTRA FP64 =====\n";

std::cout << "Indice\tFP64\t\tFP32\t\tFP16\t\tError FP32\tError FP16\n";

double error_max_fp32 = 0.0;
double error_max_fp16 = 0.0;

for (size_t i = 0; i < resultado_fp64.size(); i++) {

    double referencia = resultado_fp64[i];

    double valor_fp32 = static_cast<double>(resultado_fp32[i]);
    double valor_fp16 = static_cast<double>(resultado_fp16[i]);

    // Calcular errores absolutos
    double error_fp32 = std::abs(referencia - valor_fp32);
    double error_fp16 = std::abs(referencia - valor_fp16);

    // Actualizar errores maximos
    error_max_fp32 = std::max(error_max_fp32, error_fp32);
    error_max_fp16 = std::max(error_max_fp16, error_fp16);

    std::cout << i << "\t"
              << referencia << "\t"
              << valor_fp32 << "\t"
              << valor_fp16 << "\t"
              << error_fp32 << "\t"
              << error_fp16 << "\n";
}

std::cout << "\nError absoluto maximo FP32: " << error_max_fp32 << "\n";
std::cout << "Error absoluto maximo FP16: " << error_max_fp16 << "\n";


// Ejecutar pruebas con diferentes tamanos
probar_tamanos();
medir_tiempos();

return 0;
}