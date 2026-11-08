#include <iostream>
#include <cuda_runtime.h>

// 1. Le KERNEL : la fonction qui s'exécute sur le GPU
// __global__ indique que la fonction est appelée par le CPU mais exécutée sur le GPU
__global__ void addition_gpu(float *A, float *B, float *C, int N) {
    // Calcul de l'index unique de ce thread dans tout le GPU
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Protection pour ne pas déborder du tableau
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1000;
    size_t taille = N * sizeof(float);

    // ---- ÉTAPE 1 : Allocations sur le Host (CPU) ----
    float *h_A = (float*)malloc(taille);
    float *h_B = (float*)malloc(taille);
    float *h_C = (float*)malloc(taille);

    // Initialisation des données sur le CPU
    for(int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // ---- ÉTAPE 2 : Allocations sur le Device (GPU) ----
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, taille); // Identique à malloc, mais renvoie un pointeur VRAM
    cudaMalloc(&d_B, taille);
    cudaMalloc(&d_C, taille);

    // ---- ÉTAPE 3 : Copie CPU -> GPU ----
    cudaMemcpy(d_A, h_A, taille, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, taille, cudaMemcpyHostToDevice);

    // ---- ÉTAPE 4 : Configuration et Lancement du Kernel ----
    // On définit combien de threads on veut lancer.
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // La syntaxe <<< blocks, threads >>> est propre à CUDA
    addition_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Attendre que le GPU ait fini son calcul
    cudaDeviceSynchronize();

    // ---- ÉTAPE 5 : Copie GPU -> CPU ----
    cudaMemcpy(h_C, d_C, taille, cudaMemcpyDeviceToHost);

    // Vérification du résultat (paresse intelligente : on vérifie juste le premier et le dernier)
    std::cout << "Premier element : " << h_C[0] << " (attendu: 3.0)" << std::endl;
    std::cout << "Dernier element : " << h_C[N-1] << " (attendu: 3.0)" << std::endl;

    // ---- ÉTAPE 6 : Nettoyage de la mémoire ----
    cudaFree(d_A); // Libération mémoire GPU
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);     // Libération mémoire CPU
    free(h_B);
    free(h_C);

    return 0;
}
