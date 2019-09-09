#define XTENSOR_USE_XSIMD
#include <stdio.h>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xnpy.hpp"
#include "xtensor/xrandom.hpp"
#include <omp.h>
#include <stddef.h>
#include <typeinfo>
#include <stdexcept>

#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include "xtensor-blas/xlinalg.hpp"

template <class _Tp>
xt::xarray<_Tp> matmul(xt::xarray<_Tp> a, xt::xarray<_Tp> b) noexcept(false)
{
    if (a.shape().size() != b.shape().size()) {
        throw std::runtime_error("Shape mismatching!");
    }

    if (typeid(_Tp).hash_code() != typeid(int).hash_code()
        && typeid(_Tp).hash_code() != typeid(size_t).hash_code()
        && typeid(_Tp).hash_code() != typeid(float).hash_code()
        && typeid(_Tp).hash_code() != typeid(double).hash_code()) {
        throw std::runtime_error("Element type mismatching!");
    }

    int N = a.shape().at(0);
    int M =  b.shape().at(1);
    int items = a.shape().at(1);
    double temporal = 0.0;
    xt::xarray<_Tp> out = xt::zeros<double>({N,M});

    //
    // Both argument are 2-D, end the recursion.
    //
    //      a - (M, M)
    //      b - (M, M)
    // 
    if (out.shape().size() == 2) {
        for (int i = 0 ; i < N ; i++ ) //i para las filas de la matriz resultante
        {
            for (int k = 0 ; k < M ; k++ ) // k para las columnas de la matriz resultante
            {
                temporal = 0 ;
                for (int j = 0 ; j < items ; j++ ) //j para realizar la multiplicacion de 
                {                                   //los elementos   de la matriz
                    temporal += a(i,j) * b(j,k);
                    out(i,k) = temporal ;
                }
            }
        }
    }
    return out;
}

int endmembers(std::string dataset){
    if (!dataset.compare("samson")){
        return 3;
    } else if (!dataset.compare("jasper") || !dataset.compare("urban")) {
        return 4;
    } else if (!dataset.compare("cuprite")) {
        return 12;
    }
    return -1;

}

void EStep(long N, int M, int K, xt::xarray<double>& p, xt::xarray<double>& theta, xt::xarray<double>& lamda){
    double denominator = 0.0;
    for (int i = 0; i < N; i++){
        for (int j = 0; j < M; j++){
            denominator = 0.0;
            for (int k = 0; k < K; k++){
                p(i, j, k) = theta(k, j) * lamda(i, k);
                denominator += p(i, j, k);
            }
            if(denominator == 0.0){
                for (int k = 0; k < K; k++){
                    p(i, j, k) = 0;
                }
            } else {
                for (int k = 0; k < K; k++){
                    p(i, j, k) /= denominator;
                }
            }
        }
    }
}

void MStep(long N, int M,  int K,  xt::xarray<double>& p,  xt::xarray<float>& X, xt::xarray<double>& theta, xt::xarray<double>& lamda, double r1, double r2){
    double denominator = 0.0;
    double valueaux = 0.0;
    // MStep theta
    for (int k = 0; k < K; k++){
        denominator = 0.0;
        for (int j = 0; j < M; j++){
            theta(k,j) = 0;
            for (int i = 0; i < N; i++){
                valueaux = X(i, j) * p(i, j, k) - (r1 / float(M));
                if(valueaux > 0) theta(k,j) += valueaux;
            }
            denominator += theta(k,j);
        }
        if(denominator == 0.0){
            for (int j = 0; j < M; j++){
                theta(k,j) = 1.0 / M;
            }
        } else {
            for (int j = 0; j < M; j++){
                theta(k,j) /= denominator;
            }
        }
    }


    // MStep lamda
    for (int i = 0; i < N; i++){
        for (int k = 0; k < K; k++){
            denominator = 0.0;
            lamda(i,k) = 0;
            for (int j = 0; j < M; j++){
                valueaux = X(i, j) * p(i, j, k) - (r2 / float(K));
                if(valueaux > 0) lamda(i,k) += valueaux;
                denominator += X(i,j);
            }
            if(denominator == 0.0){
                lamda(i,k) = 1.0 / K;
            } else {
                lamda(i,k) /= denominator;
            }
        }
    }
}


void PLSA(std::string dataset, std::string id, int iterations, int K, double r1, double r2){

    std::stringstream input_stream;
    input_stream << "inputs/" << dataset << ".npy";
    std::string input = input_stream.str();
    std::string endmembers_file = "endmembers_plsa_" + dataset + "_" + id + ".npy";
    std::string abundances_file = "abundances_plsa_" + dataset + "_" + id + ".npy";
    xt::xarray<float> X = xt::load_npy<float>(input);
    auto shape = X.shape();
    
    int nRow = shape.at(0);
    int nCol = shape.at(1);
    long N = nRow * nCol;
    int M = shape.at(2);
    X.reshape({nRow * nCol, M});

    xt::random::seed(0);
    xt::xarray<double> lamda = xt::random::rand<double>({nRow * nCol, K});
    xt::xarray<double> theta = xt::random::rand<double>({K, M});

    xt::xarray<double> p = xt::zeros<double>({nRow * nCol, M, K});


    printf ("Running over image %s, with shape [%d, %d, %d] \n", dataset.c_str(), nRow, nCol, M);
    double total = omp_get_wtime();
    for(int iter = 0; iter < iterations; iter++){
        EStep(N, M, K, p, theta, lamda);
        MStep(N, M, K ,p, X, theta, lamda, r1, r2);
    }
    double finished = omp_get_wtime();
    printf("Dataset %s || Total time %f secs \n", dataset.c_str(), finished - total);

    xt::xarray<double> endmembers = xt::transpose(theta, {1,0});
    xt::dump_npy(endmembers_file, endmembers);
    lamda.reshape({nRow, nCol, K});
    xt::dump_npy(abundances_file, lamda);
}

void DPLSA(std::string dataset, std::string id, int iterations, int K_2, double r1, double r2){

    std::stringstream input_stream;
    input_stream << "inputs/" << dataset << ".npy";
    std::string input = input_stream.str();
    std::string endmembers_file = "endmembers_dplsa_" + dataset + "_" + id + ".npy";
    std::string abundances_file = "abundances_dplsa_" + dataset + "_" + id + ".npy";
    xt::xarray<float> X = xt::load_npy<float>(input);
    auto shape = X.shape();
    int iter_dplsa = 100;
    int K = 250;
    int nRow = shape.at(0);
    int nCol = shape.at(1);
    long N = nRow * nCol;
    int M = shape.at(2);
    X.reshape({nRow * nCol, M});

    xt::random::seed(0);
    xt::xarray<double> lamda = xt::random::rand<double>({nRow * nCol, K});
    xt::xarray<double> theta = xt::random::rand<double>({K, M});

    xt::xarray<double> p = xt::zeros<double>({nRow * nCol, M, K});


    printf ("Running over image %s, with shape [%d, %d, %d] \n", dataset.c_str(), nRow, nCol, M);

    double total = omp_get_wtime();
    for(int iter = 0; iter < iter_dplsa; iter++){
        EStep(N, M, K, p, theta, lamda);
        MStep(N, M, K ,p, X, theta, lamda, r1, r2);
    }
    double finished = omp_get_wtime();
    printf("Dataset %s || Total time step 1: %f secs \n", dataset.c_str(), finished - total);

    M = K;
    xt::xarray<float> X_2 = lamda;
    lamda = xt::random::rand<double>({nRow * nCol, K_2});
    xt::xarray<double> theta_2 = xt::random::rand<double>({K_2, M});

    p = xt::zeros<double>({nRow * nCol, M, K_2});

    total = omp_get_wtime();
    for(int iter = 0; iter < iterations; iter++){
        EStep(N, M, K_2, p, theta_2, lamda);
        MStep(N, M, K_2 ,p, X_2, theta_2, lamda, r1, r2);
    }

    finished = omp_get_wtime();
    printf("Dataset %s || Total time step 2: %f secs \n", dataset.c_str(), finished - total);

    xt::xarray<double> e1 = xt::transpose(theta, {1,0});
    xt::xarray<double> e2 = xt::transpose(theta_2, {1,0});
    xt::xarray<double> endmembers = matmul(e1, e2);
    xt::dump_npy(endmembers_file, endmembers);
    lamda.reshape({nRow, nCol, K_2});
    xt::dump_npy(abundances_file, lamda);
}

int main(int argc, char* argv[])
{
    if (argc != 4) {
        printf("Uso %s dataset mode id \n", argv[0]);
        return -1;
    }

    int iterations = 1000;
    double regularization1 = 0;
    double regularization2 = 0;
    std::string dataset = argv[1];
    std::string mode = argv[2];
    std::string id = argv[3];

    int K = endmembers(dataset);
    printf("Endmembers: %d \n", K);
    if (K < 0) {
        printf("Invalid dataset \n");
        return -1;
    }
    if(mode == "pLSA")
        PLSA(dataset, id, iterations, K, regularization1, regularization2);
    else if (mode == "dpLSA")
        DPLSA(dataset, id, iterations, K, regularization1, regularization2);
    else {
        printf("Invalid mode \n");
        return -1;
    }
}
