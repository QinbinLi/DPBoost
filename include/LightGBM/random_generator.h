//
// Created by qinbin on 17/07/19.
//

#ifndef LIGHTGBM_RANDOM_GENERATOR_H
#define LIGHTGBM_RANDOM_GENERATOR_H

#include <iostream>
#include <random>

class Laplace{
public:
    float scale;
    std::mt19937 generator;
    std::default_random_engine generator1;
    std::default_random_engine generator2;
    std::exponential_distribution<float> distribution;
    Laplace(int seed): generator(seed){};
    Laplace(float scale, int seed): scale(scale), generator(seed), distribution(1.0/scale){};
    float return_a_random_variable(){
        float e1 = distribution(generator);
        float e2 = distribution(generator);
        return e1-e2;
    }
    float return_a_random_variable(float scale){
        std::exponential_distribution<float> distribution1(1.0/scale);
        std::exponential_distribution<float> distribution2(1.0/scale);
        float e1 = distribution1(generator);
        float e2 = distribution2(generator);
        return e1-e2;
    }
};


#endif //LIGHTGBM_RANDOM_GENERATOR_H
