#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>

#define DIMENSION 3

constexpr int N_BODIES = 1024;
constexpr float TOTAL_TIME = 1.0f;
constexpr float DT = 0.01f;
constexpr float G = 6.674e-11f;
constexpr float G_CONST = 1.0f;
constexpr float SOFTENING = 0.1f;

constexpr int SAVE_EVERY_N_STEPS = 5;

typedef struct
{
    double pos[DIMENSION];
    double vel[DIMENSION];
    double acc[DIMENSION];
    double newAcc[DIMENSION];
    double mass;
} Body;

struct NBodySystem
{
    int n;
    Body* bodies;

    explicit NBodySystem(const int N): n(N)
    {
        bodies = new Body[n];
        memset(bodies, 0, n * sizeof(Body));
    }

    ~NBodySystem()
    {
        delete[] bodies;
    }
};