#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>

constexpr int DIMENSIONS = 3;
constexpr int N_BODIES = 1024;
constexpr float TOTAL_TIME = 1.0f;
constexpr float DT = 0.01f;
constexpr float G = 6.674e-11f;
constexpr float G_CONST = 1.0f;
constexpr float EPSILON = 0.1f;

constexpr int SAVE_EVERY_N_STEPS = 5;

typedef struct
{
    double pos[DIMENSIONS];
    double vel[DIMENSIONS];
    double acc[DIMENSIONS];
    double newAcc[DIMENSIONS];
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

float randomFloat(const float min, const float max)
{
    return min + (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * (max - min);
}

void initBodies(const NBodySystem* system, float posRange, float velRange, float massRange)
{
    srand(0); // for error checking later, I used fixed seed value

    for (int i = 0; i < system->n; ++i)
    {
        for (int d = 0; d < DIMENSIONS; ++d)
        {
            system->bodies[i].pos[d] = randomFloat(-posRange, posRange);
            system->bodies[i].vel[d] = randomFloat(-velRange, velRange);
        }
        system->bodies[i].mass = randomFloat(1.0f, massRange);
    }
}

void calculateAcc(NBodySystem* system)
{
    for (int i = 0; i < system->n; ++i)
    {
        float totalAccX = 0.0f;
        float totalAccY = 0.0f;
        float totalAccZ = 0.0f;

        for (int j = 0; j < system->n; ++j)
        {
            if (i == j) continue;

            float dx = system->bodies[j].pos[0] - system->bodies[i].pos[0];
            float dy = system->bodies[j].pos[1] - system->bodies[i].pos[1];
            float dz = system->bodies[j].pos[2] - system->bodies[i].pos[2];

            float distSq = (dx * dx) + (dy * dy) + (dz * dz) + (EPSILON * EPSILON);

            float invDist = 1.0f / sqrt(distSq);
            float invDistCubed = invDist * invDist * invDist;

            float jMass = system->bodies[j].mass;
            totalAccX += G * jMass * dx * invDistCubed;
            totalAccY += G * jMass * dy * invDistCubed;
            totalAccZ += G * jMass * dz * invDistCubed;
        }

        system->bodies[i].newAcc[0] = totalAccX;
        system->bodies[i].newAcc[1] = totalAccY;
        system->bodies[i].newAcc[2] = totalAccZ;
    }
}