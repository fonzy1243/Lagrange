#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>

constexpr int N_BODIES = 1024;
constexpr double TOTAL_TIME = 3.154e7;
constexpr double DT = 86400.0;
constexpr int OUTPUT_INTERVAL = 10;

constexpr int DIM = 3;
constexpr double G = 6.67430e-11;
constexpr double SOFTENING = 1e9;

struct Body
{
    double pos[DIM];
    double vel[DIM];
    double acc[DIM];
    double mass;

    Body() : mass(0.0) {
        for (int d = 0; d < DIM; d++) {
            pos[d] = vel[d] = acc[d] = 0.0;
        }
    }
};

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

    for (int i = 0; i < system->n; i++)
    {
        for (int d = 0; d < DIM; d++)
        {
            system->bodies[i].pos[d] = randomFloat(-posRange, posRange);
            system->bodies[i].vel[d] = randomFloat(-velRange, velRange);
        }
        system->bodies[i].mass = randomFloat(1.0f, massRange);
    }
}

void computeForces(const NBodySystem* system)
{
    for (int i = 0; i < system->n; ++i)
    {
        for (int d = 0; d < DIM; ++d)
        {
            system->bodies[i].acc[d] = 0.0;
        }
    }

    for (int i = 0; i < system->n; i++)
    {
        for (int j = i + 1; j < system->n; j++)
        {
            double r[DIM];
            double distSq = SOFTENING * SOFTENING;

            for (int d = 0; d < DIM; d++)
            {
                r[d] = system->bodies[j].pos[d] - system->bodies[i].pos[d];
                distSq += r[d] * r[d];
            }

            const double dist = std::sqrt(distSq);
            const double denominator = distSq * dist;

            const double forceMag = G * system->bodies[i].mass * system->bodies[j].mass / denominator;

            for (int d = 0; d < DIM; d++)
            {
                const double forceD = forceMag * r[d];
                system->bodies[i].acc[d] += forceD / system->bodies[i].mass;
                system->bodies[j].acc[d] -= forceD / system->bodies[j].mass;
            }
        }
    }
}
