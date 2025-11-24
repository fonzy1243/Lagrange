#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <random>
#include "Sequential.hpp"

Body::Body() : mass(0.0) {
    for (int d = 0; d < DIM; d++) {
        pos[d] = vel[d] = acc[d] = 0.0f;
    }
}

::Sequential::Sequential(int numBodies, int numSteps) :
    numBodies(numBodies),
    numSteps(numSteps)
{
    bodies.resize(numBodies);
    oldAcc.resize(numBodies * DIM);
}

void Sequential::initialize() {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> posDist(-2.5f, 2.5f);
    std::uniform_real_distribution<float> velDist(0.0f, 0.0f);
    std::uniform_real_distribution<float> massDist(0.5f, 1.5f);

    for (int i = 0; i < numBodies; i++) {
        for (int d = 0; d < DIM; d++) {
            bodies[i].pos[d] = posDist(gen);
            bodies[i].vel[d] = velDist(gen);
            bodies[i].acc[d] = 0.0f;
        }
        bodies[i].mass = massDist(gen);
    }
}

void Sequential::computeForces() {
    float distSq = SOFTENING * SOFTENING;
    for (int i = 0; i < numBodies; ++i) {
        for (int d = 0; d < DIM; ++d) {
            bodies[i].acc[d] = 0.0f;
        }
    }

    for (int i = 0; i < numBodies; i++) {
        for (int j = i + 1; j < numBodies; j++) {
            float r[DIM];

            for (int d = 0; d < DIM; d++) {
                r[d] = bodies[j].pos[d] - bodies[i].pos[d];
                distSq += r[d] * r[d];
            }

            const float dist = std::sqrt(distSq);
            const float denominator = distSq * dist;

            const float forceMag = G * bodies[i].mass * bodies[j].mass / denominator;

            for (int d = 0; d < DIM; d++) {
                const float forceD = forceMag * r[d];
                bodies[i].acc[d] += forceD / bodies[i].mass;
                bodies[j].acc[d] -= forceD / bodies[j].mass;
            }
        }
    }
}

void ::Sequential::update_gl_buffer() {
    for (int i = 0; i < numBodies; i++) {
        for (int d = 0; d < DIM; d++) {
            bodies[i].vel[d] += 0.5f * (oldAcc[i * DIM + d] + bodies[i].acc[d]) * DT;
        }
    }
}

std::vector<Body>& ::Sequential::getBodies() {
    return bodies;
}

void ::Sequential::step() {
    for (int i = 0; i < numBodies; i++) {
        for (int d = 0; d < DIM; d++) {
            bodies[i].pos[d] += bodies[i].vel[d] * DT + 0.5 * bodies[i].acc[d] * DT * DT;
            oldAcc[i * DIM + d] = bodies[i].acc[d];
        }
    }

    computeForces();
}

void ::Sequential::run() {
    std::cout << "--- Phase 1: C++ Sequential N-Body Simulation ---" << std::endl;
    std::cout << "Number of bodies: " << numBodies << std::endl;
    std::cout << "Time steps: " << numSteps << std::endl;
    std::cout << "===================\n";

    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Starting simulation... \n";
    for (int step = 0; step <= numSteps; step++) {
        update_gl_buffer();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end - start;

    // calculate GFLOPS
    float interactionsPerStep = (float)numBodies * (float)(numBodies - 1) / 2.0f;
    float flopsPerStep = interactionsPerStep * 20.0f + (float)numBodies * 15.0f;
    float gFlops = (flopsPerStep * (float)numSteps) / (elapsed.count() * 1e9f);

    std::cout << "===================\n";
    std::cout << "Total execution time: " << elapsed.count() << " seconds\n";
    std::cout << "Average time per step: " << (elapsed.count() * 1000.0f / (float)numSteps) << " ms\n";
    std::cout << "Performance: " << gFlops << " GFLOPS\n";
}