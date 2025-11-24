#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <random>
#include "../../src/Sequential.hpp"

Body::Body() : mass(0.0) {
    for (int d = 0; d < DIM; d++) {
        pos[d] = vel[d] = acc[d] = 0.0;
    }
}

::NBodySystem::NBodySystem() {
    bodies.resize(N_BODIES);
    oldAcc.resize(N_BODIES * DIM);
}

void NBodySystem::initBodies() {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<double> posDist(-1.0, 1.0);
    std::uniform_real_distribution<double> velDist(0.0, 0.0);
    std::uniform_real_distribution<double> massDist(0.5, 1.5);

    for (int i = 0; i < N_BODIES; i++) {
        for (int d = 0; d < DIM; d++) {
            bodies[i].pos[d] = posDist(gen);
            bodies[i].vel[d] = velDist(gen);
            bodies[i].acc[d] = 0.0;
        }
        bodies[i].mass = massDist(gen);
    }
}

void NBodySystem::computeForces() {
    double distSq = SOFTENING * SOFTENING;
    for (int i = 0; i < N_BODIES; ++i) {
        for (int d = 0; d < DIM; ++d) {
            bodies[i].acc[d] = 0.0;
        }
    }

    for (int i = 0; i < N_BODIES; i++) {
        for (int j = i + 1; j < N_BODIES; j++) {
            double r[DIM];

            for (int d = 0; d < DIM; d++) {
                r[d] = bodies[j].pos[d] - bodies[i].pos[d];
                distSq += r[d] * r[d];
            }

            const double dist = std::sqrt(distSq);
            const double denominator = distSq * dist;

            const double forceMag = G * bodies[i].mass * bodies[j].mass / denominator;

            for (int d = 0; d < DIM; d++) {
                const double forceD = forceMag * r[d];
                bodies[i].acc[d] += forceD / bodies[i].mass;
                bodies[j].acc[d] -= forceD / bodies[j].mass;
            }
        }
    }
}

void ::NBodySystem::integrateVelocityVerlet() {
    for (int i = 0; i < N_BODIES; i++) {
        for (int d = 0; d < DIM; d++) {
            bodies[i].pos[d] += bodies[i].vel[d] * DT + 0.5 * bodies[i].acc[d] * DT * DT;
            oldAcc[i * DIM + d] = bodies[i].acc[d];
        }
    }

    computeForces();

    for (int i = 0; i < N_BODIES; i++) {
        for (int d = 0; d < DIM; d++) {
            bodies[i].vel[d] += 0.5 * (oldAcc[i * DIM + d] + bodies[i].acc[d]) * DT;
        }
    }
}

const std::vector<Body>& ::NBodySystem::getBodies() const {
    return bodies;
}

void ::NBodySystem::step() {
    integrateVelocityVerlet();
}

void ::NBodySystem::run() {
    std::cout << "--- Phase 1: C++ Sequential N-Body Simulation ---" << std::endl;
    std::cout << "Number of bodies: " << N_BODIES << std::endl;
    std::cout << "Time steps: " << N_STEPS << std::endl;
    std::cout << "===================\n";

    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Starting simulation... \n";
    for (int step = 0; step <= N_STEPS; step++) {
        integrateVelocityVerlet();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // calculate GFLOPS
    double interactionsPerStep = (double)N_BODIES * (double)(N_BODIES - 1) / 2.0;
    double flopsPerStep = interactionsPerStep * 20.0 + N_BODIES * 15.0;
    double gFlops = (flopsPerStep * N_STEPS) / (elapsed.count() * 1e9);

    std::cout << "===================\n";
    std::cout << "Total execution time: " << elapsed.count() << " seconds\n";
    std::cout << "Average time per step: " << (elapsed.count() * 1000.0 / N_STEPS) << " ms\n";
    std::cout << "Performance: " << gFlops << " GFLOPS\n";
}