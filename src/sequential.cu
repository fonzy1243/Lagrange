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

struct Body {
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

class NBodySystem {
private:
    std::vector<Body> bodies;
    std::vector<double> oldAcc;
    int n;

public:
    explicit NBodySystem(const int N) : n(N) {
        bodies.resize(N);
        oldAcc.resize(N * DIM);
    }

    static float randomFloat(const float min, const float max) {
        return min + (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * (max - min);
    }

    void initBodies(const float posRange, const float velRange, const float massRange) {
        srand(0); // for error checking later, I used fixed seed value

        for (int i = 0; i < n; i++) {
            for (int d = 0; d < DIM; d++) {
                bodies[i].pos[d] = randomFloat(-posRange, posRange);
                bodies[i].vel[d] = randomFloat(-velRange, velRange);
                bodies[i].acc[d] = 0.0;
            }
            bodies[i].mass = randomFloat(1.0f, massRange);
        }
    }

    void computeForces() {
        for (int i = 0; i < n; ++i) {
            for (int d = 0; d < DIM; ++d) {
                bodies[i].acc[d] = 0.0;
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double r[DIM];
                double distSq = SOFTENING * SOFTENING;

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

    void integrateVelocityVerlet() {
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < DIM; d++) {
                bodies[i].pos[d] += bodies[i].vel[d] * DT + 0.5 * bodies[i].acc[d] * DT * DT;
            }
        }

        for (int i = 0; i < n; i++) {
            for (int d = 0; d < DIM; d++) {
                oldAcc[i * DIM + d] = bodies[i].acc[d];
            }
        }

        computeForces();

        for (int i = 0; i < n; i++) {
            for (int d = 0; d < DIM; d++) {
                bodies[i].vel[d] += 0.5 * (oldAcc[i * DIM + d] + bodies[i].acc[d]) * DT;
            }
        }
    }

};