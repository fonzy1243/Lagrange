#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>

constexpr int N_BODIES = 1024;
constexpr double TOTAL_TIME = 10.0;
constexpr double DT = 0.001;
constexpr int OUTPUT_INTERVAL = 10;

constexpr int DIM = 3;
constexpr double G = 1.0; // rescaled from 6.67430e-11;
constexpr double SOFTENING = 0.05;

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

public:
    explicit NBodySystem() {
        bodies.resize(N_BODIES);
        oldAcc.resize(N_BODIES * DIM);
    }

    static double randomDouble(const double min, const double max) {
        return min + (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * (max - min);
    }

    void initBodies(const double posRange, const double velRange, const double minMass, const double maxMass) {
        srand(0); // for error checking later, I used fixed seed value

        for (int i = 0; i < N_BODIES; i++) {
            for (int d = 0; d < DIM; d++) {
                bodies[i].pos[d] = randomDouble(-posRange, posRange);
                bodies[i].vel[d] = randomDouble(-velRange, velRange);
                bodies[i].acc[d] = 0.0;
            }
            bodies[i].mass = randomDouble(minMass, maxMass);
        }
    }

    void computeForces() {
        for (int i = 0; i < N_BODIES; ++i) {
            for (int d = 0; d < DIM; ++d) {
                bodies[i].acc[d] = 0.0;
            }
        }

        for (int i = 0; i < N_BODIES; i++) {
            for (int j = i + 1; j < N_BODIES; j++) {
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
        for (int i = 0; i < N_BODIES; i++) {
            for (int d = 0; d < DIM; d++) {
                bodies[i].pos[d] += bodies[i].vel[d] * DT + 0.5 * bodies[i].acc[d] * DT * DT;
            }
        }

        for (int i = 0; i < N_BODIES; i++) {
            for (int d = 0; d < DIM; d++) {
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

    double calculateEnergy () const {
        double kinetic = 0.0;
        double potential = 0.0;

        for (int i = 0; i < N_BODIES; i++) {
            double vSq = 0.0;
            for (int d = 0; d < DIM; d++) {
                vSq += bodies[i].vel[d] * bodies[i].vel[d];
            }
            kinetic += 0.5 * bodies[i].mass * vSq;
        }

        for (int i = 0; i < N_BODIES; i++) {
            for (int j = i + 1; j < N_BODIES; j++) {
                double rSq = SOFTENING * SOFTENING;
                for (int d = 0; d < DIM; d++) {
                    const double dr = bodies[j].pos[d] - bodies[i].pos[d];
                    rSq += dr * dr;
                }
                potential -= G * bodies[i].mass * bodies[j].mass / std::sqrt(rSq);
            }
        }

        return kinetic + potential;
    }

    void run() {
        std::cout << "N-Body Problem Simulation\n";
        std::cout << "=======================\n";
        std::cout << "Number of bodies: " << N_BODIES << "\n";
        std::cout << "Time step: " << std::scientific << DT << " s\n";
        std::cout << "Total time: " << TOTAL_TIME << " s\n";
        std::cout << "Output interval: " << OUTPUT_INTERVAL << " steps\n\n";

        const double initialEnergy = calculateEnergy();
        std::cout << "Initial energy: " << std::scientific << std::setprecision(6) << initialEnergy << " J\n\n";

        // loop starts here
        const int nSteps = static_cast<int>(TOTAL_TIME / DT);
        auto start = std::chrono::high_resolution_clock::now();

        std::cout << "Starting simulation... \n";
        for (int step = 0; step <= nSteps; step++) {
            const double currTime  = step * DT;

            if (step % OUTPUT_INTERVAL == 0) {
                if (step % (OUTPUT_INTERVAL * 10) == 0) {
                    const double energy = calculateEnergy();
                    const double energyError = std::abs((energy - initialEnergy) / initialEnergy) * 100.0;
                    std::cout << "Step " << std::setw(6) << step << "/" << nSteps
                              << " | Time: " << std::scientific << std::setprecision(3) << currTime
                              << " s | Energy error: " << std::fixed << std::setprecision(6)
                              << energyError << "%\n";
                }
            }

            if (step < nSteps) {
                integrateVelocityVerlet();
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        const double finalEnergy = calculateEnergy();
        const double energyError = std::abs((finalEnergy - initialEnergy) / initialEnergy) * 100.0;
        std::cout << "\nSimulation done\n";
        std::cout << "===================\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
        std::cout << "Time per step: " << (elapsed.count() * 1000.0 / nSteps) << " ms\n";
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "Initial energy: " << initialEnergy << " J\n";
        std::cout << "Final energy: " << finalEnergy << " J\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Energy error: " << energyError << "%\n";
        std::cout << std::setprecision(3);
        std::cout << "Performance: " << (static_cast<double>(N_BODIES) * N_BODIES * nSteps / elapsed.count() / 1e6) << " M-interactions/sec\n";
    }
};

int main(int argc, char *argv[]) {
    NBodySystem nBodySystem;

    nBodySystem.initBodies(1.0, 0.2, 0.5, 1.5);
    nBodySystem.run();

    return 0;
}