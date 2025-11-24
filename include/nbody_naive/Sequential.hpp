#ifndef N_BODY_SYSTEM_HPP
#define N_BODY_SYSTEM_HPP

#include <vector>

// constexpr int N_BODIES = 1024;
constexpr int DIM = 3;
constexpr double DT = 0.001;
constexpr double G = 1.0;
constexpr double SOFTENING = 1e-5;
// constexpr int N_STEPS = 10000;

struct Body {
    double pos[DIM]{};
    double vel[DIM]{};
    double acc[DIM]{};
    double mass;

    Body();
};

class NBodySystem {
private:
    std::vector<Body> bodies;
    std::vector<double> oldAcc;

    void computeForces();

public:
    NBodySystem();

    void initBodies();
    void integrateVelocityVerlet();

    const std::vector<Body>& getBodies() const;
    void step();
    void run();
};

#endif
