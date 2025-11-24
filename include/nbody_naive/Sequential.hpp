#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include <vector>

// constexpr int N_BODIES = 1024;
constexpr int DIM = 3;
constexpr float DT = 0.001f;
constexpr float G = 1.0f;
constexpr float SOFTENING = 1e-5f;
// constexpr int N_STEPS = 10000;

struct Body {
    float pos[DIM]{};
    float vel[DIM]{};
    float acc[DIM]{};
    float mass;

    Body();
};

class Sequential {
private:
    std::vector<Body> bodies;
    std::vector<float> oldAcc;

    void computeForces();

public:
    Sequential(int numBodies, int numSteps);

    int numBodies;
    int numSteps;
    void initialize();
    void update_gl_buffer();

    std::vector<Body>& getBodies();
    void step();
    void run();
};

#endif
