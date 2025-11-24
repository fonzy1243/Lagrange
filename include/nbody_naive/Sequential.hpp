#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include <vector>
#include <glad/glad.h>

// constexpr int N_BODIES = 1024;
constexpr int DIM = 3;
constexpr float DT = 0.001f;
constexpr float G = 1.0f;
constexpr float SOFTENING = 1e-5f;

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

    GLuint vbo;
    bool gl_initialized;

    void computeForces();

public:
    Sequential(int numBodies);

    int numBodies;

    void initialize(const float* initial_pos, const float* initial_vel);
    void setup_gl_interop(GLuint vertex_buffer);
    void update_gl_buffer();

    std::vector<Body>& getBodies();
    void step();
    void run();
};

#endif
