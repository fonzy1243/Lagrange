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

::Sequential::Sequential(int numBodies) :
    numBodies(numBodies),
    gl_initialized(false),
    vbo(0)
{
    bodies.resize(numBodies);
    oldAcc.resize(numBodies * DIM);
}

void Sequential::initialize(const float* initial_pos, const float* initial_vel) {
    for (int i = 0; i < numBodies; i++) {
        for (int d = 0; d < DIM; d++) {
            bodies[i].pos[d] = initial_pos[i * 4 + d];
            bodies[i].vel[d] = initial_vel[i * 4 + d];
            bodies[i].acc[d] = 0.0f;
        }
        bodies[i].mass = initial_pos[i * 4 + 3];
    }
}

void Sequential::setup_gl_interop(GLuint vertex_buffer) {
    vbo = vertex_buffer;
    gl_initialized = true;
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
    if (!gl_initialized) return;

    float* buffer = new float[numBodies * 3];
    for (int i = 0; i < numBodies; i++) {
        buffer[i * 3 + 0] = bodies[i].pos[0];
        buffer[i * 3 + 1] = bodies[i].pos[1];
        buffer[i * 3 + 2] = bodies[i].pos[2];
    }

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, numBodies * 3 * sizeof(float), buffer);

    delete[] buffer;
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