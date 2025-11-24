#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <shader/shader.h>

#include <nbody_naive/Sequential.hpp>
#include <nbody_barnes_cuda/barnes_cuda.cuh>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;
const int N_BODIES = 10000;

int main() {
    // glfw init and configure
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // glfw window creation
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Lagrange", nullptr, nullptr);
    if (window == nullptr) {
        std::cout << "Failed to create GLFW window." << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad load OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD." << std::endl;
        return -1;
    }

    // Enable point rendering
    glEnable(GL_PROGRAM_POINT_SIZE);
    glPointSize(2.0f);

    // Initialize shaders
    Shader particleShader("../shaders/particle.vert", "../shaders/particle.frag");

    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, N_BODIES * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Initialize simulator
    std::cout << "Initializing simulator with " << N_BODIES << " bodies." << std::endl;

    BarnesHut simulator(N_BODIES);

    // Initialize positions
    float4* initial_pos = new float4[N_BODIES];
    float4* initial_vel = new float4[N_BODIES];

    srand(time(nullptr));
    for (int i = 0; i < N_BODIES; i++) {
        initial_pos[i].x = ((float)rand() / RAND_MAX - 0.5f) * 5.0f;
        initial_pos[i].y = ((float)rand() / RAND_MAX - 0.5f) * 5.0f;
        initial_pos[i].z = ((float)rand() / RAND_MAX - 0.5f) * 5.0f;
        initial_pos[i].w = 1.f / N_BODIES;

        initial_vel[i].x = 0.f;
        initial_vel[i].y = 0.f;
        initial_vel[i].z = 0.f;
        initial_vel[i].w = 0.f;
    }

    simulator.initialize(initial_pos, initial_vel);
    simulator.setup_gl_interop(VBO);

    delete[] initial_pos;
    delete[] initial_vel;

    // Setup viewing matrices
    glm::mat4 projection = glm::perspective(45.0f * glm::pi<float>() / 180.f, (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);

    particleShader.use();
    GLint projLoc = glGetUniformLocation(particleShader.ID, "projection");
    GLint viewLoc = glGetUniformLocation(particleShader.ID, "view");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection[0][0]);

    std::cout << "Starting simulation..." << std::endl;

    int frame_count = 0;
    double last_time = glfwGetTime();

    // render loop
    while (!glfwWindowShouldClose(window)) {
        // input
        processInput(window);

        // Update simulation
        simulator.step();
        simulator.update_gl_buffer();

        // Calculate camera position
        float time = (float)glfwGetTime();
        float camX = sin(time * 0.2f) * 5.f;
        float camZ = cos(time * 0.2f) * 5.f;

        glm::vec3 eye = glm::vec3(camX, 2.f, camZ);
        glm::vec3 center = glm::vec3(0.0f, 0.0f, 0.0f);
        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

        glm::mat4 view = glm::lookAt(eye, center, up);
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

        // render
        glClearColor(0.0f, 0.0f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        particleShader.use();
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, N_BODIES);

        // FPS counter
        frame_count++;
        double current_time = glfwGetTime();
        if (current_time - last_time >= 1.0) {
            std::cout << "FPS: " << frame_count << std::endl;
            frame_count = 0;
            last_time = current_time;
        }

        // swap buffers and poll IO
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    // glfw terminate
    glfwTerminate();
    return 0;
}

// Process all input
void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

// GLFW window size change callback
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}