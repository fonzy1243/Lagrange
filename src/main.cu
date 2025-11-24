#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <shader/shader.h>
#include <camera/camera.h>

#include <nbody_naive/Sequential.hpp>
#include <nbody_barnes_cuda/barnes_cuda.cuh>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);

const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;
const int N_BODIES = 100000;

Camera camera(glm::vec3(0.0f, 10.0f, 50.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

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
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad load OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD." << std::endl;
        return -1;
    }

    // Enable point rendering
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
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

    // Comment out depending on what implementation to demo
    // Sequential simulator(N_BODIES, N_STEPS);
    BarnesHut simulator(N_BODIES);

    // Initialize positions ==> TODO: Maybe we can put this inside BarnesHut's .initialize() na lang?
    float4* initial_pos = new float4[N_BODIES];
    float4* initial_vel = new float4[N_BODIES];

    srand(time(nullptr));

    // Galaxy parameters
    const float central_mass_fraction = 0.0008f;
    const float scale_length = 4.0f;
    const float max_radius = 25.0f;
    const float scale_height = 2.f;  // Slightly thicker for stability
    const float min_radius = 5.0f;

    // Black hole
    initial_pos[0].x = 0.0f;
    initial_pos[0].y = 0.0f;
    initial_pos[0].z = 0.0f;
    initial_pos[0].w = central_mass_fraction;
    initial_vel[0] = {0.0f, 0.0f, 0.0f, 0.0f};

    float particle_mass = (1.f - central_mass_fraction) / (N_BODIES - 1);

    const float spiral_amplitude = 0.15f;
    const float spiral_pitch = 0.3f;       // Tightness of spiral pattern

    for (int i = 1; i < N_BODIES; i++) {
        float u = (float)rand() / RAND_MAX;
        float r = min_radius + (-scale_length * log(1.0f - (u * 0.98f)));
        if (r > max_radius) r = max_radius;

        float theta = ((float)rand() / RAND_MAX) * 2.f * glm::pi<float>();

        float r1 = (float)rand() / RAND_MAX;
        float r2 = (float)rand() / RAND_MAX;
        float z_raw = (r1 + r2 - 1.0f);
        float current_thickness = scale_height * (1.0f + (r / max_radius));
        float z = z_raw * current_thickness;

        initial_pos[i].x = r * cos(theta);
        initial_pos[i].y = z;
        initial_pos[i].z = r * sin(theta);
        initial_pos[i].w = particle_mass;

        // Calculate base circular velocity
        float x = r / scale_length;
        float disk_mass_fraction = 1.0f - central_mass_fraction;
        float exponential_enclosed = 1.0f - exp(-x) * (1.0f + x);
        float mass_enclosed = central_mass_fraction + (disk_mass_fraction * exponential_enclosed);

        float v_circular = sqrt(G * mass_enclosed / r);

        // Add m=2 spiral perturbation (two-armed spiral seed)
        float spiral_phase = 2.0f * theta - spiral_pitch * log(r / scale_length);
        float radial_kick = spiral_amplitude * v_circular * cos(spiral_phase);
        float tangential_kick = spiral_amplitude * v_circular * sin(spiral_phase);

        // Add random "temperature" for velocity dispersion
        float v_random_r = 0.1f * v_circular * ((float)rand() / RAND_MAX - 0.5f);
        float v_random_t = 0.1f * v_circular * ((float)rand() / RAND_MAX - 0.5f);
        float v_random_z = 0.05f * v_circular * ((float)rand() / RAND_MAX - 0.5f);

        // Combine velocities
        float v_radial = radial_kick + v_random_r;
        float v_tangential = v_circular + tangential_kick + v_random_t;

        // Convert to Cartesian
        initial_vel[i].x = v_radial * cos(theta) - v_tangential * sin(theta);
        initial_vel[i].y = v_random_z;
        initial_vel[i].z = v_radial * sin(theta) + v_tangential * cos(theta);
        initial_vel[i].w = 0.f;
    }

    simulator.initialize(initial_pos, initial_vel);
    simulator.setup_gl_interop(VBO);

    delete[] initial_pos;
    delete[] initial_vel;

    particleShader.use();
    GLint projLoc = glGetUniformLocation(particleShader.ID, "projection");
    GLint viewLoc = glGetUniformLocation(particleShader.ID, "view");
    GLint modelLoc = glGetUniformLocation(particleShader.ID, "model");

    glm::mat4 model = glm::mat4(1.0f);
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

    std::cout << "Starting simulation..." << std::endl;

    int frame_count = 0;
    double last_time = glfwGetTime();

    // render loop
    while (!glfwWindowShouldClose(window)) {
        // time logic
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - last_time;
        last_time = currentFrame;

        // input
        processInput(window);

        const int sub_steps = 25;

        // Update simulation
        for (int i = 0; i < sub_steps; i++) {
            simulator.step();
        }

        simulator.update_gl_buffer();

        // render
        glClearColor(0.0f, 0.0f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        particleShader.use();

        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 1000.0f);
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

        glm::mat4 view = camera.GetViewMatrix();
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

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

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        camera.ProcessKeyboard(FORWARD, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        camera.ProcessKeyboard(LEFT, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        camera.ProcessKeyboard(RIGHT, deltaTime);
    }
}

// GLFW window size change callback
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow *window, double xposIn, double yposIn) {
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    camera.ProcessMouseScroll(yoffset);
}