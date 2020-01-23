#ifdef __EMSCRIPTEN__

// emcc main.cpp -o index.html -s USE_WEBGL2=1 -s FULL_ES3=1 -s USE_GLFW=3 -s WASM=1 -Oz -std=c++14
#include <emscripten.h>
#define GL_GLEXT_PROTOTYPES
#define EGL_EGLEXT_PROTOTYPES
#include <GLFW/glfw3.h>

#else

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#endif

#include <cstdio>
#include <cstdlib>
#include <string>
#include <cassert>
#include <memory>
#include <numeric>
#include <queue>

const char *VERT_CODE = R"(#version 300 es
in vec2 coord;
in vec4 instance;

out vec2 center;
out vec2 delta;
out float size;
out vec2 uv;

uniform float aspectRatio;
uniform float unitSize;

void main()
{
    center = instance.xy;
    delta = instance.zw;
    size = unitSize;
    uv = coord * 2.;

    vec2 resize = vec2(1.);
    if (aspectRatio < 1.) {
        resize.x /= aspectRatio;
    } else {
        resize.y *= aspectRatio;
    }
    resize *= size * 4.;
    gl_Position = vec4((coord + center) * resize, 0., 1.);
}
)";

// based on: https://www.shadertoy.com/view/WdtXz2
const char *FRAG_CODE = R"(#version 300 es
precision highp float;
#define EDGE_FALLOFF 0.0001

in vec2 center;
in vec2 delta;
in float size;
in vec2 uv;

out vec4 outColor;

uniform float runTime;
uniform float maxDeltaMagnitude;

float sphereDistance(vec2 frag, float radius)
{
    return length(frag) - radius;
}

float smoothCombine(float o1, float o2, float smoothness)
{
    float k = clamp((o1 - o2) / smoothness * 0.5 + 0.5, 0.0, 1.0);
    return mix(o1, o2, k) - k * (1.0 - k) * smoothness;
}

float hash12(vec2 x)
{
    return fract(sin(dot(x, vec2(533.59731, 821.49221))) * 4315.212331);
}

vec2 hashedCenter(vec2 steppedFrag)
{
    float hash = hash12(steppedFrag);
    vec2 relativePos = vec2(sin(hash * 532.121 + runTime * 2.0), cos(hash * 532.121 + runTime * 2.0)) * 0.5 + 0.5;
    return steppedFrag + relativePos;
}

float hashedGrid(vec2 frag, float radius, float gridScale)
{
    frag *= gridScale;
    vec2 steppedFrag = floor(frag);
    float minSDF = 99999.0;
    for (float x = -1.0; x <= 1.0; x++)
    {
        for (float y = -1.0; y <= 1.0; y++)
        {
            float sphereDistance = sphereDistance(frag - hashedCenter(steppedFrag + vec2(x, y)), radius);
            minSDF = smoothCombine(minSDF, sphereDistance, 0.3);
        }
    }

    return minSDF / gridScale;
}

void main()
{
    vec2 frag = uv + delta;
    vec2 alaz = delta / maxDeltaMagnitude;
    float alazFactor = abs(length(alaz));

    float object = sphereDistance(frag + alaz * -.2, mix(0.25, 0.2, alazFactor));
    object = smoothCombine(object, sphereDistance(frag + .125 * alaz, mix(0.1, 0.05, alazFactor)), pow(0.5, 1. / alazFactor));
    object = smoothCombine(object, hashedGrid(frag + center, smoothstep(0.6, 0.0, length(frag)) - 0.6, 2.0), mix(.075, 0.2, alazFactor));
    object = smoothCombine(object, hashedGrid(frag + center, smoothstep(0.5, 0.0, length(frag)) - 0.5, 4.0), mix(.05, 0.1, alazFactor));

    float body = smoothstep(EDGE_FALLOFF, -EDGE_FALLOFF, object);
    outColor = vec4(.75, .58, .68, body);
}
)";

class Program {
public:

    Program() : m_statusFlag{0}, m_infoLogLength{0}, m_programId{0} {}

    ~Program() {
        glUseProgram(0);
        glDeleteProgram(m_programId);
    }

    Program &init(const char *vertSource, const char *fragSource) {
        m_programId = glCreateProgram();
        makeShadersFromSources(vertSource, fragSource);

        return *this;
    }

    Program &useProgram() {
        glUseProgram(m_programId);

        return *this;
    }

    Program &setUniform(std::string const &name, float value) {
        int location = getUniformLocation(name);
        glUniform1f(location, value);

        return *this;
    }

    Program &operator=(Program const &) = delete;

    Program(Program const &) = delete;

private:
    int getUniformLocation(std::string const &name) {
        int location = glGetUniformLocation(m_programId, name.c_str());

        if (location == -1)
            printf("Missing Uniform: %s\n", name.c_str());

        return location;
    }

    unsigned int compileAttachShader(const char *source, GLenum shaderEnum) {
        auto shader = glCreateShader(shaderEnum);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);

        glGetShaderiv(shader, GL_COMPILE_STATUS, &m_statusFlag);
        if (!m_statusFlag) {
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &m_infoLogLength);
            std::unique_ptr<char[]> buffer(new char[m_infoLogLength]);
            glGetShaderInfoLog(shader, m_infoLogLength, nullptr, buffer.get());
            printf("%s\n", buffer.get());
        }
        assert(m_statusFlag == 1);

        return shader;
    }

    void checkProgramLog() {
        if (!m_statusFlag) {
            glGetProgramiv(m_programId, GL_INFO_LOG_LENGTH, &m_infoLogLength);
            std::unique_ptr<char[]> buffer(new char[m_infoLogLength]);
            glGetProgramInfoLog(m_programId, m_infoLogLength, nullptr, buffer.get());
            printf("%s\n", buffer.get());
        }
        assert(m_statusFlag == 1);
    }

    void makeShadersFromSources(const char *vertSource, const char *fragSource) {
        glUseProgram(m_programId);

        auto vertShader = compileAttachShader(vertSource, GL_VERTEX_SHADER);
        auto fragShader = compileAttachShader(fragSource, GL_FRAGMENT_SHADER);

        glAttachShader(m_programId, vertShader);
        glAttachShader(m_programId, fragShader);

        glLinkProgram(m_programId);
        glGetProgramiv(m_programId, GL_LINK_STATUS, &m_statusFlag);
        checkProgramLog();

        glValidateProgram(m_programId);
        glGetProgramiv(m_programId, GL_VALIDATE_STATUS, &m_statusFlag);
        checkProgramLog();

        glUseProgram(0);
        glDeleteShader(vertShader);
        glDeleteShader(fragShader);
    }

    unsigned int m_programId;
    int m_statusFlag;
    int m_infoLogLength;
};

#define ALAZ_MAX_DELTA 0.01f

class Alaz {
public:
    Alaz() : m_VAO{}, m_instanceQuadVBO{}, m_VBO{}, m_program{}, m_buffer{0.f, 0.f, 0.f, 0.f}, m_unitSize{.25f} {}

    ~Alaz() {
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDeleteBuffers(1, &m_VBO);
        glDeleteBuffers(1, &m_instanceQuadVBO);

        glBindVertexArray(0);
        glDeleteVertexArrays(1, &m_VAO);
    }

    void init() {
        m_program
                .init(VERT_CODE, FRAG_CODE)
                .useProgram()
                .setUniform("aspectRatio", 1.f)
                .setUniform("maxDeltaMagnitude", ALAZ_MAX_DELTA)
                .setUniform("runTime", 0.f);
        glUseProgram(0);

        glGenVertexArrays(1, &m_VAO);
        glBindVertexArray(m_VAO);

        static const float instanceQuadBuffer[] = {
                -.5f, -.5f,
                .5f, -.5f,
                -.5f, .5f,
                .5f, .5f,
        };
        glGenBuffers(1, &m_instanceQuadVBO);
        glBindBuffer(GL_ARRAY_BUFFER, m_instanceQuadVBO);
        glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), instanceQuadBuffer, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(
                0,
                2,  // x, y coordinates
                GL_FLOAT,
                GL_FALSE,
                0,
                nullptr
        );

        glGenBuffers(1, &m_VBO);
        glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
        glBufferData(GL_ARRAY_BUFFER, m_count * m_bufferLen * sizeof(float), nullptr,
                     GL_STREAM_DRAW);  // buffer orphaning
        glBufferSubData(GL_ARRAY_BUFFER, 0, m_count * m_bufferLen * sizeof(float), m_buffer);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(
                1,
                4,  // center x, y delta x, y and unit size of an instanced quad
                GL_FLOAT,
                GL_FALSE,
                0,
                nullptr
        );

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void move(float deltaX, float deltaY) {
        auto offScreenX = m_unitSize / 2.f;
        auto offScreenY = m_unitSize / 2.f;

        m_buffer[0][0] += deltaX;
        if (m_buffer[0][0] > 1.f + offScreenX) m_buffer[0][0] = -1.f - offScreenX;
        if (m_buffer[0][0] < -1.f - offScreenX) m_buffer[0][0] = 1.f + offScreenX;

        // TODO temporary hack for aspect ratio 1.77 (1. / .5625)
        //  implement a camera system
        m_buffer[0][1] += deltaY;
        if (m_buffer[0][1] > .5625f + offScreenY) m_buffer[0][1] = -.5625f - offScreenY;
        if (m_buffer[0][1] < -.5625f - offScreenY) m_buffer[0][1] = .5625f + offScreenY;

        m_buffer[0][2] = deltaX;
        m_buffer[0][3] = deltaY;


        // TODO delta attrs won't be flushed to zero
        //  if move isn't called with something close to zero
        glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, 4 * sizeof(float), m_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void step(float aspectRatio, float currentTime) {
        m_program
                .useProgram()
                .setUniform("unitSize", m_unitSize)
                .setUniform("aspectRatio", aspectRatio)
                .setUniform("runTime", currentTime);

        glBindVertexArray(m_VAO);

        // TODO instancing may not be supported for some mobile phones
        //  not really using it anyway (yet?)
        glVertexAttribDivisor(0, 0); // coords of the quad (to instamce)
        glVertexAttribDivisor(1, 1); // center, delta, size (one per instanced quad)
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, m_count);

        glBindVertexArray(0);

        glUseProgram(0);
    }

    Alaz &operator=(Alaz const &) = delete;

    Alaz(Alaz const &) = delete;

private:
    unsigned int m_VAO;
    unsigned int m_instanceQuadVBO;
    unsigned int m_VBO;
    static constexpr unsigned int m_count{1};
    static constexpr unsigned int m_bufferLen{4};
    float m_buffer[m_count][m_bufferLen];
    float m_unitSize;
    Program m_program;
};

void clearGLStuff() {
    glClearColor(.8, .8, .8, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
}

typedef struct GlobalStuff {

    GLFWwindow *window{nullptr};
    float aspectRatio;

    int lastConnectedGamePad{0};

    std::deque<float> xQ{0.f};
    std::deque<float> yQ{0.f};
    Alaz alazSystem{};

} GlobalStuff;

GlobalStuff stuff{};

#ifdef __EMSCRIPTEN__

int glfwJoystickIsGamepad(int id) {
    return glfwJoystickPresent(id);
}

#endif

void initLastConnectedGamepad() {
    for (int i = 0; i < 16; i++) {
        if (glfwJoystickIsGamepad(i)) {
            stuff.lastConnectedGamePad = i;
        }
    }
}

void gamepadConnectionCB(int gamePadId, int event) {
    if (event == GLFW_CONNECTED) {
        stuff.lastConnectedGamePad = gamePadId;
    } else if (event == GLFW_DISCONNECTED) {
        initLastConnectedGamepad();
    }
}

void windowSizeChanged(GLFWwindow *, int width, int height) {
    stuff.aspectRatio = (float) width / (float) height;
    glViewport(0, 0, width, height);
}

typedef void (*StepFunction)();

void step() {
    double currentTimePrecise = glfwGetTime();
    auto currentTime = (float) currentTimePrecise;

    // TODO support more input options and fix delta bug in move function
    if (glfwJoystickIsGamepad(stuff.lastConnectedGamePad)) {
        int axesCount = 0;
        auto axes = glfwGetJoystickAxes(stuff.lastConnectedGamePad, &axesCount);
        if (axesCount < 2) {
            stuff.lastConnectedGamePad = -1;
        }

        auto x = ALAZ_MAX_DELTA *
                 ((std::accumulate(stuff.xQ.begin(), stuff.xQ.end(), 0.f) / stuff.xQ.size() + axes[0]) / 2.f);
        auto y = ALAZ_MAX_DELTA *
                 ((std::accumulate(stuff.yQ.begin(), stuff.yQ.end(), 0.f) / stuff.yQ.size() + axes[1]) / 2.f);

        stuff.alazSystem.move(x, -y);

        stuff.xQ.push_front(axes[0]);
        stuff.yQ.push_front(axes[1]);
        if (stuff.xQ.size() > 16) stuff.xQ.pop_back();
        if (stuff.yQ.size() > 16) stuff.yQ.pop_back();
    }

    clearGLStuff();

    stuff.alazSystem.step(stuff.aspectRatio, currentTime);

    glfwSwapBuffers(stuff.window);
    glfwPollEvents();
}

#ifdef __EMSCRIPTEN__

void initGL() {}

void loop(StepFunction step) {
    emscripten_set_main_loop(step, 0, true);
}

#else

void initGL() {
    gladLoadGL();
}

void loop(StepFunction step) {
    while (!glfwWindowShouldClose(stuff.window)) {
        if (glfwGetKey(stuff.window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(stuff.window, true);

        step();
    }
}

#endif

int main() {
    if (!glfwInit())
        exit(EXIT_FAILURE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    stuff.window = glfwCreateWindow(864, 486, "alaz", nullptr, nullptr);
    if (!stuff.window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    stuff.aspectRatio = 864.f / 486.f;
    glfwSetFramebufferSizeCallback(stuff.window, windowSizeChanged);
    glfwMakeContextCurrent(stuff.window);

    initGL();
    printf("OpenGL %s\n", glGetString(GL_VERSION));
    stuff.alazSystem.init();

    glViewport(0, 0, 864, 486);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    initLastConnectedGamepad();
    glfwSetJoystickCallback(gamepadConnectionCB);

    loop(step);

    glfwDestroyWindow(stuff.window);
    exit(EXIT_SUCCESS);
}
