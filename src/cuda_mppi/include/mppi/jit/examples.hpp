#ifndef MPPI_JIT_EXAMPLES_HPP
#define MPPI_JIT_EXAMPLES_HPP

#include <string>

namespace mppi {
namespace jit {
namespace examples {

/**
 * @brief Example pendulum dynamics
 *
 * State: [theta, theta_dot]
 * Control: [torque]
 */
const std::string PENDULUM_DYNAMICS = R"(
struct UserDynamics {
    static constexpr float g = 9.81f;
    static constexpr float m = 1.0f;
    static constexpr float l = 1.0f;

    __device__ void step(const float* x, const float* u, float* x_next, float dt) {
        // x[0] = theta, x[1] = theta_dot
        // u[0] = torque
        float theta = x[0];
        float theta_dot = x[1];
        float torque = u[0];

        // Pendulum dynamics: theta_ddot = (torque - m*g*l*sin(theta)) / (m*l^2)
        float theta_ddot = (torque - m * g * l * sinf(theta)) / (m * l * l);

        // Euler integration
        x_next[0] = theta + theta_dot * dt;
        x_next[1] = theta_dot + theta_ddot * dt;
    }
};
)";

/**
 * @brief Example double integrator dynamics
 *
 * State: [position, velocity]
 * Control: [acceleration]
 */
const std::string DOUBLE_INTEGRATOR_DYNAMICS = R"(
struct UserDynamics {
    __device__ void step(const float* x, const float* u, float* x_next, float dt) {
        // x[0] = position, x[1] = velocity
        // u[0] = acceleration
        float pos = x[0];
        float vel = x[1];
        float acc = u[0];

        // Double integrator dynamics
        x_next[0] = pos + vel * dt;
        x_next[1] = vel + acc * dt;
    }
};
)";

/**
 * @brief Example quadratic cost for pendulum
 *
 * Goal: Stabilize at theta = 0 (upright position)
 */
const std::string PENDULUM_QUADRATIC_COST = R"(
struct UserCost {
    static constexpr float Q_theta = 10.0f;
    static constexpr float Q_theta_dot = 1.0f;
    static constexpr float R_torque = 0.1f;
    static constexpr float Q_terminal_theta = 100.0f;
    static constexpr float Q_terminal_theta_dot = 10.0f;

    __device__ float compute(const float* x, const float* u, int t) {
        // State cost: penalize deviation from upright (theta = 0)
        float theta = x[0];
        float theta_dot = x[1];
        float torque = u[0];

        float state_cost = Q_theta * theta * theta + Q_theta_dot * theta_dot * theta_dot;
        float control_cost = R_torque * torque * torque;

        return state_cost + control_cost;
    }

    __device__ float terminal_cost(const float* x) {
        float theta = x[0];
        float theta_dot = x[1];
        return Q_terminal_theta * theta * theta + Q_terminal_theta_dot * theta_dot * theta_dot;
    }
};
)";

/**
 * @brief Example quadratic cost for double integrator
 *
 * Goal: Reach position = 0
 */
const std::string DOUBLE_INTEGRATOR_QUADRATIC_COST = R"(
struct UserCost {
    static constexpr float Q_pos = 1.0f;
    static constexpr float Q_vel = 0.1f;
    static constexpr float R_acc = 0.01f;
    static constexpr float Q_terminal_pos = 10.0f;
    static constexpr float Q_terminal_vel = 1.0f;

    __device__ float compute(const float* x, const float* u, int t) {
        float pos = x[0];
        float vel = x[1];
        float acc = u[0];

        float state_cost = Q_pos * pos * pos + Q_vel * vel * vel;
        float control_cost = R_acc * acc * acc;

        return state_cost + control_cost;
    }

    __device__ float terminal_cost(const float* x) {
        float pos = x[0];
        float vel = x[1];
        return Q_terminal_pos * pos * pos + Q_terminal_vel * vel * vel;
    }
};
)";

/**
 * @brief Example cart-pole dynamics
 *
 * State: [x, x_dot, theta, theta_dot]
 * Control: [force]
 */
const std::string CARTPOLE_DYNAMICS = R"(
struct UserDynamics {
    static constexpr float g = 9.81f;
    static constexpr float mc = 1.0f;  // cart mass
    static constexpr float mp = 0.1f;  // pole mass
    static constexpr float l = 0.5f;   // pole length

    __device__ void step(const float* x, const float* u, float* x_next, float dt) {
        // x[0] = cart position, x[1] = cart velocity
        // x[2] = pole angle, x[3] = pole angular velocity
        // u[0] = force on cart
        float pos = x[0];
        float vel = x[1];
        float theta = x[2];
        float theta_dot = x[3];
        float force = u[0];

        float sin_theta = sinf(theta);
        float cos_theta = cosf(theta);

        // Cart-pole dynamics
        float temp = (force + mp * l * theta_dot * theta_dot * sin_theta) / (mc + mp);
        float theta_ddot = (g * sin_theta - cos_theta * temp) / (l * (4.0f/3.0f - mp * cos_theta * cos_theta / (mc + mp)));
        float x_ddot = temp - mp * l * theta_ddot * cos_theta / (mc + mp);

        // Euler integration
        x_next[0] = pos + vel * dt;
        x_next[1] = vel + x_ddot * dt;
        x_next[2] = theta + theta_dot * dt;
        x_next[3] = theta_dot + theta_ddot * dt;
    }
};
)";

/**
 * @brief Example quadratic cost for cart-pole
 *
 * Goal: Balance pole upright (theta = 0) at cart position = 0
 */
const std::string CARTPOLE_QUADRATIC_COST = R"(
struct UserCost {
    static constexpr float Q_pos = 1.0f;
    static constexpr float Q_vel = 0.1f;
    static constexpr float Q_theta = 100.0f;
    static constexpr float Q_theta_dot = 1.0f;
    static constexpr float R_force = 0.01f;

    __device__ float compute(const float* x, const float* u, int t) {
        float pos = x[0];
        float vel = x[1];
        float theta = x[2];
        float theta_dot = x[3];
        float force = u[0];

        float state_cost = Q_pos * pos * pos + Q_vel * vel * vel +
                          Q_theta * theta * theta + Q_theta_dot * theta_dot * theta_dot;
        float control_cost = R_force * force * force;

        return state_cost + control_cost;
    }

    __device__ float terminal_cost(const float* x) {
        float pos = x[0];
        float vel = x[1];
        float theta = x[2];
        float theta_dot = x[3];
        return Q_pos * pos * pos + Q_vel * vel * vel +
               Q_theta * theta * theta + Q_theta_dot * theta_dot * theta_dot;
    }
};
)";

} // namespace examples
} // namespace jit
} // namespace mppi

#endif // MPPI_JIT_EXAMPLES_HPP
