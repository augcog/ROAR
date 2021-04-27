from casadi import Opti, sin, cos, tan, vertcat

import numpy as np
import matplotlib.pyplot as plt


def bicycle_robot_model(q, u, L=0.3, dt=0.01):
    """
    Implements the discrete time dynamics of your robot.
    i.e. this function implements F in
    q_{t+1} = F(q_{t}, u_{t})
    dt is the discretization timestep.
    L is the axel-to-axel length of the car.
    q = array of casadi MX.sym symbolic variables [x, y, theta, phi].
    u = array of casadi MX.sym symbolic variables [u1, u2] (velocity and steering inputs).
    Use the casadi sin, cos, tan functions.
    The casadi vertcat or vcat functions may also be useful. Note that to turn a list of
    casadi expressions into a column vector of those expressions, you should use vertcat or
    vcat. vertcat takes as input a variable number of arguments, and vcat takes as input a list.
    Example:
        x = MX.sym('x')
        y = MX.sym('y')
        z = MX.sym('z')
        q = vertcat(x + y, y, z) # makes a 3x1 matrix with entries x + y, y, and z.
        # OR
        q = vcat([x + y, y, z]) # makes a 3x1 matrix with entries x + y, y, and z.
    """
    x, y, theta, sigma = q[0], q[1], q[2], q[3]
    v, w = u[0], u[1]
    x_dot = v * cos(theta)
    y_dot = v * sin(theta)
    theta_dot = v * tan(sigma) / L
    sigma_dot = w

    result = vertcat(x + x_dot * dt,
                     y + y_dot * dt,
                     theta + theta_dot * dt,
                     sigma + sigma_dot * dt)
    return result


def initial_cond(q_start: np.ndarray, q_goal: np.ndarray, n):
    """
    Construct an initial guess for a solution to "warm start" the optimization.
    An easy way to initialize our optimization is to say that our robot will move
    in a straight line in configuration space. Of course, this isn't always possible
    since our dynamics are nonholonomic, but we don't necessarily need our initial
    condition to be feasible. We just want it to be closer to the final trajectory
    than any of the other simple initialization strategies (random, all zero, etc).
    We'll set our initial guess for the inputs to zeros to hopefully bias our solver into
    picking low control inputs
    n is the number of timesteps.
    This function will return two arrays:
        q0 is an array of shape (4, n+1) representing the initial guess for the state
        optimization variables.
        u0 is an array of shape (2, n) representing the initial guess for the state
        optimization variables.

    """
    q0 = np.zeros((4, n + 1))
    u0 = np.zeros((2, n))

    # Your code here.
    q0 = np.linspace(q_start, q_goal, n + 1)
    return q0, u0


def objective_func(q, u, q_goal, Q, R, P):
    """
    Implements the objective function. q is an array of states and u is an array of inputs. Together,
    these two arrays contain all the optimization variables in our problem.
    In particular,
    q has shape (4, N+1), so that each column of q is an array q[:, i] = [q0, q1, q2, q3]
    (i.e. [x, y, theta, phi]), the state at time-step i.
    u has shape (2, N), so that each column of u is an array u[:, i] = [u1, u2], the two control inputs
    (velocity and steering) of the bicycle model.
    This function should create an expression of the form
    sum_{i = 1, ..., N} ((q(i) - q_goal)^T * Q * (q(i) - q_goal) + (u(i)^T * R * u(i)))
    + (q(N+1) - q_goal)^T * P * (q(N+1) - q_goal)
    Note: When dealing with casadi symbolic variables, you can use @ for matrix multiplication,
    and * for standard, numpy-style element-wise (or broadcasted) multiplication.

    """

    n = q.shape[1] - 1
    obj = 0
    for i in range(n):
        qi = q[:, i]
        ui = u[:, i]

        # Define one term of the summation here: ((q(i) - q_goal)^T * Q * (q(i) - q_goal) + (u(i)^T * R * u(i)))
        term = (qi - q_goal).T @ Q @ (qi - q_goal) + (ui.T @ R @ ui)
        obj += term

    q_last = q[:, n]
    # Define the last term here: (q(N+1) - q_goal)^T * P * (q(N+1) - q_goal)
    term_last = (q[:, n] - q_goal).T @ P @ (q[:, n] - q_goal)
    obj += term_last
    return obj


def constraints(q, u, q_lb, q_ub, u_lb, u_ub, obs_list, q_start, q_goal, L=0.3, dt=0.01):
    """
    Constructs a list where each entry is a casadi.MX symbolic expression representing
    a constraint of our optimization problem.
    q has shape (4, N+1), so that each column of q is an array q[:, i] = [q0, q1, q2, q3]
    (i.e. [x, y, theta, phi]), the state at time-step i.
    u has shape (2, N), so that each column of u is an array u[:, i] = [u1, u2], the two control inputs
    (velocity and steering) of the bicycle model.
    q_lb is a size (4,) array [x_lb, y_lb, theta_lb, phi_lb] containing lower bounds for each state variable.
    q_ub is a size (4,) array [x_ub, y_ub, theta_ub, phi_ub] containing upper bounds for each state variable.
    u_lb is a size (2,) array [u1_lb, u2_lb] containing lower bounds for each input.
    u_ub is a size (2,) array [u1_ub, u2_ub] containing upper bounds for each input.
    obs_list is a list of obstacles, where each obstacle is represented by  3-tuple (x, y, r)
            representing the (x, y) center of the obstacle and its radius r. All obstacles are modelled as
            circles.
    q_start is a size (4,) array representing the starting state of the plan.
    q_goal is a size (4,) array representing the goal state of the plan.
    L is the axel-to-axel length of the car.
    dt is the discretization timestep.
    """
    constraints = []

    # State constraints
    constraints.extend([q_lb[0] <= q[0, :], q[0, :] <= q_ub[0]])
    constraints.extend([q_lb[1] <= q[1, :], q[1, :] <= q_ub[1]])
    constraints.extend([q_lb[2] <= q[2, :], q[2, :] <= q_ub[2]])
    constraints.extend([q_lb[3] <= q[3, :], q[3, :] <= q_ub[3]])

    # Input constraints
    constraints.extend([u_lb[0] <= u[0, :], u[0, :] <= u_ub[0]])
    constraints.extend([u_lb[1] <= u[1, :], u[1, :] <= u_ub[1]])

    # Dynamics constraints
    for t in range(q.shape[1] - 1):
        q_t = q[:, t]
        q_tp1 = q[:, t + 1]
        u_t = u[:, t]
        constraints.append(q_tp1 - bicycle_robot_model(q=q_t, u=u_t, L=L, dt=dt) == 0)

    # Obstacle constraints
    for obj in obs_list:
        obj_x, obj_y, obj_r = obj
        for t in range(q.shape[1]):
            constraints.append(
                (q[0, t] - obj_x) ** 2 + (q[1, t] - obj_y) ** 2 >= obj_r ** 2)  # Define the obstacle constraints.

    # Initial and final state constraints
    constraints.append(q[:, 0] == q_start)  # Constraint on start state.
    constraints.append(q[:, -1] == q_goal)  # Constraint on final state.

    return constraints


def plan_to_pose(q_start: np.ndarray, q_goal: np.ndarray, q_lb, q_ub, u_lb, u_ub, obs_list, L=0.3, n=1000, dt=0.01):
    """
    Plans a path from q_start to q_goal.
    q_lb, q_ub are the state lower and upper bounds repspectively.
    u_lb, u_ub are the input lower and upper bounds repspectively.
    obs_list is the list of obstacles, each given as a tuple (x, y, r).
    L is the length of the car.
    n is the number of timesteps.
    dt is the discretization timestep.
    Returns a plan (shape (4, n+1)) of waypoints and a sequence of inputs
    (shape (2, n)) of inputs at each timestep.
    """
    opti = Opti()

    q = opti.variable(4, n + 1)
    u = opti.variable(2, n)

    Q = np.diag([1, 1, 2, 0.1])
    R = 2 * np.diag([1, 0.5])
    P = n * Q

    q0, u0 = initial_cond(q_start, q_goal, n)

    obj = objective_func(q, u, q_goal, Q, R, P)

    opti.minimize(obj)

    opti.subject_to(constraints(q, u, q_lb, q_ub, u_lb, u_ub, obs_list, q_start, q_goal, dt=dt))

    opti.set_initial(q, q0.T)
    opti.set_initial(u, u0)

    ###### CONSTRUCT SOLVER AND SOLVE ######

    opti.solver('ipopt')
    p_opts = {"expand": False}
    s_opts = {"max_iter": 1e4}

    opti.solver('ipopt', p_opts, s_opts)
    sol = opti.solve()

    plan = sol.value(q)
    inputs = sol.value(u)
    return plan, inputs


def plot(plan, inputs, times, q_lb, q_ub, obs_list):
    # Trajectory plot
    ax = plt.subplot(1, 1, 1)
    ax.set_aspect(1)
    ax.set_xlim(q_lb[0], q_ub[0])
    ax.set_ylim(q_lb[1], q_ub[1])

    for obs in obs_list:
        xc, yc, r = obs
        circle = plt.Circle((xc, yc), r, color='black')
        ax.add_artist(circle)

    plan_x = plan[0, :]
    plan_y = plan[1, :]
    ax.plot(plan_x, plan_y, color='green')

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.show()

    # States plot
    plt.plot(times, plan[0, :], label='x')
    plt.plot(times, plan[1, :], label='y')
    plt.plot(times, plan[2, :], label='theta')
    plt.plot(times, plan[3, :], label='phi')

    plt.xlabel('Time (s)')
    plt.legend()
    plt.show()

    # Inputs plot
    plt.plot(times[:-1], inputs[0, :], label='u1')
    plt.plot(times[:-1], inputs[1, :], label='u2')

    plt.xlabel('Time (s)')
    plt.legend()
    plt.show()


def main():
    ###### PROBLEM PARAMS ######

    n = 1000
    dt = 0.01
    L = 0.3

    xy_low = [-5, -1]
    xy_high = [10, 10]
    phi_max = 0.6
    u1_max = 2
    u2_max = 3
    obs_list = [[5, 5, 1], [-3, 4, 1], [4, 2, 2]]
    q_start = np.array([0, 0, 0, 0])
    q_goal = np.array([7, 3, 0, 0])

    ###### SETUP PROBLEM ######

    q_lb = xy_low + [-1000, -phi_max]
    q_ub = xy_high + [1000, phi_max]

    u_lb = [-u1_max, -u2_max]
    u_ub = [u1_max, u2_max]

    ###### CONSTRUCT SOLVER AND SOLVE ######

    plan, inputs = plan_to_pose(q_start, q_goal, q_lb, q_ub, u_lb, u_ub, obs_list, L=L, n=n, dt=dt)

    ###### PLOT ######

    times = np.arange(0.0, (n + 1) * dt, dt)
    print("Final Position:", plan[:4, -1])
    plot(plan, inputs, times, q_lb, q_ub, obs_list)


if __name__ == '__main__':
    main()