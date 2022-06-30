from ROAR.control_module.controller import Controller
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle
from ROAR.utilities_module.data_structures_models import Transform, Location
import numpy as np
import logging
from ROAR.agent_module.agent import Agent
from typing import Tuple
import json
from pathlib import Path
import cvxpy as cp
import scipy
import scipy.signal
import scipy.linalg


class MPCController(Controller):
    def __init__(self, agent, steering_boundary: Tuple[float, float],
                 throttle_boundary: Tuple[float, float], **kwargs):
        super().__init__(agent, **kwargs)
        self.max_speed = self.agent.agent_settings.max_speed
        self.throttle_boundary = throttle_boundary
        self.steering_boundary = steering_boundary
        self.config = json.load(
            Path(agent.agent_settings.mpc_config_file_path).open(mode='r'))
        self.controller = FullMPCController(agent=agent,
                                            throttle_boundary=throttle_boundary,
                                            steering_boundary=steering_boundary,
                                            max_speed=self.max_speed,
                                            config=self.config)
        self.logger = logging.getLogger(__name__)

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> VehicleControl:
        long_control, lat_control = self.controller.run_in_series(next_waypoint=next_waypoint,
                                                           target_speed=kwargs.get("target_speed", self.max_speed))
        
        long_control = float(np.clip(long_control, *self.throttle_boundary))
        lat_control = float(np.clip(lat_control, *self.steering_boundary))

        return VehicleControl(throttle=long_control, steering=lat_control)


class FullMPCController(Controller):
    def __init__(self, agent, config: dict,
                 throttle_boundary: Tuple[float, float],
                 steering_boundary: Tuple[float, float],
                 max_speed: float,
                 dt: float = 0.03, **kwargs):
        super().__init__(agent, **kwargs)
        self.config = config
        self.max_speed = max_speed
        self.throttle_boundary = throttle_boundary
        self.steering_boundary = steering_boundary
        self._dt = dt
        self.A_matrices, self.B_matrices = self.construct_linearized_matrices(max_speed)
        self.last_steer_CMD = 0


    def get_throttle_CMD(self, Fr_x, vx):
        """Calculates the motor input command

        Calculates the motor input command based on the optimal rear tire longitudinal force 
        given by solving the CVXPY problem. The optimal rear tire longitudinal force is then 
        used with the longitudinal dynamics model to solve for the actual motor input command.

        Args:
            Fr_x: Optimal rear tire longitudinal force
            vx: Current longitudinal velocity

        Returns:
            Motor input command
        """

        return (Fr_x + self.config['F_friction'] + self.config['C_d'] * vx**2) / self.config['b_motor']


    def get_steer_CMD(self, Ff_y, beta, r, vx):
        """Calculates the steering input command

        Calculates the steering input command based on the optimal front tire lateral force 
        given by solving the CVXPY problem. The optimal front tire lateral force is then 
        used with the lateral dynamics model to solve for the actual steering input command.

        Args:
            Ff_y: Optimal front tire lateral force
            beta: Current side slip angle of vehicle
            r: Current angular velocity
            vx: Current longitudinal velocity

        Returns:
            steer_cmd
        """

        # Makes sure the argument to the arcsin function on the following line is valid
        arcsin_arg = np.clip(Ff_y / (-self.config['mu'] * self.config['Ff_z']), -1, 1)
        alpha_f = np.tan(np.arcsin(arcsin_arg) / self.config['C']) / self.config['B']
        steer_angle = np.arctan(beta + ((r * self.config['Lf']) / (vx + 10e-1))) - alpha_f 
        steer_cmd = steer_angle / self.config['max_angle']
        self.last_steer_CMD = np.abs(steer_cmd)

        return steer_cmd


    def linearize_around_steer_angle(self, steer_angle_eq, speed_eq):
        """Calculates linearized state space equations

        Linearizes and discretizes the state space equations of the vehicle dynamics model
        around a given equilibrium steering angle and equilibrium speed.

        Args:
            steer_angle_eq: Equilibrium steering angle to linearize around
            speed_eq: Equilibrium vehicle speed to linearize around

        Returns:
            Ad: The linearized and discretized A matrix in the state space model
            Bd: The linearized and discretized B matrix in the state space model
        """

        # Linearize system state equations around a steering angle and 100km/hr
        beta_eq = np.arctan((self.config['Lr'] / self.config['wheelbase']) * np.tan(steer_angle_eq))
        vx_eq = speed_eq * np.cos(beta_eq)
        r_eq = (speed_eq / self.config['Lr']) * np.sin(beta_eq)

        alpha_f = np.arctan(beta_eq + (r_eq * self.config['Lf']) / vx_eq) - steer_angle_eq
        Ff_y_eq = -self.config['mu'] * self.config['Ff_z'] * np.sin(self.config['C'] * np.arctan(self.config['B'] * alpha_f))
        Fr_y_eq = (self.config['Lf'] * Ff_y_eq * np.cos(steer_angle_eq)) / self.config['Lr']

        # Find partial derivative entries for A and B matrices
        a_13 = -(Fr_y_eq + Ff_y_eq * np.cos(steer_angle_eq)) / (self.config['mass'] * vx_eq)
        a_31 = -vx_eq * r_eq

        # Below is a more complex a_13 term that comes from Gonzales dissertation, found to not be needed but may be useful for improving performance
        # a_31 = vx_eq * r_eq \
            # + ((Ff_y_eq * np.cos(steer_angle_eq)) / mass) \
            # * (1 /(1 + (beta_eq + ((r_eq * Lf) / vx_eq))**2))

        Ac = np.array([
            [0, -1, a_13], 
            [0, 0, 0,], 
            [a_31, 0, 0]])
        
        b_11 = np.cos(steer_angle_eq) / (self.config['mass'] * vx_eq)
        b_21 = np.cos(steer_angle_eq) * self.config['Lf'] / self.config['Izz']
        b_31 = -np.sin(steer_angle_eq) / self.config['mass'] 

        Bc = np.array([
            [b_11, 0],
            [b_21, 0],
            [b_31, 1/self.config['mass']]])

        # C and D are just for calling cont2discrete
        Cc = np.zeros((3, 3))
        Dc = np.zeros((3, 2))
        system = (Ac, Bc, Cc, Dc)
        Ad, Bd, Cd, Dd, dt = scipy.signal.cont2discrete(system, self._dt)

        return Ad, Bd


    def construct_linearized_matrices(self, speed_eq):
        """Constructs dicts to hold A and B matrices

        Runs through the array of equilibrium steering angles and calculates
        the linearized A and B matrices for each angle. Those matrices then get
        put into dicts that can be called while CARLA is running. The vehicle dynamics
        change at different steering angles so the optimizer needs to change which
        matrices it is working with or else it cannot solve for optimal vehicle inputs

        Args:
            speed_eq: Equilibrium vehicle speed to linearize around

        Returns:
            A_matrices: Dict holding the linearized and discretized A matrices
            B_matrices: Dict holding the linearized and discretized B matrices
        """
        
        A_matrices = {}
        B_matrices = {}
        for angle in self.config['equilibrium_angles']:
            A, B = self.linearize_around_steer_angle(angle, speed_eq)
            A_matrices.update({angle: A})
            B_matrices.update({angle: B})
        return A_matrices, B_matrices


    def get_linearized_matrices(self, steer_angle):
        """Returns the correct A and B matrices for a given angle

        Args:
            steer_angle: Current steering angle of the car (should be absolute value)

        Returns:
            A and B matrices for the given steering angle
        """

        for i, angle_entry in enumerate(self.config['equilibrium_angles']):
            if i > 0 and steer_angle < angle_entry:
                angle_eq = self.config['equilibrium_angles'][i-1]
                return self.A_matrices.get(angle_eq), self.B_matrices.get(angle_eq)
            elif i == len(self.config['equilibrium_angles']) - 1:
                angle_eq = self.config['equilibrium_angles'][-1]
                return self.A_matrices.get(angle_eq), self.B_matrices.get(angle_eq)
        

    def solve_cftoc(self, target_state, current_state, state_bounds, input_bounds):
        """Solves for optimal vehicle inputs

        Takes in the current vehicle state and the target state that the car should be at, 
        and then solves for the optimal input sequence to reach the target state. Vehicle 
        states are beta, yaw and longitudinal speed for a total of 3 state variables.
        Vehicle inputs are front tire lateral force and rear tire longitudinal force, for a 
        total of 2 input variables.

        Args:
            target_state: The state that the vehicle should be at
            current_state: The current vehicle state
            state_bounds: Bounds that the state variables should not exceed or be under
            input_bounds: Bounds that the inputs should not exceed or be under

        Returns:
            The optimal steering and throttle commands for the current time step
        """

        # Number of future time steps to optimize over
        M = 10
        # Number of state variables, which are beta, yaw and longitudinal speed
        nx = 3
        # Number of input variables, which are front tire lateral force and rear tire longitudinal force
        nu = 2

        # Initialize the array of variables for each time step
        x = cp.Variable((nx, M + 1))
        u = cp.Variable((nu, M))

        # Initialize cost and constraints
        cost = 0
        constr = []

        # Set Initial State
        constr += [x[:, 0] == current_state]

        # Get correct linearized dynamics matrices based on the last steering angle
        A, B = self.get_linearized_matrices(self.last_steer_CMD * self.config['max_angle'])

        for m in range(M):
            
            # Cost function: basically a sum of squares between the current beta, yaw and speed values and the target values
            # The different coefficients come from the magnitude of the state values (i.e. beta is on the range of 0-2 while 
            # longitudinal speed can range from 0-100), and the importance of the state variables as well.
            cost += 10**3 * cp.sum_squares(x[0, m] - target_state[0])
            cost += cp.sum_squares(x[2, m] - target_state[2])
            
            # The cost function value relating to the yaw is removed when the car needs to make a large turn 
            if np.abs(target_state[0]) < np.pi / 20:
                cost += 10**1 * cp.sum_squares(x[1, m] - target_state[1])

            # Constraint for dynamic model
            constr += [x[:, m + 1] == A @ x[:, m] + B @ u[:, m]]
            
            # Constraints for setting bounds on the input values
            constr += [input_bounds[:, 0] <= u[:, m]]
            constr += [input_bounds[:, 1] >= u[:, m]]

            u_delta_limits = np.array(self.config['delta_lim'])
            if m < M - 1:
                # Constraint limiting how much inputs can change between time steps - ensures "smoother" input profiles
                constr += [u[:, m + 1] - u[:, m] <= u_delta_limits, u[:, m + 1] - u[:, m] >= -u_delta_limits]

        # Set terminal cost values
        cost += 10**3 * cp.sum_squares(x[0, M] - target_state[0])
        cost += cp.sum_squares(x[2, M] - target_state[2])

        # Again, the terminal cost function value relating to the yaw is removed when the car needs to make a large turn 
        if np.abs(target_state[0]) < np.pi / 20:
            cost += 10**1 * cp.sum_squares(x[1, M] - target_state[1])

        problem = cp.Problem(cp.Minimize(cost), constr)

        try:
            problem.solve(warm_start=True)
            uOpt = u.value

            # In case optimizer doesnt return any values for u
            if uOpt is None or uOpt.size == 0:
                if np.isnan(uOpt[0][0]):
                    if target_state[0] < 0:
                        Ff_y_cmd = 1000
                    else:
                        Ff_y_cmd = -1000

                if np.isnan(uOpt[0][1]):
                    Fr_x_cmd = 5000

            else:
                Ff_y_cmd = u.value[0, 0]
                Fr_x_cmd = u.value[1, 0]

        except:
            # Sometimes the solver cant find a solution at all for a time step, but input values still need to be returned
            Ff_y_cmd = 0.0
            Fr_x_cmd = 5000
            
        return self.get_throttle_CMD(Fr_x_cmd, current_state[2]), self.get_steer_CMD(Ff_y_cmd, *current_state)


    def run_in_series(self, next_waypoint: Transform, **kwargs) -> float:

        # Calculate current steering angle, beta and vehicle speed. All angles are in radians
        current_steer = self.last_steer_CMD * self.config['max_angle']
        current_beta = np.arctan((self.config['Lr'] / self.config['wheelbase']) * np.tan(current_steer))
        current_speed = Vehicle.get_speed(self.agent.vehicle)

        # Longitudinal speed will be different from the vehicles current speed if beta != 0
        current_vx = current_speed * np.cos(current_beta)

        # Calculate a vector that represent where you are going
        v_begin = self.agent.vehicle.transform.location.to_array()
        current_yaw = np.deg2rad(self.agent.vehicle.transform.rotation.yaw)
        direction_vector = np.array([-np.sin(current_yaw),
                                     0,
                                     -np.cos(current_yaw)])
        v_end = v_begin + direction_vector
        v_vec = np.array([(v_end[0] - v_begin[0]), 0, (v_end[2] - v_begin[2])])

        # Calculate error projection
        w_vec = np.array(
            [
                next_waypoint.location.x - v_begin[0],
                0,
                next_waypoint.location.z - v_begin[2],
            ]
        )

        v_vec_normed = v_vec / np.linalg.norm(v_vec)
        w_vec_normed = w_vec / np.linalg.norm(w_vec)
        error = np.arccos(np.dot(v_vec_normed, w_vec_normed))
        _cross = np.cross(v_vec_normed, w_vec_normed)
        if _cross[1] > 0:
            error *= -1

        # Set the target speed, target beta angle and target longitudinal velocity
        target_speed = self.max_speed
        target_beta = -error
        target_vx = target_speed * np.cos(current_beta)

        # The actual yaw is not needed or important for the optimization problem, as it just needs a "relative" yaw to solve with. 
        # However, the first yaw angle does need to be 0, as the linearized matrices were calculated with yaw = 0.
        # The starting yaw is different for each map: for berkely minor map it is -1.570796 rad (90 degrees), 
        # for easy map it is 0 rad.
        current_yaw = current_yaw - self.config['starting_yaw']

        # Make sure the yaw angle is in [-pi/2, pi/2] or else the optimizer cannot solve for correct steering angle
        current_yaw = np.mod(current_yaw + np.pi / 4, np.pi/2) - np.pi / 4

        # Current optimization setup does not need state bounds, so that's why all state_bounds arrays are 0
        motor_cmd, steer_cmd = self.solve_cftoc(
            target_state=np.array([target_beta, current_yaw, target_vx]), 
            current_state=np.array([current_beta, current_yaw, current_vx]), 
            state_bounds=np.array([[0, 0], [0, 0], [0, 0]]), 
            input_bounds=np.array([[-6000, 6000], [-1000, 10000]]))

        return motor_cmd, steer_cmd
