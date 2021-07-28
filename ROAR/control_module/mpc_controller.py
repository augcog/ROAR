# References:
# https://github.com/asap-report/carla/blob/racetrack/PythonClient/racetrack
# /model_predictive_control.py

import logging
import numpy as np
import pandas as pd
import random
import sympy as sym

from pathlib import Path
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
from sympy.tensor.array import derive_by_array
from ROAR.control_module.controller import Controller
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle
from ROAR.utilities_module .data_structures_models import Transform
from ROAR.agent_module.agent import Agent


class _EqualityConstraints(object):
    """Class for storing equality constraints in the MPC."""

    def __init__(self, N, state_vars):
        self.dict = {}
        for symbol in state_vars:
            self.dict[symbol] = N * [None]

    def __getitem__(self, key):
        return self.dict[key]

    def __setitem__(self, key, value):
        self.dict[key] = value


class VehicleMPCController(Controller):
    def __init__(self,
                 agent: Agent,
                 route_file_path: Path,  # read in route
                 target_speed=float("inf"),
                 steps_ahead=10,
                 max_throttle=1,
                 max_steering=1,
                 dt=0.1):
        super().__init__(agent=agent)
        self.logger = logging.getLogger(__name__)
        # Read in route file
        self.track_DF = pd.read_csv(route_file_path, header=None)
        # Fit the route to a curve
        spline_points = 10000
        self.pts_2D = self.track_DF.loc[:, [0, 1]].values
        tck, u = splprep(self.pts_2D.T, u=None, s=2.0, per=1, k=3)
        u_new = np.linspace(u.min(), u.max(), spline_points)
        x_new, y_new = splev(u_new, tck, der=0)
        self.pts_2D = np.c_[x_new, y_new]

        # Modified parm
        self.prev_cte = 0

        self.target_speed = target_speed
        self.state_vars = ('x', 'y', 'v', 'ψ', 'cte', 'eψ')

        self.steps_ahead = steps_ahead
        self.dt = dt

        # Cost function coefficients
        self.cte_coeff = 100  # 100
        self.epsi_coeff = 100  # 100
        self.speed_coeff = 0.4  # 0.2
        self.acc_coeff = 1  # 1
        self.steer_coeff = 0.1  # 0.1
        self.consec_acc_coeff = 50
        self.consec_steer_coeff = 50

        # Front wheel L
        self.Lf = 2.5

        # How the polynomial fitting the desired curve is fitted
        self.steps_poly = 30  # modify to 3 when using 3D data
        self.poly_degree = 3

        # Bounds for the optimizer
        self.bounds = (
                6 * self.steps_ahead * [(None, None)]
                + self.steps_ahead * [(0, max_throttle)]  # throttle bounds
                + self.steps_ahead * [(-max_steering, max_steering)]
            # steer bounds
        )

        # State 0 placeholder
        num_vars = (len(
            self.state_vars) + 2)  # State variables and two actuators
        self.state0 = np.zeros(self.steps_ahead * num_vars)

        # Lambdify and minimize stuff
        self.evaluator = 'numpy'
        self.tolerance = 1
        self.cost_func, self.cost_grad_func, self.constr_funcs = \
            self.get_func_constraints_and_bounds()

        # To keep the previous state
        self.steer = 0
        self.throttle = 0

        self.logger.debug("MPC Controller initiated")
        # self.logger.debug(f"  cost_func:      {self.cost_func}")
        # self.logger.debug(f"  cost_grad_func: {self.cost_grad_func}")
        # self.logger.debug(f"  constr_funcs:   {self.constr_funcs}")

    def run_in_series(self, next_waypoint: Transform) -> VehicleControl:
        super(VehicleMPCController, self).run_in_series(next_waypoint)
        # get vehicle location (x, y)
        # location = self.vehicle.transform.location

        location = self.agent.vehicle.control.location
        x, y = location.x, location.y
        # get vehicle rotation
        # rotation = self.vehicle.transform.rotation
        rotation = self.agent.vehicle.control.rotation
        ψ = rotation.yaw / 180 * np.pi  # transform into radient
        cos_ψ = np.cos(ψ)
        sin_ψ = np.sin(ψ)
        # get vehicle speed
        # v = Vehicle.get_speed(self.vehicle)
        v = Vehicle.get_speed(self.agent.vehicle)
        # get next waypoint location
        wx, wy = next_waypoint.location.x, next_waypoint.location.y
        # debug logging
        # self.logger.debug(f"car location:  ({x}, {y})")
        # self.logger.debug(f"car ψ: {ψ}")
        # self.logger.debug(f"car speed: {v}")
        # self.logger.debug(f"next waypoint: ({wx}, {wy})")

        ### 3D ###
        # get the index of next waypoint
        # waypoint_index = self.get_closest_waypoint_index_3D(location,
        # next_waypoint.location)
        # # find more waypoints index to fit a polynomial
        # waypoint_index_shifted = waypoint_index - 2
        # indeces = waypoint_index_shifted + self.steps_poly * np.arange(
        # self.poly_degree + 1)
        # indeces = indeces % self.track_DF.shape[0]
        # # get waypoints for polynomial fitting
        # pts = np.array([[self.track_DF.iloc[i][0], self.track_DF.iloc[i][
        # 1]] for i in indeces])

        ### 2D ###
        index_2D = self.get_closest_waypoint_index_2D(location,
                                                      next_waypoint.location)
        index_2D_shifted = index_2D - 5
        indeces_2D = index_2D_shifted + self.steps_poly * np.arange(
            self.poly_degree + 1)
        indeces_2D = indeces_2D % self.pts_2D.shape[0]
        pts = self.pts_2D[indeces_2D]

        # self.logger.debug(f'\nwaypoint index:\n  {index_2D}')
        # self.logger.debug(f'\nindeces:\n  {indeces_2D}')

        # transform waypoints from world to car coorinate
        pts_car = VehicleMPCController.transform_into_cars_coordinate_system(
            pts,
            x,
            y,
            cos_ψ,
            sin_ψ
        )
        # fit the polynomial
        poly = np.polyfit(pts_car[:, 0], pts_car[:, 1], self.poly_degree)

        # Debug
        # self.logger.debug(f'\nwaypoint index:\n  {waypoint_index}')
        # self.logger.debug(f'\nindeces:\n  {indeces}')
        # self.logger.debug(f'\npts for poly_fit:\n  {pts}')
        # self.logger.debug(f'\npts_car:\n  {pts_car}')

        ###########

        cte = poly[-1]
        eψ = -np.arctan(poly[-2])

        init = (0, 0, 0, v, cte, eψ, *poly)
        self.state0 = self.get_state0(v, cte, eψ, self.steer, self.throttle,
                                      poly)
        result = self.minimize_cost(self.bounds, self.state0, init)

        # self.steer = -0.6 * cte - 5.5 * (cte - self.prev_cte)
        # self.prev_cte = cte
        # self.throttle = VehicleMPCController.clip_throttle(self.throttle,
        # v, self.target_speed)

        control = VehicleControl()
        if 'success' in result.message:
            self.steer = result.x[-self.steps_ahead]
            self.throttle = result.x[-2 * self.steps_ahead]
        else:
            self.logger.debug('Unsuccessful optimization')

        control.steering = self.steer
        control.throttle = self.throttle

        return control

    def get_func_constraints_and_bounds(self):
        """
        Defines MPC's cost function and constraints.
        """
        # Polynomial coefficients will also be symbolic variables
        poly = self.create_array_of_symbols('poly', self.poly_degree + 1)

        # Initialize the initial state
        x_init = sym.symbols('x_init')
        y_init = sym.symbols('y_init')
        ψ_init = sym.symbols('ψ_init')
        v_init = sym.symbols('v_init')
        cte_init = sym.symbols('cte_init')
        eψ_init = sym.symbols('eψ_init')

        init = (x_init, y_init, ψ_init, v_init, cte_init, eψ_init)

        # State variables
        x = self.create_array_of_symbols('x', self.steps_ahead)
        y = self.create_array_of_symbols('y', self.steps_ahead)
        ψ = self.create_array_of_symbols('ψ', self.steps_ahead)
        v = self.create_array_of_symbols('v', self.steps_ahead)
        cte = self.create_array_of_symbols('cte', self.steps_ahead)
        eψ = self.create_array_of_symbols('eψ', self.steps_ahead)

        # Actuators
        a = self.create_array_of_symbols('a', self.steps_ahead)
        δ = self.create_array_of_symbols('δ', self.steps_ahead)

        vars_ = (
            # Symbolic arrays (but NOT actuators)
            *x, *y, *ψ, *v, *cte, *eψ,

            # Symbolic arrays (actuators)
            *a, *δ,
        )

        cost = 0
        for t in range(self.steps_ahead):
            cost += (
                # Reference state penalties
                    self.cte_coeff * cte[t] ** 2
                    + self.epsi_coeff * eψ[t] ** 2 +
                    + self.speed_coeff * (v[t] - self.target_speed) ** 2

                    # Actuator penalties
                    + self.acc_coeff * a[t] ** 2
                    + self.steer_coeff * δ[t] ** 2
            )

        # Penalty for differences in consecutive actuators
        for t in range(self.steps_ahead - 1):
            cost += (
                    self.consec_acc_coeff * (a[t + 1] - a[t]) ** 2
                    + self.consec_steer_coeff * (δ[t + 1] - δ[t]) ** 2
            )

        # Initialize constraints
        eq_constr = _EqualityConstraints(self.steps_ahead, self.state_vars)
        eq_constr['x'][0] = x[0] - x_init
        eq_constr['y'][0] = y[0] - y_init
        eq_constr['ψ'][0] = ψ[0] - ψ_init
        eq_constr['v'][0] = v[0] - v_init
        eq_constr['cte'][0] = cte[0] - cte_init
        eq_constr['eψ'][0] = eψ[0] - eψ_init

        for t in range(1, self.steps_ahead):
            curve = sum(
                poly[-(i + 1)] * x[t - 1] ** i for i in range(len(poly)))
            # The desired ψ is equal to the derivative of the polynomial
            # curve at
            #  point x[t-1]
            ψdes = sum(poly[-(i + 1)] * i * x[t - 1] ** (i - 1) for i in
                       range(1, len(poly)))

            eq_constr['x'][t] = x[t] - (
                    x[t - 1] + v[t - 1] * sym.cos(ψ[t - 1]) * self.dt)
            eq_constr['y'][t] = y[t] - (
                    y[t - 1] + v[t - 1] * sym.sin(ψ[t - 1]) * self.dt)
            eq_constr['ψ'][t] = ψ[t] - (
                    ψ[t - 1] - v[t - 1] * δ[t - 1] / self.Lf * self.dt)
            eq_constr['v'][t] = v[t] - (v[t - 1] + a[t - 1] * self.dt)
            eq_constr['cte'][t] = cte[t] - (
                    curve - y[t - 1] + v[t - 1] * sym.sin(
                eψ[t - 1]) * self.dt)
            eq_constr['eψ'][t] = eψ[t] - (ψ[t - 1] - ψdes - v[t - 1] * δ[
                t - 1] / self.Lf * self.dt)

        # Generate actual functions from
        cost_func = self.generate_fun(cost, vars_, init, poly)
        cost_grad_func = self.generate_grad(cost, vars_, init, poly)

        constr_funcs = []
        for symbol in self.state_vars:
            for t in range(self.steps_ahead):
                func = self.generate_fun(eq_constr[symbol][t], vars_, init,
                                         poly)
                grad_func = self.generate_grad(eq_constr[symbol][t], vars_,
                                               init, poly)
                constr_funcs.append(
                    {'type': 'eq', 'fun': func, 'jac': grad_func,
                     'args': None},
                )

        return cost_func, cost_grad_func, constr_funcs

    def generate_fun(self, symb_fun, vars_, init, poly):
        """
        Generates a function of the form `fun(x, *args)`
        """
        args = init + poly
        return sym.lambdify((vars_, *args), symb_fun, self.evaluator)

    def generate_grad(self, symb_fun, vars_, init, poly):
        """
        TODO: add comments
        """
        args = init + poly
        return sym.lambdify(
            (vars_, *args),
            derive_by_array(symb_fun, vars_ + args)[:len(vars_)],
            self.evaluator
        )

    def get_state0(self, v, cte, epsi, a, delta, poly):
        a = a or 0
        delta = delta or 0

        x = np.linspace(0, 1, self.steps_ahead)
        y = np.polyval(poly, x)
        psi = 0

        self.state0[:self.steps_ahead] = x
        self.state0[self.steps_ahead:2 * self.steps_ahead] = y
        self.state0[2 * self.steps_ahead:3 * self.steps_ahead] = psi
        self.state0[3 * self.steps_ahead:4 * self.steps_ahead] = v
        self.state0[4 * self.steps_ahead:5 * self.steps_ahead] = cte
        self.state0[5 * self.steps_ahead:6 * self.steps_ahead] = epsi
        self.state0[6 * self.steps_ahead:7 * self.steps_ahead] = a
        self.state0[7 * self.steps_ahead:8 * self.steps_ahead] = delta
        return self.state0

    def minimize_cost(self, bounds, x0, init):
        for constr_func in self.constr_funcs:
            constr_func['args'] = init

        return minimize(
            fun=self.cost_func,
            x0=x0,
            args=init,
            jac=self.cost_grad_func,
            bounds=bounds,
            constraints=self.constr_funcs,
            method='SLSQP',
            tol=self.tolerance,
        )

    def get_closest_waypoint_index_3D(self, car_location, waypoint_location):
        """Get the index of the closest waypoint in self.track_DF
            car_location: current car location
            waypoint_location: next_waypoint
        """
        index = self.track_DF.loc[(self.track_DF[0] == waypoint_location.x)
                                  & (self.track_DF[
                                         1] == waypoint_location.y)].index
        if len(index) > 0:
            return index[0]
        else:
            location_arr = np.array([
                car_location.x,
                car_location.y,
                car_location.z,
            ])
            dists = np.linalg.norm(self.track_DF - location_arr, axis=1)
            return np.argmin(dists)

    def get_closest_waypoint_index_2D(self, car_location, waypoint_location):
        """Get the index of the closest waypoint in self.pts_2D
            Note: it may give wrong index when the route is overlapped
        """
        location_arr = np.array([
            car_location.x,
            car_location.y
        ])
        dists = np.linalg.norm(self.pts_2D - location_arr, axis=1)
        return np.argmin(dists)

    @staticmethod
    def create_array_of_symbols(str_symbol, N):
        return sym.symbols('{symbol}0:{N}'.format(symbol=str_symbol, N=N))

    @staticmethod
    def transform_into_cars_coordinate_system(pts, x, y, cos_ψ, sin_ψ):
        diff = (pts - [x, y])
        pts_car = np.zeros_like(diff)
        pts_car[:, 0] = cos_ψ * diff[:, 0] + sin_ψ * diff[:, 1]
        pts_car[:, 1] = sin_ψ * diff[:, 0] - cos_ψ * diff[:, 1]
        return pts_car

    @staticmethod
    def clip_throttle(throttle, curr_speed, target_speed):
        return np.clip(
            throttle - 0.01 * (curr_speed - target_speed),
            0.4,
            0.9
        )
