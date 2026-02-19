import ast
from curses import init_pair, nonl
from dataclasses import dataclass, field
from encodings.punycode import T
from tracemalloc import stop
from typing import Union

# might need numpy and sympy imports
# import numpy as np
# import sympy as spy
from click import Parameter
import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import (
    SatelliteGeometry,
    SatelliteParameters,
)
from dg_commons.sim.models.obstacles import StaticObstacle
from typing import Sequence

from kiwisolver import Solver
from networkx import center
from shapely import buffer

from pdm4ar.exercises.ex13.discretization import *
from pdm4ar.exercises_def.ex13.goal import DockingTarget, SpaceshipTarget
from pdm4ar.exercises_def.ex13.utils_params import PlanetParams, AsteroidParams

from shapely.geometry import LineString
from dg_commons.sim.scenarios.structures import DgScenario


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "CLARABEL"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 100  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable weight
    lambda_nu_docking: float = 1e5
    weight_p: NDArray = field(default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time

    tr_radius: float = 5  # initial trust region radius
    min_tr_radius: float = 1e-4  # min trust region radius
    max_tr_radius: float = 100  # max trust region radius
    rho_0: float = 0.0  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.0  # div factor trust region update
    beta: float = 3.2  # mult factor trust region update

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    convergence_check: int = (
        2  # switch between convergence check  types: predicted improvement (0), change in state trajectory (1), combo (2)
    )
    stop_crit_pred_improvement: float = 1  # Stopping criteria constant for predicted improvement
    stop_crit_state_diff: float = (
        5e-2  # Stopping criteria constant for max change in norm of state at each discritization step
    )
    stop_crit_norm_pred_improvement: float = (
        5e-3  # Stopping criteria constant for predicted improvement normalized by the cost
    )


class SatellitePlanner:
    """
    Feel free to change anything in this class.
    """

    planets: dict[PlayerName, PlanetParams]
    asteroids: dict[PlayerName, AsteroidParams]
    satellite: SatelliteDyn
    sg: SatelliteGeometry
    sp: SatelliteParameters
    params: SolverParameters
    scenario: DgScenario
    goal: SpaceshipTarget | DockingTarget
    asteroid_centers: list[np.ndarray]  # for each astroid, array with center of [2 x K]

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        asteroids: dict[PlayerName, AsteroidParams],
        sg: SatelliteGeometry,
        sp: SatelliteParameters,
        scenario: DgScenario,
        goal: SpaceshipTarget | DockingTarget,
        init_state: SatelliteState,
    ):
        """
        Pass environment information to the planner.
        """
        # COMMit to Evaluate
        # DEBUGGING FLAGS:
        self.debug_slacks = False
        self.debug_convergence = False
        self.debug_trust_region = False
        self.debug_nonlin_cost = False

        # CONSTRAINT FLAGS:
        self.use_tol = False
        self.use_initial_slack = False

        # ENVIRONMENT INFO
        self.goal = goal
        self.planets = planets
        self.asteroids = asteroids
        self.sg = sg
        self.sp = sp
        self.static_planets = scenario
        self.init_state = init_state
        self.is_docking = False
        self.tf_max = 30  # COMPLETELY RANDOM, NO INFO ON MAX VELOCITY
        self.max_vel = self.sp.m_v
        # Some parameters don't need to be updated even when replanning.
        # Used in convexification
        self.on_init = True

        # FIND BOUNDARIES OF THE ENVIRONMENT
        for obs in self.static_planets.static_obstacles:
            if isinstance(obs.shape, LineString):
                self.minx, self.miny, self.maxx, self.maxy = obs.shape.bounds
                break

        # INPUT CONVEX CONSTRAINTS
        self.u_min = self.sp.F_limits[0]
        self.u_max = self.sp.F_limits[1]

        # CREATE SATELLITE BUFFER
        max_vert = max(self.sg.l_f, self.sg.l_r)
        max_hor = self.sg.l_f + self.sg.l_c
        self.r_buffer = max(np.sqrt((self.sg.w_panel + self.sg.w_half) ** 2 + max_vert**2), max_hor)

        # DOCKING PARAMETERS
        if isinstance(goal, DockingTarget):
            self.dock_A, self.dock_B, self.dock_C, self.dock_A1, self.dock_A2, self.dock_p = (
                goal.get_landing_constraint_points()
            )
            self.docking_center = (np.array(self.dock_A1) + np.array(self.dock_A2)) / 2.0
            goal_vec = np.array([self.goal.target.x, self.goal.target.y])
            center_to_goal = goal_vec - self.docking_center
            init = np.array([self.init_state.x, self.init_state.y])
            goal_to_init = init - goal_vec
            if np.dot(center_to_goal, goal_to_init) <= 0:
                self.is_docking = True

            if self.is_docking:
                self.dock_A, self.dock_B, self.dock_C, self.dock_A1, self.dock_A2, self.dock_p = (
                    goal.get_landing_constraint_points()
                )
                self.docking_center = (np.array(self.dock_A1) + np.array(self.dock_A2)) / 2.0
                self.docking_radius = np.linalg.norm(np.array(self.dock_A1) - np.array(self.dock_A2)) / 2.0

                center_to_goal = goal_vec - self.docking_center
                self.center_to_goal_dist = np.linalg.norm(center_to_goal)
                self.c_to_g_norm = center_to_goal / self.center_to_goal_dist

                self.docking_goal = self.docking_center + 2 * center_to_goal
                self.docking_psi = self.goal.target.psi

                # COULD TRY TO IMPLEMENT TIME VARYING RADIUS
                dock_planet = PlanetParams(
                    center=self.docking_center.tolist(),
                    radius=float(self.docking_radius),
                )
                self.planets[PlayerName("Dock")] = dock_planet

        # Solver Parameters
        self.params = SolverParameters()
        if self.is_docking:
            # TIME STEP ACTIVATION OF CONE DOCKING CONSTRAINT
            self.K_docking_start = int(self.params.K - 10)
        self.tr_radius = self.params.tr_radius

        # Satellite Dynamics
        self.satellite = SatelliteDyn(self.sg, self.sp)

        # Discretization Method
        self.integrator = FirstOrderHold(self.satellite, self.params.K, self.params.N_sub)

        # SOLVER VARIABLES
        self.variables = self._get_variables()

        # SOLVER PARAMETERS
        self.problem_parameters = self._get_problem_parameters()

        # SOLVER CONSTRAINTS
        self.constraints = self._get_constraints()

        # SOLVER OBJECTIVE FUNCTION
        self.objective = self._get_objective()

        # SOLVER PROBLEM
        self.problem = cvx.Problem(self.objective, self.constraints)

    def compute_trajectory(
        self, init_state: SatelliteState, goal_state: DynObstacleState
    ) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Compute a trajectory from init_state to goal_state.
        For SCvx:
            initial guess interpolation
            while stopping criterion not satisfied
                convexify
                solve convex sub problem
                update trust region
                update stopping criterion
        """
        # SCvx pipeline
        self.init_state = init_state
        self.goal_state = goal_state

        # INITIAL GUESS WITH STRAIGHT LINE INTERPOLATION
        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        for _ in range(self.params.max_iterations):
            ######## PERFORM CONVEXIFICATION ########
            self._convexification()

            ######## SOLVE THE CONVEX PROBLEM ########
            try:
                opt_cost = self.problem.solve(verbose=self.params.verbose_solver, solver=self.params.solver)

            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")

            ######## UPDATE TRUST REGION ########
            accept_soln = self._update_trust_region(opt_cost)

            ######## CHECK CONVERGENCE ########
            if self._check_convergence(opt_cost):
                break

            ######## UPDATE REFERENCE TRAJECTORY ########
            if accept_soln:
                # print("soln accepted")
                self.X_bar = self.variables["X"].value
                self.U_bar = self.variables["U"].value
                self.p_bar = self.variables["p"].value

        if self.debug_slacks:
            dyn_slacks = self.variables["v_d"].value
            obs_slacks = self.variables["v_p"].value
            if len(self.asteroids) > 0:
                astr_slacks = self.variables["v_a"].value

            x_slacks = dyn_slacks[0, :]
            y_slacks = dyn_slacks[1, :]
            dir_slacks = dyn_slacks[2, :]
            v_x_slacks = dyn_slacks[3, :]
            v_y_slacks = dyn_slacks[4, :]
            ddir_slacks = dyn_slacks[5, :]

            try:

                assert np.any(dir_slacks < 1e-5), f"Error with direction slack : {dir_slacks}"
                assert np.any(x_slacks < 1e-5), f"Error with x slack : {x_slacks}"
                assert np.any(y_slacks < 1e-5), f"Error with y slack : {y_slacks}"
                assert np.any(v_x_slacks < 1e-5), f"Error with direction slack : {v_x_slacks}"
                assert np.any(v_y_slacks < 1e-5), f"Error with direction slack : {v_y_slacks}"
                assert np.any(ddir_slacks < 1e-5), f"Error with direction slack : {ddir_slacks}"
                assert np.any(obs_slacks < 1e-5), f"Error with obst slack : {obs_slacks}"
                if len(self.asteroids) > 0:
                    assert np.any(astr_slacks < 1e-5), f"Error with obst slack : {astr_slacks}"
            except Exception as e:
                print(e)

        # EXTRACT THE COMPUTED TRAJECTORY AND COMMANDS
        mycmds, mystates = self._extract_seq_from_array(self.X_bar, self.U_bar, self.p_bar)

        return mycmds, mystates

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """

        # In this baseline implementation, just a linear interpolation between initial and end variables.
        # p is set to be a maximum time, which has been set at random.

        # In next implementations, initial guess could be physically meaningful and for not on_init calls,
        # could implement chopping of initial trajectory which was not followed properly.

        K = self.params.K

        # Interpolating values in the state trajectory
        X = np.empty((self.satellite.n_x, K))
        x_init = self.init_state
        x_goal = self.goal_state
        minx = self.minx + self.r_buffer
        maxx = self.maxx - self.r_buffer
        miny = self.miny + self.r_buffer
        maxy = self.maxy - self.r_buffer

        if self.is_docking:
            feasibility_offset = 0
            docking_goal_x_valid = np.clip(self.docking_goal[0], minx, maxx)
            docking_goal_y_valid = np.clip(self.docking_goal[1], miny, maxy)
            docking_goal = np.array([docking_goal_x_valid, docking_goal_y_valid])
            print(f"Docking goal{docking_goal}")

            x_1 = np.linspace(x_init.x, docking_goal[0], self.K_docking_start - feasibility_offset)
            y_1 = np.linspace(x_init.y, docking_goal[1], self.K_docking_start - feasibility_offset)

            x_2 = np.linspace(docking_goal[0], x_goal.x, K - self.K_docking_start + feasibility_offset)
            y_2 = np.linspace(docking_goal[1], x_goal.y, K - self.K_docking_start + feasibility_offset)

            x = np.concatenate([x_1, x_2])
            y = np.concatenate([y_1, y_2])
        else:
            x = np.linspace(x_init.x, x_goal.x, K)
            y = np.linspace(x_init.y, x_goal.y, K)

        psi = np.linspace(x_init.psi, x_goal.psi, K)
        psi = np.mod(psi, 2 * np.pi)

        vx = np.linspace(x_init.vx, x_goal.vx, K)
        vy = np.linspace(x_init.vy, x_goal.vy, K)
        dpsi = np.linspace(x_init.dpsi, x_goal.dpsi, K)

        u_init = np.zeros((2, K))

        X = np.vstack((x, y, psi, vx, vy, dpsi))
        U = u_init
        goal = np.array([x_goal.x, x_goal.y])
        init = np.array([x_init.x, x_init.y])
        init_to_goal = goal - init
        dist = np.linalg.norm(init_to_goal)
        tf_min = 1.1 * dist / self.max_vel
        p = np.full(self.satellite.n_p, tf_min)

        return X, U, p

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """

        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p

        n_planets = len(self.planets)
        n_asteroids = len(self.asteroids)

        K = self.params.K

        variables = {
            "X": cvx.Variable((n_x, K)),  # [X x K]: State variables for each timestep
            "U": cvx.Variable((n_u, K)),  # [U x K]: Input variables for each timestep
            "p": cvx.Variable(n_p),  # [P]: Parameters (final time only?)
            "v_d": cvx.Variable((n_x, K)),  # [X x K]: Slack variables for dynamics
            "v_p": cvx.Variable((n_planets, K)),  # [#Planets x K]: Slack variables for planet constraints
        }

        if self.is_docking:
            variables["v_dock"] = cvx.Variable((2, K - self.K_docking_start))
        if self.use_initial_slack:
            variables["v_ic"] = cvx.Variable(n_x)  # [X]: Slack variables for initial condition

        if n_asteroids != 0:
            variables["v_a"] = cvx.Variable((n_asteroids, K))

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        #   - dynamics: A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar, E
        #   - convex state space: x_min, x_max, y_min, y_max from boundaries (see config yaml) (no limits on velocities or rotation)
        #   - convex input space: u_min, u_max
        #   - convexified inequality constraints: Ck, Dk, Gk, r'
        #   - initial state: initial state
        #   - final sate: final state
        #   - sum of change in ...: X_bar, U_bar, P_bar

        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p

        n_planets = len(self.planets)
        n_asteroids = len(self.asteroids)

        K = self.params.K

        problem_parameters = {
            # Dynamics for each discretization step: columns of each matrix are stacked vertically, then stacked horizontally
            "A_bar": cvx.Parameter((n_x * n_x, K - 1)),
            "B_plus_bar": cvx.Parameter((n_x * n_u, K - 1)),
            "B_minus_bar": cvx.Parameter((n_x * n_u, K - 1)),
            "F_bar": cvx.Parameter((n_x * n_p, K - 1)),
            "r_bar": cvx.Parameter((n_x, K - 1)),
            "E": cvx.Parameter((n_x * n_x, K - 1)),
            # Convex constraints (state and input space)
            "x_min": cvx.Parameter((2, K)),
            "x_max": cvx.Parameter((2, K)),
            "u_min": cvx.Parameter((n_u, K)),
            "u_max": cvx.Parameter((n_u, K)),
            # Non-convex constraints for each discritization: columns of each matrix are stacked vertically, then stacked horizontally
            "C_p_bar": cvx.Parameter((n_planets * 2, K)),
            "r_prime_p_bar": cvx.Parameter((n_planets, K)),
            # Boundary constraints
            "x_0": cvx.Parameter(n_x),
            "x_f": cvx.Parameter(n_x),
            # Trust region for each discretization step stacked vertically
            "X_bar": cvx.Parameter((n_x, K)),
            "U_bar": cvx.Parameter((n_u, K)),
            "p_bar": cvx.Parameter((n_p)),
            "tr_radius": cvx.Parameter(),
        }

        if n_asteroids != 0:
            problem_parameters.update(
                {
                    "C_a_bar": cvx.Parameter((n_asteroids * 2, K)),
                    "G_a_bar": cvx.Parameter((n_asteroids * n_p, K)),
                    "r_prime_a_bar": cvx.Parameter((n_asteroids, K)),
                }
            )

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """
        #   - dynamics
        #   - convex state space (xmin/xmax)
        #   - convex input space (umin/umax)
        #   - convexified inequality constraints (planets)
        #   - initial state (no slack for now)
        #   - initial inputs (ONLY zero on episode_init; otherwise, not constrained)
        #   - final state (ineqality for tolerance)
        #   - final inputs (zero)
        #   - sum of change in trajectory, control inputs, and parameters (1-norm) is below tolerance eta

        # --------EXTRACTING TO IMPROVE READABILITY----------------#
        vrs = self.variables
        prb_prs = self.problem_parameters
        prs = self.params
        satellite = self.satellite

        X = vrs["X"]
        U = vrs["U"]
        p = vrs["p"]

        x_0 = prb_prs["x_0"]
        x_f = prb_prs["x_f"]
        x_max = prb_prs["x_max"]
        x_min = prb_prs["x_min"]

        u_min = prb_prs["u_min"]
        u_max = prb_prs["u_max"]

        x_bar = prb_prs["X_bar"]
        u_bar = prb_prs["U_bar"]
        p_bar = prb_prs["p_bar"]

        K = prs.K
        if self.use_tol:
            p_tol = self.goal.pos_tol
            o_tol = self.goal.dir_tol
            v_tol = self.goal.vel_tol

        if self.use_initial_slack:
            init_slack = vrs["v_ic"]

        constraints = [
            # ----------------BOUNDARY STATE CONSTRAINTS----------------#
            X[0, 1:-1] <= x_max[0, 1:-1],
            X[0, 1:-1] >= x_min[0, 1:-1],
            X[1, 1:-1] <= x_max[1, 1:-1],
            X[1, 1:-1] >= x_min[1, 1:-1],
            # ----------------INIT AND FINAL + BOUNDARY INPUT CONSTRAINTS----------------#
            U[:, 0] == np.array([0, 0]),
            U[:, K - 1] == np.array([0, 0]),
            U[:, 1:-1] <= u_max[:, 1:-1],
            U[:, 1:-1] >= u_min[:, 1:-1],
            # ----------------FINAL TIME CONSTRAINTS----------------#
            p >= 0,
        ]

        # --------------------INITIAL STATE CONSTRAINTS----------------#
        if self.use_initial_slack:
            constraints.append(X[:, 0] == x_0 + init_slack)
        else:
            constraints.append(X[:, 0] == x_0)
        ## --------------------FINAL STATE CONSTRAINTS----------------#
        if self.use_tol:
            constraints.append(cvx.norm(X[0:2, K - 1] - x_f[0:2], 2) <= p_tol)
            constraints.append(cvx.abs(X[2, K - 1] - x_f[2]) <= o_tol)
            constraints.append(cvx.norm(X[3:5, K - 1] - x_f[3:5], 2) <= v_tol)
        else:
            constraints.append(X[0:5, K - 1] == x_f[0:5])

        # --------------------TRUST CONSTRAINTS----------------#
        for k in range(K):
            constraints.append(
                cvx.norm(X[:, k] - x_bar[:, k], "inf")
                + cvx.norm(U[:, k] - u_bar[:, k], "inf")
                + cvx.norm(p - p_bar, "inf")
                <= prb_prs["tr_radius"]
            )

        # ----------------DYNAMICS CONSTRAINTS----------------#
        for constraint in self._get_dynamics_constraints(satellite, vrs, prb_prs, X, U, p, K):
            constraints.append(constraint)

        # ----------------OBSTACLES CONSTRAINTS----------------#
        for constraint in self._get_obstacles_constraints(vrs, prb_prs, X, p, K):
            constraints.append(constraint)

        if self.is_docking:
            for constraint in self._get_docking_constraints(X=X, vrs=vrs, K_docking=self.K_docking_start, K=K):
                constraints.append(constraint)

        return constraints

    def _get_docking_constraints(self, X: cvx.Variable, vrs: dict, K_docking: int, K: int):

        A = np.array(self.dock_A)
        B = np.array(self.dock_B)
        C = np.array(self.dock_C)
        v_dock = vrs["v_dock"]

        # goal = np.array([self.goal_state.x, self.goal_state.y])
        A_to_B = np.array(B - A)
        A_to_C = np.array(C - A)

        normal_AtoB_plane = np.array([A_to_B[1], -A_to_B[0]])

        normal_AtoC_plane = np.array([-A_to_C[1], A_to_C[0]])

        constraints = []
        for k in range(K_docking, K):
            # SHIFT PLANE FROM A TO DOCKING LINE CENTER FOR IMPROVED FEASIBILITY
            dist_vec = X[0:2, k] - A

            constraints.append(cvx.vdot(dist_vec, normal_AtoB_plane) >= v_dock[0, k - K_docking])
            constraints.append(cvx.vdot(dist_vec, normal_AtoC_plane) >= v_dock[1, k - K_docking])
        return constraints

    def _get_dynamics_constraints(
        self,
        satellite: SatelliteDyn,
        vrs: dict,
        prb_prs: dict,
        X: cvx.Variable,
        U: cvx.Variable,
        p: cvx.Variable,
        K: int,
    ) -> list[cvx.Constraint]:
        """
        Define dynamics constraints for SCvx.
        """
        dyn_constraints = []
        n_x = satellite.n_x
        n_u = satellite.n_u
        n_p = satellite.n_p

        A_bar = prb_prs["A_bar"]
        B_plus_bar = prb_prs["B_plus_bar"]
        B_minus_bar = prb_prs["B_minus_bar"]
        F_bar = prb_prs["F_bar"]
        r_bar = prb_prs["r_bar"]

        v_d = vrs["v_d"]
        # -------PURE DYNAMICS CONSTRAINTS-------#
        for k in range(K - 1):
            # Extracting the relevant matrices for timestep k
            A_k = cvx.reshape(A_bar[:, k], (n_x, n_x), order="F")
            B_plus_k = cvx.reshape(B_plus_bar[:, k], (n_x, n_u), order="F")
            B_minus_k = cvx.reshape(B_minus_bar[:, k], (n_x, n_u), order="F")
            F_k = cvx.reshape(F_bar[:, k], (n_x, n_p), order="F")
            r_k = r_bar[:, k]

            # Dynamics constraint with slack variable
            dyn_constraints.append(
                X[:, k + 1] == A_k @ X[:, k] + B_plus_k @ U[:, k + 1] + B_minus_k @ U[:, k] + F_k @ p + r_k + v_d[:, k]
            )

        return dyn_constraints

    def _get_obstacles_constraints(
        self,
        vrs: dict,
        prb_prs: dict,
        X: cvx.Variable,
        p: cvx.Variable,
        K: int,
    ) -> list[cvx.Constraint]:
        """
        Define obstacles constraints for SCvx.
        """
        obs_constraints = []
        n_planets = len(self.planets)
        C_p = prb_prs["C_p_bar"]
        r_p = prb_prs["r_prime_p_bar"]

        v_p = vrs["v_p"]
        v_a = vrs.get("v_a", None)

        # --------PLANETS CONSTRAINTS-------#
        for j in range(n_planets):
            for k in range(K):
                # rows j and j+n_planets correspond to x and y parts
                row_x = C_p[j, k] * X[0, k] + C_p[j + n_planets, k] * X[1, k]
                obs_constraints.append(row_x + r_p[j, k] <= v_p[j, k])

        # --------ASTEROIDS CONSTRAINTS-------#
        if v_a is not None:
            n_a = len(self.asteroids)
            C_a = prb_prs["C_a_bar"]
            G_a = prb_prs["G_a_bar"]
            r_a = prb_prs["r_prime_a_bar"]

            for j in range(n_a):
                for k in range(K):
                    # rows j and j+n_asteroids correspond to x and y parts
                    row_x = C_a[j, k] * X[0, k] + C_a[j + n_a, k] * X[1, k]
                    row_p = G_a[j, k] * p[0]
                    obs_constraints.append(row_x + row_p + r_a[j, k] <= v_a[j, k])

        return obs_constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # DONE: define objective dependent on:
        #   - time optimality (final time minimization)
        #   - slack variable penalization (dynamics and constraints (asteroids and planets))
        #   - initial mismatch penalization
        #   - usage of thrusters along the whole trajectory

        vrs = self.variables
        K = self.params.K

        # ----------------TIME RELATED PENALTIES----------------#
        p = vrs["p"]  # final time
        w_p = self.params.weight_p  # weight for final time
        time_penalty = w_p @ p

        # ----------------SLACK VARIABLES PENALTIES----------------#
        v_d = vrs["v_d"]  # dynamics slack
        v_p = vrs["v_p"]  # planet constraints slack

        # handle asteroid slack if asteroids exist
        v_a = vrs.get("v_a", None)
        lambda_nu = self.params.lambda_nu  # slack variable weight (same for all?)
        lambda_nu_docking = self.params.lambda_nu_docking

        slack_penalty = lambda_nu * sum(
            cvx.norm(v_d[:, k], 1) + (cvx.norm(v_a[:, k], 1) if v_a is not None else 0) for k in range(K)
        )

        if self.is_docking:
            v_dock = vrs["v_dock"]
            slack_penalty += lambda_nu * sum(cvx.norm(v_p[0:-1, k], 1) for k in range(K))
            slack_penalty += lambda_nu_docking * sum(cvx.norm(v_p[-1, k], 1) for k in range(self.K_docking_start))
            slack_penalty += lambda_nu * sum(cvx.norm(v_dock[:, k], 1) for k in range(K - self.K_docking_start))
        else:
            slack_penalty += lambda_nu * sum(cvx.norm(v_p[:, k], 1) for k in range(K))

        if self.use_initial_slack:
            v_ic = vrs["v_ic"]
            slack_penalty += cvx.norm(v_ic, 1)

        # ----------------THRUSTERS PENALTIES----------------#
        U = vrs["U"]  # input variables
        ctrl_penalty = sum([cvx.norm(U[:, k], 1) for k in range(K)])

        # Objective function to minimize
        objective = time_penalty + slack_penalty + ctrl_penalty
        return cvx.Minimize(objective)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        n_x = self.satellite.n_x
        K = self.params.K
        n_asteroids = len(self.asteroids)

        # FOH
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(
            self.X_bar, self.U_bar, self.p_bar
        )

        # Planet Constraints
        C_p_bar, r_prime_p_bar = self._calculate_linear_constr_planets()

        # NOTE: be aware that the matrices returned by calculate_discretization are flattened in F order (this way affect your code later when you use them)

        # self.problem_parameters is a dict with keys as in _get_problem_parameters

        # Dynamics parameters
        self.problem_parameters["A_bar"].value = A_bar
        self.problem_parameters["B_plus_bar"].value = B_plus_bar
        self.problem_parameters["B_minus_bar"].value = B_minus_bar
        self.problem_parameters["F_bar"].value = F_bar
        self.problem_parameters["r_bar"].value = r_bar

        # Non-convex constraints: columns of each matrix are stacked vertically, then stacked horizontally
        self.problem_parameters["C_p_bar"].value = C_p_bar
        self.problem_parameters["r_prime_p_bar"].value = r_prime_p_bar

        # Boundary init constraint : needs to be updated
        self.problem_parameters["x_0"].value = np.array(
            [
                self.init_state.x,
                self.init_state.y,
                self.init_state.psi,
                self.init_state.vx,
                self.init_state.vy,
                self.init_state.dpsi,
            ]
        )

        # Trust region constraints params
        self.problem_parameters["X_bar"].value = self.X_bar
        self.problem_parameters["U_bar"].value = self.U_bar
        self.problem_parameters["p_bar"].value = self.p_bar
        self.problem_parameters["tr_radius"].value = self.tr_radius

        if n_asteroids != 0:
            C_a_bar, r_prime_a_bar, G_a_bar = self._calculate_linear_constr_asteroids()
            self.problem_parameters["C_a_bar"].value = C_a_bar
            self.problem_parameters["r_prime_a_bar"].value = r_prime_a_bar
            self.problem_parameters["G_a_bar"].value = G_a_bar

        if self.on_init:

            # E matrix is initialized as an identity matrix
            eye_flat = np.eye(n_x).flatten(order="F")
            eye_flat = eye_flat.reshape(-1, 1)  # from (n,) → (n,1)
            E = np.repeat(eye_flat, K - 1, axis=1)
            self.problem_parameters["E"].value = E

            # Convex constraints:
            x_min = np.array([[self.minx, self.miny]]).T.repeat(K, axis=1)
            x_max = np.array([[self.maxx, self.maxy]]).T.repeat(K, axis=1)

            u_min = np.array([[self.u_min, self.u_min]]).T.repeat(K, axis=1)
            u_max = np.array([[self.u_max, self.u_max]]).T.repeat(K, axis=1)

            r_buffer = self.r_buffer
            self.problem_parameters["x_min"].value = x_min + r_buffer
            self.problem_parameters["x_max"].value = x_max - r_buffer
            self.problem_parameters["u_min"].value = u_min
            self.problem_parameters["u_max"].value = u_max

            # Boundary goal constraint
            self.problem_parameters["x_f"].value = np.array(
                [
                    self.goal_state.x,
                    self.goal_state.y,
                    self.goal_state.psi,
                    self.goal_state.vx,
                    self.goal_state.vy,
                    self.goal_state.dpsi,
                ]
            )

            self.on_init = False

    def _check_convergence(self, opt_cost) -> bool:
        """
        Check convergence of SCvx.
        """
        # DONE: Check convergence based on...
        #   - denominator of rho is small
        #   - goal reached (both state and parameter don't change)
        #   - denominator of rho is less than scaled relative magnitude change or goal reached

        # convergence check based on the denominator of rho (predicted improvement)
        J_ref = self._nonlinear_cost(self.X_bar, self.U_bar, self.p_bar)
        L_opt = opt_cost
        rho_denom_small = J_ref - L_opt < SolverParameters.stop_crit_pred_improvement

        # convergence check based on magnitude of change in time and state
        diff = self.X_bar - self.variables["X"].value
        raw_dir_diff = np.abs(diff[2, :]) % (2 * np.pi)
        dir_diff = np.minimum(raw_dir_diff, 2 * np.pi - raw_dir_diff)
        diff[2, :] = dir_diff
        diff_norms = np.linalg.norm(diff, axis=0, ord=2)
        diff_p = np.linalg.norm(self.p_bar - self.variables["p"].value)

        if self.debug_convergence:
            print(f"Norms for convergence: {diff_norms.max()}")
            idx = np.argmax(diff_norms)
            print(f"Norms for convergence : {diff[:, idx]} at timestep {idx}")
            print(f"State at that index: {self.X_bar[:, idx]}")

        diff_small = diff_p + diff_norms.max() < SolverParameters.stop_crit_state_diff

        # convergence check based on normalized predicted improvement
        rho_denom_small_norm = J_ref - L_opt < SolverParameters.stop_crit_norm_pred_improvement * np.abs(J_ref)
        combo_wombo_check = diff_small or rho_denom_small_norm

        match (SolverParameters.convergence_check):
            case 0:
                return rho_denom_small
            case 1:
                return diff_small
            case _:
                return combo_wombo_check

    def _update_trust_region(self, opt_cost) -> bool:
        """
        Update trust region radius.

        Returns True if problem solution should be accepted
        """
        # DONE: update eta based on rho
        #   - calculate rho (based on defects)
        #   - update eta based on where rho falls in comparison to rho_0, rho_1, rho_2 (see pg35 of paper)

        accept = True  # flag indicating if solution is accepted (T) or rejected (F)

        J_ref = float(self._nonlinear_cost(self.X_bar, self.U_bar, self.p_bar))
        J_opt = float(
            self._nonlinear_cost(self.variables["X"].value, self.variables["U"].value, self.variables["p"].value)
        )
        L_opt = float(opt_cost)

        actual_improvement = J_ref - J_opt
        predicted_improvement = J_ref - L_opt

        rho = actual_improvement / predicted_improvement

        # Update trust region
        if rho < SolverParameters.rho_0:
            # inaccurate: shrink trust region and reject solution
            self.tr_radius = max(SolverParameters.min_tr_radius, self.tr_radius / SolverParameters.alpha)
            accept = False
        elif (SolverParameters.rho_0 <= rho) and (rho < SolverParameters.rho_1):
            # a bit inaccurate: shrink trust region and accept solution
            self.tr_radius = max(SolverParameters.min_tr_radius, self.tr_radius / SolverParameters.alpha)
            accept = True
        elif (SolverParameters.rho_1 <= rho) and (rho < SolverParameters.rho_2):
            # quite accurate: keep trust region and accept solution
            accept = True
        else:
            # conservative: expand trust region and accept solution
            self.tr_radius = min(SolverParameters.max_tr_radius, self.tr_radius * SolverParameters.beta)
            accept = True

        if self.debug_trust_region:
            print(f"L_opt: {L_opt}")
            print(f"J_opt: {J_opt}")
            print(f"rho: {rho}")
            print(
                f"Actual improvement: {actual_improvement}\nPredicted improvement: {predicted_improvement}\nTrust Region: {self.tr_radius}"
            )

        # Return flag indicating if solution is accepted (T) or rejected (F)
        return accept

    def _nonlinear_cost(self, x: np.ndarray, u: np.ndarray, p: np.ndarray) -> float:
        """
        Calculate the non-linear cost for a given solution x, u, p
        """
        running_cost = np.sum(np.abs(u))
        nonlin_constraint_costs = self._P_k_vec(self._defects(x, u, p), self._nonlinear_pos(x, u, p))
        terminal_cost = self.params.weight_p * p[0]

        if self.use_initial_slack:
            running_cost += self.params.lambda_nu * np.linalg.norm(
                x[:, 0] - self.problem_parameters["x_0"].value, ord=1
            )

        if self.debug_nonlin_cost:
            print("defefcts", np.linalg.norm(self._defects(x, u, p), axis=0, ord=1).sum())
            print("nonlinear_pos", np.linalg.norm(self._nonlinear_pos(x, u, p), axis=0, ord=1).sum())
            print("non linear constrain costs", nonlin_constraint_costs)
            print(
                "sum defects and nonlinear_pos",
                np.linalg.norm(self._defects(x, u, p), axis=0, ord=1).sum()
                + np.linalg.norm(self._nonlinear_pos(x, u, p), axis=0, ord=1).sum(),
            )

        return running_cost + nonlin_constraint_costs + terminal_cost

    def _P_k_vec(self, a: np.ndarray, b: np.ndarray):
        """
        Calculate the sum for each discritization step of the L-1 norms of a and b (|a| + |b|).
        """
        K = self.params.K
        norm_a = self.params.lambda_nu * sum(np.linalg.norm(a[:, k], ord=1) for k in range(K))
        if self.is_docking:
            norm_b = self.params.lambda_nu * sum(np.linalg.norm(b[0:-1, k], ord=1) for k in range(K))
            norm_b += self.params.lambda_nu_docking * sum(np.abs(b[-1, k]) for k in range(self.K_docking_start))
        else:
            norm_b = self.params.lambda_nu * sum(np.linalg.norm(b[:, k], ord=1) for k in range(K))
        return norm_a + norm_b

    def _defects(self, x: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Calculate the defects ( x_bar(k+1) - flow map applied from t(k) to t(k+1) to x(k) ) for each discritization step k
        """
        # raw differences
        unshifted = x - self.integrator.integrate_nonlinear_piecewise(x, u, p)

        # shift differences so defect(0) corrisponds to the difference between x(1) and x(0) propogated through the flow map for one time step
        shifted = np.column_stack([unshifted[:, 1:], np.zeros((6, 1))])

        # pull out the raw rotational differences and handle 0-2pi boundry
        raw_dir_diff = np.abs(shifted[2, :]) % (2 * np.pi)
        dir_diff = np.minimum(raw_dir_diff, 2 * np.pi - raw_dir_diff)
        shifted[2, :] = dir_diff

        return shifted

    def _nonlinear_pos(self, x: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Calculate the positive-part function of the value of the LHS of the nonlinear constraints: s(tk, xk, uk, p)
        """
        # For each planet and astroid, calculate -(xk - xp)^2 - (yk - yp) + r ^2 (with xp/yp as function of center for asteroids)
        num_planets = len(self.planets)
        num_astroids = len(self.asteroids)
        if self.is_docking:
            s = np.zeros((num_planets + num_astroids + 2, self.params.K))
        else:
            s = np.zeros((num_planets + num_astroids, self.params.K))
        # idx to insert plant/astroid constraint values at
        idx = 0
        ast_centers = self._calculate_asteroid_centers(p)
        for a_idx, astroid in enumerate(self.asteroids.values()):
            center_k = ast_centers[a_idx]
            val = (
                -((x[0, :] - center_k[0, :]) ** 2)
                - ((x[1, :] - center_k[1, :]) ** 2)
                + (self.r_buffer + astroid.radius) ** 2
            )
            s[idx, :] = val
            idx += 1

        if self.is_docking:
            A = np.array(self.dock_A)
            B = np.array(self.dock_B)
            C = np.array(self.dock_C)

            # goal = np.array([self.goal_state.x, self.goal_state.y])
            A_to_B = np.array(B - A)
            A_to_C = np.array(C - A)

            normal_AtoB_plane = np.array([A_to_B[1], -A_to_B[0]])

            normal_AtoC_plane = np.array([-A_to_C[1], A_to_C[0]])

            for i in range(2):
                for k in range(self.params.K):
                    # SHIFT PLANE FROM A TO DOCKING LINE CENTER FOR IMPROVED FEASIBILITY
                    if k >= self.K_docking_start:
                        dist_vec = x[0:2, k] - A
                        if i == 0:
                            value = -1 * np.dot(dist_vec, normal_AtoB_plane)
                        else:
                            value = -1 * np.dot(dist_vec, normal_AtoC_plane)

                        s[idx, k] = value
                    else:
                        s[idx, k] = 0
                idx += 1

        for planet in self.planets.values():
            val = (
                -((x[0, :] - planet.center[0]) ** 2)
                - ((x[1, :] - planet.center[1]) ** 2)
                + (self.r_buffer + planet.radius) ** 2
            )
            s[idx, :] = val
            idx += 1

        return np.clip(s, 0, None)

    def _extract_seq_from_array(
        self, X_star, U_star, p_star
    ) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Example of how to create a DgSampledSequence from numpy arrays and timestamps.
        """

        K = self.params.K
        ts = [(float(p_star) / (K - 1)) * k for k in range(K)]

        U_0 = U_star[0, :]
        U_1 = U_star[1, :]
        cmds_list = [SatelliteCommands(u_0, u_1) for u_0, u_1 in zip(U_0, U_1)]
        mycmds = DgSampledSequence[SatelliteCommands](timestamps=ts, values=cmds_list)

        # in case my state trajectory is in a 2d array
        states = [SatelliteState(*X_star[:, k]) for k in range(K)]
        mystates = DgSampledSequence[SatelliteState](timestamps=ts, values=states)

        return mycmds, mystates

    def _calculate_linear_constr_planets(self) -> tuple[NDArray, NDArray]:
        """
        Compute the linearized obstacle-avoidance constraints for static planets.

        The function produces:
        - C_p : linear coefficients of the constraint w.r.t. the state
        - r_p : offset term of the linearized constraint

        These terms are evaluated at the current linearization point (self.X_bar)
        and must be recomputed at every SCvx iteration.

        """

        planets = self.planets
        n_planets = len(planets)

        K = self.params.K
        pos_bar = np.vstack((self.X_bar[0, :], self.X_bar[1, :]))

        C_p = np.zeros((n_planets * 2, K))
        for i in range(2):
            for j, planet in enumerate(planets.values()):
                row = n_planets * i + j
                C_p[row, :] = -2 * (pos_bar[i, :] - planet.center[i])

        r_p = np.zeros((n_planets, K))
        for j, planet in enumerate(planets.values()):
            r_p[j] = (
                -((pos_bar[0, :] - planet.center[0]) ** 2)
                - ((pos_bar[1, :] - planet.center[1]) ** 2)
                + (planet.radius + self.r_buffer) ** 2
                + 2 * (pos_bar[0, :] - planet.center[0]) * pos_bar[0, :]
                + 2 * (pos_bar[1, :] - planet.center[1]) * pos_bar[1, :]
            )

        return C_p, r_p

    def _calculate_asteroid_centers(self, p) -> list[np.ndarray]:

        asteroids = self.asteroids
        asteroid_centers = []
        K = self.params.K

        dt = float(p[0]) / (K - 1)
        t_k = np.arange(K) * dt

        for asteroid in asteroids.values():
            start = np.array(asteroid.start)
            velocity = np.array(asteroid.velocity)
            real_vel = np.array(
                [
                    velocity[0] * np.cos(asteroid.orientation) - velocity[1] * np.sin(asteroid.orientation),
                    velocity[0] * np.sin(asteroid.orientation) + velocity[1] * np.cos(asteroid.orientation),
                ]
            )
            asteroid_centers.append(start[:, None] + real_vel[:, None] * t_k[None, :])

        return asteroid_centers

    def _calculate_linear_constr_asteroids(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Compute the linearized obstacle-avoidance constraints for moving asteroids.

        The function returns:
        - C_a : linear coefficients w.r.t. the state
        - r_a : offset term of the linearized constraint
        - G_a : sensitivity of the constraint w.r.t. the final-time variable p

        Asteroid positions are updated based on the current estimate of p,
        and all terms must be recomputed at every SCvx iteration.
        """
        asteroids = self.asteroids
        n_asteroids = len(asteroids)

        K = self.params.K
        pos_bar = np.vstack((self.X_bar[0, :], self.X_bar[1, :]))
        p_bar = self.p_bar
        tau_k = np.linspace(0, 1, K)
        # Build asteroid trajectory
        asteroid_centers = self._calculate_asteroid_centers(p_bar)
        self.asteroid_centers = asteroid_centers

        C_a = np.zeros((n_asteroids * 2, K))
        for i in range(2):
            for j, asteroid in enumerate(asteroids.values()):
                row = n_asteroids * i + j
                ast_center = asteroid_centers[j]
                C_a[row, :] = -2 * (pos_bar[i, :] - ast_center[i, :])

        G_a = np.zeros((n_asteroids, K))

        for j, asteroid in enumerate(asteroids.values()):
            v_x = asteroid.velocity[0]
            v_y = asteroid.velocity[1]

            world_vx = v_x * np.cos(asteroid.orientation) - v_y * np.sin(asteroid.orientation)

            world_vy = v_x * np.sin(asteroid.orientation) + v_y * np.cos(asteroid.orientation)
            ast_center = asteroid_centers[j]

            G_a[j] = -2 * world_vx * tau_k * (ast_center[0, :] - pos_bar[0, :]) - 2 * world_vy * tau_k * (
                ast_center[1, :] - pos_bar[1, :]
            )

        r_a = np.zeros((n_asteroids, K))
        for j, asteroid in enumerate(asteroids.values()):
            ast_center = asteroid_centers[j]
            v_x = asteroid.velocity[0]
            v_y = asteroid.velocity[1]

            world_vx = v_x * np.cos(asteroid.orientation) - v_y * np.sin(asteroid.orientation)
            world_vy = v_x * np.sin(asteroid.orientation) + v_y * np.cos(asteroid.orientation)

            r_a[j] = (
                -((pos_bar[0, :] - ast_center[0, :]) ** 2)
                - ((pos_bar[1, :] - ast_center[1, :]) ** 2)
                + (asteroid.radius + self.r_buffer) ** 2
                + 2 * (pos_bar[0, :] - ast_center[0, :]) * pos_bar[0, :]
                + 2 * (pos_bar[1, :] - ast_center[1, :]) * pos_bar[1, :]
                + 2 * world_vx * tau_k * (ast_center[0, :] - pos_bar[0, :]) * p_bar
                + 2 * world_vy * tau_k * (ast_center[1, :] - pos_bar[1, :]) * p_bar
            )

        return C_a, r_a, G_a
