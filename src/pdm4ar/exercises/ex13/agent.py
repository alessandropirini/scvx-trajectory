from dataclasses import dataclass
from typing import Sequence
import numpy as np

from dg_commons import DgSampledSequence, PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import SatelliteGeometry, SatelliteParameters

from pdm4ar.exercises.ex13.planner import SatellitePlanner
from pdm4ar.exercises_def.ex13.goal import SpaceshipTarget, DockingTarget
from pdm4ar.exercises_def.ex13.utils_params import PlanetParams, AsteroidParams
from pdm4ar.exercises_def.ex13.utils_plot import plot_traj


# HINT: as a good practice we suggest to use the config class to centralise activation of the debugging options
class Config:
    PLOT = True
    VERBOSE = False


@dataclass(frozen=True)
class MyAgentParams:
    """
    You can for example define some agent parameters.
    """

    my_tol: float = 0.1


class SatelliteAgent(Agent):
    # How does it enter in the simulation? The SpaceshipAgent object is created as value
    # corresponding to key "PDM4ARSpaceship" in dict "players", which is an attribute of
    # SimContext returned by "sim_context_from_yaml" in utils_config.py
    """
    This is the PDM4AR agent.
    Do *NOT* modify this class name
    Do *NOT* modify the naming of the existing methods and input/output types.
    """

    init_state: SatelliteState
    planets: dict[PlayerName, PlanetParams]
    asteroids: dict[PlayerName, AsteroidParams]
    goal_state: DynObstacleState

    cmds_plan: DgSampledSequence[SatelliteCommands]
    state_traj: DgSampledSequence[SatelliteState]
    myname: PlayerName
    planner: SatellitePlanner
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: SatelliteGeometry
    sp: SatelliteParameters

    def __init__(
        self,
        init_state: SatelliteState,
        planets: dict[PlayerName, PlanetParams],
        asteroids: dict[PlayerName, AsteroidParams],
    ):
        """
        Initializes the agent.
        This method is called by the simulator only before the beginning of each simulation.
        Provides the SatelliteAgent with information about its environment, i.e. planet and satellite parameters and its initial position.
        """
        self.actual_trajectory = []
        self.init_state = init_state
        self.planets = planets
        self.asteroids = asteroids

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        """
        This method is called by the simulator only at the beginning of each simulation.
        We suggest to compute here an initial trajectory/node graph/path, used by your planner to navigate the environment.

        Do **not** modify the signature of this method.

        the time spent in this method is **not** considered in the score.
        """
        self.myname = init_sim_obs.my_name
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params
        self.scenario = init_sim_obs.dg_scenario

        assert isinstance(init_sim_obs.goal, SpaceshipTarget | DockingTarget)
        self.goal_state = init_sim_obs.goal.target

        self.planner = SatellitePlanner(
            planets=self.planets,
            asteroids=self.asteroids,
            sg=self.sg,
            sp=self.sp,
            scenario=self.scenario,
            goal=init_sim_obs.goal,
            init_state=self.init_state,
        )

        # Plot docking station (this is optional, for better visualization)
        if Config.PLOT and isinstance(init_sim_obs.goal, DockingTarget):
            A, B, C, A1, A2, half_p_angle = init_sim_obs.goal.get_landing_constraint_points()
            init_sim_obs.goal.plot_landing_points(A, B, C, A1, A2)
            init_sim_obs.goal.get_landing_constraint_points_offset()

        # NOTE: The initial trajectory and optional docking constraints are handled within self.planner.compute_trajectory

        self.cmds_plan, self.state_traj = self.planner.compute_trajectory(self.init_state, self.goal_state)

    def get_commands(self, sim_obs: SimObservations) -> SatelliteCommands:
        """
        This method is called by the simulator at every simulation time step. (0.1 sec)
        We suggest to perform two tasks here:
         - Track the computed trajectory (open or closed loop)
         - Plan a new trajectory if necessary
         (e.g., our tracking is deviating from the desired trajectory, the obstacles are moving, etc.)

        NOTE: this function is not run in real time meaning that the simulation is stopped when the function is called.
        Thus the time efficiency of the replanning is not critical for the simulation.
        However the average time spent in get_commands is still considered in the score.

        Do **not** modify the signature of this method.
        """
        current_state = sim_obs.players[self.myname].state
        self.actual_trajectory.append(current_state)
        expected_state = self.state_traj.at_interp(sim_obs.time)

        # plotting the trajectory every 2.5 sec (this is optional, for better visualization)
        if Config.PLOT and int(10 * sim_obs.time) % 25 == 0:
            plot_traj(self.state_traj, self.actual_trajectory)

        # Done: Check if replan is needed (define tolerance between current and expected); if it is:
        #   - recompute optimal trajectory
        #   - shift timestamps to account for current time being the start time

        # # Use tolerances from goal for position, rotation, and velocity
        # pos_tol = self.planner.goal.pos_tol
        # dir_tol = self.planner.goal.dir_tol
        # vel_tol = self.planner.goal.vel_tol

        # # Use ratio of position to velocity to estimate reasonable rotational velocity tolerance
        # rot_vel_tol = vel_tol / pos_tol * dir_tol

        # # Check for deviation (of any state variable) from desired trajectory
        # raw_dir_diff = np.abs(expected_state.psi - current_state.psi) % (2 * np.pi)
        # dir_diff = np.minimum(raw_dir_diff, 2 * np.pi - raw_dir_diff)
        # if (
        #     (abs(expected_state.x - current_state.x) > pos_tol)
        #     or ((abs(expected_state.y - current_state.y) > pos_tol))
        #     or (dir_diff > dir_tol)
        #     or (abs(expected_state.vx - current_state.vx) > vel_tol)
        #     or (abs(expected_state.vy - current_state.vy) > vel_tol)
        #     or (abs(expected_state.dpsi - current_state.dpsi) > rot_vel_tol)
        # ):

        #     # Recompute trajectory (returned sequences start at time 0)
        #     cmds_plan, state_traj = self.planner.compute_trajectory(current_state, self.goal_state)

        #     # Shift command and state sequences to begin from current time
        #     self.cmds_plan = cmds_plan.shift_timestamps(float(sim_obs.time))
        #     self.state_traj = state_traj.shift_timestamps(float(sim_obs.time))

        # ZeroOrderHold
        # cmds = self.cmds_plan.at_or_previous(sim_obs.time)

        # FirstOrderHold
        cmds = self.cmds_plan.at_interp(sim_obs.time)

        # NOTE: could implement closed loop control (open loop right now)
        #   - open loop: apply cmds directly according to optimal trajectory
        #   - closed loop: adjust commands based on position/velocity errors with kp/kd

        return cmds
