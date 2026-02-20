# Sequential Convex Optimization for Trajectory Optimization
Difficult maneuvers in space demand more than intuition. This repository implements a Sequential Convexification (SCvx) framework to compute safe feasible satellite trajectories for autonomous docking with dynamically changing environments. Obstacle avoidance, control limits, and nonlinear orbital dynamics are handled through an iterative scheme that linearizes the problem around the current trajectory and solves a sequence of convex programs until convergence. The result is a trajectory that respects physical constraints while guiding the spacecraft smoothly and safely to its docking target.

For a detailed explanation on Sequential Convexification, see the SCvx section (p. 31) of the following reference, on which the implementation is based:[here](https://arxiv.org/abs/2106.09125)

<table>
  <tr>
    <td style="vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/6bc05a31-a0fe-432f-9ca6-865c10171915" width="100%">
    </td>
    <td style="vertical-align: top;">
      <img width="600" height="450" src="https://github.com/user-attachments/assets/e8109989-450c-4b25-91bd-f4cab7824832" width="100%">
    </td>
  </tr>
</table>
