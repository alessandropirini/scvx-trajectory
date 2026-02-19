# Sequential Convex Optimization for Trajectory Generation and Control
Handed the controls for a small satellite, what's the *best* trajectory from A to B? One strategy for trajectory generation and control allocation is sequential convex optimization. This allows both convex and non-convex constraints to be accomodated by iteratively linearizing the constraints and handling potentially infeasable constraints with slack variables that incur a large penalty. The problem then boils down to discritizing the dynamics, linearizing the non-convex constraints, solving the optimization problem, and repeating, using the previous solution as the trajectory to linearize about. More details (a lot) can be found [here](https://arxiv.org/abs/2106.09125), specifically in the section on SCvx (pg. 31).

<table>
  <tr>
    <td style="vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/557a9159-3703-47d1-8733-2774c19d191a" width="100%">
    </td>
    <td style="vertical-align: top;">
      <img width="600" height="450" src="https://github.com/user-attachments/assets/420abdb0-8202-46bf-8a6d-e953ee1f1adf" width="100%">
    </td>
  </tr>
</table>
