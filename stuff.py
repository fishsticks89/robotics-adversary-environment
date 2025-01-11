import pybullet as p
import pybullet_data
import time

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set up the simulation environment
p.resetSimulation()
p.setGravity(0, 0, -9.8)

# Load a plane
plane_id = p.loadURDF("plane.urdf")

# Create a simple box (brick)
start_pos = [0, 0, 0.5]
start_orientation = [0, 0, 0, 1]  # No rotation (quaternion)
box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
box_body = p.createMultiBody(baseMass=1,
                             baseCollisionShapeIndex=box_id,
                             basePosition=start_pos,
                             baseOrientation=start_orientation)

# Set friction for the box
p.changeDynamics(box_body, -1, lateralFriction=0.5, spinningFriction=0.5, rollingFriction=0.5)

# Physics engine parameters
p.setPhysicsEngineParameter(enableConeFriction=1)

# Simulation parameters
force_magnitude = 50  # Magnitude of the applied force
time_step = 1 / 240  # Simulation time step

# Main simulation loop
while True:
    keys = p.getKeyboardEvents()
    force = [0, 0, 0]

    # WASD/Arrow key controls
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        force[1] += force_magnitude  # Forward (W)
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        force[1] -= force_magnitude  # Backward (S)
    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        force[0] -= force_magnitude  # Left (A)
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        force[0] += force_magnitude  # Right (D)

    # Apply force to the box
    p.applyExternalForce(
        objectUniqueId=box_body,
        linkIndex=-1,
        forceObj=force,
        posObj=p.getBasePositionAndOrientation(box_body)[0],
        flags=p.WORLD_FRAME,
    )

    # Step the simulation
    p.stepSimulation()
    time.sleep(time_step)
