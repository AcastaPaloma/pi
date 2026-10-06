"""Working observation loop in MuJoCo. See docs/OBSERVATIONS.md for SO-101."""
from pi_sim import RobotEnv
from pi_observe import Observer, SimSource

with RobotEnv(control_mode="joint") as env:
    _, info = env.reset(seed=0)
    with Observer(SimSource(env), task=info["instruction"], history=1) as observer:
        observation = observer.read()
        x = observation.inputs
        print("Cameras:", x["camera_names"])
        print("Encoder images:", x["images"].shape)
        print("Proprioception:", x["proprioception"])
        print("Task:", str(x["task"]))
        print("Goal image mask:", x["subgoal_valid"])
        # Your encoder consumes x. Native-resolution RGB is also available:
        front_rgb = observation.images["front"]
        print("Front RGB:", front_rgb.shape)
        # observation.save("outputs/my_capture")  # New directory required.
        # After env.step(action), call observer.read() again.
