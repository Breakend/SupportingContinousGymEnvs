"""
Warning:
    Please note that running this test may use up your RAM, so that please carefully deal with this script.
    On my local, it used about 10GB RAM.
"""

# ALL envs included in gym-extensions
ALL_ENVS = [
"PusherMovingGoal-v1",
"PusherLeftSide-v1",
"PusherFullRange-v1",
"StrikerMovingStart-v1",
"AntGravityMars-v1",
"AntGravityHalf-v1",
"AntGravityOneAndHalf-v1",
"HopperGravityHalf-v1",
"HopperGravityThreeQuarters-v1",
"HopperGravityOneAndHalf-v1",
"HopperGravityOneAndQuarter-v1",
"Walker2dGravityHalf-v1",
"Walker2dGravityThreeQuarters-v1",
"Walker2dGravityOneAndHalf-v1",
"Walker2dGravityOneAndQuarter-v1",
"HalfCheetahGravityHalf-v1",
"HalfCheetahGravityThreeQuarters-v1",
"HalfCheetahGravityOneAndHalf-v1",
"HalfCheetahGravityOneAndQuarter-v1",
"HumanoidGravityHalf-v1",
"HumanoidGravityThreeQuarters-v1",
"HumanoidGravityOneAndHalf-v1",
"HumanoidGravityOneAndQuarter-v1",
"AntMaze-v1",
"HopperStairs-v1",
"HopperSimpleWall-v1",
"HopperWithSensor-v1",
"Walker2dWall-v1",
"Walker2dWithSensor-v1",
"HalfCheetahWall-v1",
"HalfCheetahWithSensor-v1",
"HumanoidWall-v1",
"HumanoidWithSensor-v1",
"HumanoidStandupWithSensor-v1",
"HumanoidStandupAndRunWall-v1",
"HumanoidStandupAndRunWithSensor-v1",
"HumanoidStandupAndRun-v1",
"HopperBigTorso-v1",
"HopperBigThigh-v1",
"HopperBigLeg-v1",
"HopperBigFoot-v1",
"HopperSmallTorso-v1",
"HopperSmallThigh-v1",
"HopperSmallLeg-v1",
"HopperSmallFoot-v1",
"Walker2dBigTorso-v1",
"Walker2dBigThigh-v1",
"Walker2dBigLeg-v1",
"Walker2dBigFoot-v1",
"Walker2dSmallTorso-v1",
"Walker2dSmallThigh-v1",
"Walker2dSmallLeg-v1",
"Walker2dSmallFoot-v1",
"HalfCheetahBigTorso-v1",
"HalfCheetahBigThigh-v1",
"HalfCheetahBigLeg-v1",
"HalfCheetahBigFoot-v1",
"HalfCheetahSmallTorso-v1",
"HalfCheetahSmallThigh-v1",
"HalfCheetahSmallLeg-v1",
"HalfCheetahSmallFoot-v1",
"HalfCheetahSmallHead-v1",
"HalfCheetahBigHead-v1",
"HumanoidBigTorso-v1",
"HumanoidBigThigh-v1",
"HumanoidBigLeg-v1",
"HumanoidBigFoot-v1",
"HumanoidSmallTorso-v1",
"HumanoidSmallThigh-v1",
"HumanoidSmallLeg-v1",
"HumanoidSmallFoot-v1",
"HumanoidSmallHead-v1",
"HumanoidBigHead-v1",
"HumanoidSmallArm-v1",
"HumanoidBigArm-v1",
"HumanoidSmallHand-v1",
"HumanoidBigHand-v1"
]

import gym, time
from gym_extensions.continuous import mujoco

for env_name in ALL_ENVS:
    print("Env: {}".format(env_name))
    env = gym.make(env_name)
    env.reset()
    for _ in range(10):
        env.render()
        s, r, d, i = env.step(env.action_space.sample()) # take a random action
        # print(s.shape, r, d, i)
    env.close()
    time.sleep(1) # since opening some env consumes a lot of RAM we should be sleeping a bit.