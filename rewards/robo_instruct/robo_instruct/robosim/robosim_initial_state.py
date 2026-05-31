from rewards.robo_instruct.roboeval.benchmark.simulator import State

# arbitrarily initalize with 4 locations
initial_state = State().addLocation("loc_a").addLocation("loc_b").addLocation("loc_c").addLocation("start_loc").addRobotLocation("start_loc")
