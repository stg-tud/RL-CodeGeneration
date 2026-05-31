from roboeval.benchmark.rtl import *
from roboeval.benchmark.simulator import State
from roboeval.misc.benchmark_utils import *
from typing import List

name = "LongHorizon2"

prompts = [
    "Go to my office. See if I have a table, a chair, and a monitor there. If there is not, go to Jason's office and checks if he is in his office. If he is, ask him if I can borrow any missing ones. If he says yes, for each missing item, pick it up and bring it to my office. If Jason is not in his office or he says no, come back and tell me the reason",
]

tests = []

