from roboeval.benchmark.rtl import *
from roboeval.benchmark.simulator import State
from roboeval.misc.benchmark_utils import *
from typing import List

name = "LongHorizon1"

prompts = [
    "Let's play a game: Double and give it to the next person. Start with 1 dollar. Go to rooms A, B, C, D, E, F, and G. If you see someone, tell them how much money you have. Then ask if they would like to take the money now or double the amount and give it to the next person. If they choose to take it, the game is over, and you should come back to me. Otherwise, double your money and continue. If, in the end, no one takes the money, tell me how much you still have."
]

tests = []

