from typing import NamedTuple
import chex


class State(NamedTuple):
    mu: chex.Array
    Sigma: chex.Array
    alpha: chex.Array
    beta: chex.Array


class Char(NamedTuple):
    """
    difficulty, 1-5 * as per in game
    archetype_vec: zoner, grappler, strikethrow, etc...
    execution barrier: harder to quantify, would an ebedding be better?
    footsies/neutral, how brainded is the char. 2mk -> dr = -points. harder buttons, less pokes = more neutral that needs to be played
    tier: float/int, using like a couple of pros' tier lists maybe

    """
    difficulty: float
    archetype_vec: chex.Array
    execution_level: float
    neutral_required: float
    tier: float


class UserInfo(NamedTuple):
    """
    the one that should be updated over time. 
    """
    skill_level: float
    games_played: int
    chars_attempted_mask: chex.Array
    wr: chex.Array
    playtime: chex.Array
    pref_archetype: chex.Array

