from itertools import chain, combinations


def _all_permutations(iterable):
    "all_permutation([0,1,2]) --> () (0,) (1,) (2,) (0,1) (0,2) (1,2) (0,1,2)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


def feasible_permutations(players_number, number_of_rt_players):
    set_players = []

    if number_of_rt_players is None:
        # one half rt and the other half nrt
        number_of_rt_players = round((players_number - 1) / 2)
    # assign type to players
    for i in range(players_number):
        # valid only for SPs
        # SPs NRT
        if i != 0 and number_of_rt_players > 0:
            set_players.append((i, "rt"))
            number_of_rt_players -= 1
        # SPs NRT
        elif i != 0:
            set_players.append((i, "nrt"))
        # NO
        else:
            set_players.append((i, "NO"))

    perms = _all_permutations(set_players)

    return perms
