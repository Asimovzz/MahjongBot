import argparse
import json
from pathlib import Path

import numpy as np

from .paths import DATA_DIR


FEATURE_NUM = 70
OBS_AUGMENT_INDEX = [
    [1, 2, 3],
    [1, 3, 2],
    [2, 1, 3],
    [2, 3, 1],
    [3, 1, 2],
    [3, 2, 1],
]

OFFSET_ACT = {
    "Pass": 0,
    "Hu": 1,
    "Play": 2,
    "Chi": 36,
    "Peng": 99,
    "Gang": 133,
    "AnGang": 167,
    "BuGang": 201,
}

OFFSET_ACT_BY_SUIT = {
    "Pass": 0,
    "Hu": 1,
    "Play_W": 2,
    "Play_T": 11,
    "Play_B": 20,
    "Chi_W": 36,
    "Chi_T": 57,
    "Chi_B": 78,
    "Peng": 99,
    "Gang": 133,
    "AnGang": 167,
    "BuGang": 201,
}

TILE_LIST = [
    *("W%d" % (i + 1) for i in range(9)),
    *("T%d" % (i + 1) for i in range(9)),
    *("B%d" % (i + 1) for i in range(9)),
    *("F%d" % (i + 1) for i in range(4)),
    *("J%d" % (i + 1) for i in range(3)),
]
OFFSET_TILE = {c: i for i, c in enumerate(TILE_LIST)}


def action2response(action):
    if action < OFFSET_ACT["Hu"]:
        return "Pass"
    if action < OFFSET_ACT["Play"]:
        return "Hu"
    if action < OFFSET_ACT["Chi"]:
        return "Play " + TILE_LIST[action - OFFSET_ACT["Play"]]
    if action < OFFSET_ACT["Peng"]:
        t = (action - OFFSET_ACT["Chi"]) // 3
        return "Chi " + "WTB"[t // 7] + str(t % 7 + 2)
    if action < OFFSET_ACT["Gang"]:
        return "Peng"
    if action < OFFSET_ACT["AnGang"]:
        return "Gang"
    if action < OFFSET_ACT["BuGang"]:
        return "Gang " + TILE_LIST[action - OFFSET_ACT["AnGang"]]
    return "BuGang " + TILE_LIST[action - OFFSET_ACT["BuGang"]]


def response2action(response):
    t = response.split()
    if t[0] == "Pass":
        return OFFSET_ACT["Pass"]
    if t[0] == "Hu":
        return OFFSET_ACT["Hu"]
    if t[0] == "Play":
        return OFFSET_ACT["Play"] + OFFSET_TILE[t[1]]
    if t[0] == "Chi":
        return OFFSET_ACT["Chi"] + "WTB".index(t[1][0]) * 7 * 3 + (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1
    if t[0] == "Peng":
        return OFFSET_ACT["Peng"] + OFFSET_TILE[t[1]]
    if t[0] == "Gang":
        return OFFSET_ACT["Gang"] + OFFSET_TILE[t[1]]
    if t[0] == "AnGang":
        return OFFSET_ACT["AnGang"] + OFFSET_TILE[t[1]]
    if t[0] == "BuGang":
        return OFFSET_ACT["BuGang"] + OFFSET_TILE[t[1]]
    return OFFSET_ACT["Pass"]


def augment_match(index, input_dir, output_dir):
    d = np.load(Path(input_dir) / f"{index}.npz")
    cache = {k: d[k] for k in d}

    obs_augment = []
    mask_augment = []
    act_augment = []

    act_0 = np.zeros([cache["act"].size, 235])
    for row in range(cache["act"].size):
        act_0[row][cache["act"][row]] = 1

    for perm in OBS_AUGMENT_INDEX:
        obs_1 = cache["obs"].copy()
        for j in range(3):
            obs_1[:, 2:FEATURE_NUM, j, :] = cache["obs"][:, 2:FEATURE_NUM, perm[j] - 1, :]
        obs_augment.append(obs_1)

        mask_1 = cache["mask"].copy()
        act_1 = act_0.copy()
        for j in range(3):
            src = perm[j] - 1
            mask_1[:, OFFSET_ACT_BY_SUIT["Chi_W"] + 21 * j : OFFSET_ACT_BY_SUIT["Chi_T"] + 21 * j] = cache["mask"][:, OFFSET_ACT_BY_SUIT["Chi_W"] + 21 * src : OFFSET_ACT_BY_SUIT["Chi_T"] + 21 * src]
            mask_1[:, OFFSET_ACT_BY_SUIT["Play_W"] + 9 * j : OFFSET_ACT_BY_SUIT["Play_T"] + 9 * j] = cache["mask"][:, OFFSET_ACT_BY_SUIT["Play_W"] + 9 * src : OFFSET_ACT_BY_SUIT["Play_T"] + 9 * src]
            mask_1[:, OFFSET_ACT_BY_SUIT["Peng"] + 9 * j : OFFSET_ACT_BY_SUIT["Peng"] + 9 * (j + 1)] = cache["mask"][:, OFFSET_ACT_BY_SUIT["Peng"] + 9 * src : OFFSET_ACT_BY_SUIT["Peng"] + 9 * (src + 1)]
            mask_1[:, OFFSET_ACT_BY_SUIT["Gang"] + 9 * j : OFFSET_ACT_BY_SUIT["Gang"] + 9 * (j + 1)] = cache["mask"][:, OFFSET_ACT_BY_SUIT["Gang"] + 9 * src : OFFSET_ACT_BY_SUIT["Gang"] + 9 * (src + 1)]
            mask_1[:, OFFSET_ACT_BY_SUIT["AnGang"] + 9 * j : OFFSET_ACT_BY_SUIT["AnGang"] + 9 * (j + 1)] = cache["mask"][:, OFFSET_ACT_BY_SUIT["AnGang"] + 9 * src : OFFSET_ACT_BY_SUIT["AnGang"] + 9 * (src + 1)]
            mask_1[:, OFFSET_ACT_BY_SUIT["BuGang"] + 9 * j : OFFSET_ACT_BY_SUIT["BuGang"] + 9 * (j + 1)] = cache["mask"][:, OFFSET_ACT_BY_SUIT["BuGang"] + 9 * src : OFFSET_ACT_BY_SUIT["BuGang"] + 9 * (src + 1)]

            act_1[:, OFFSET_ACT_BY_SUIT["Chi_W"] + 21 * j : OFFSET_ACT_BY_SUIT["Chi_T"] + 21 * j] = act_0[:, OFFSET_ACT_BY_SUIT["Chi_W"] + 21 * src : OFFSET_ACT_BY_SUIT["Chi_T"] + 21 * src]
            act_1[:, OFFSET_ACT_BY_SUIT["Play_W"] + 9 * j : OFFSET_ACT_BY_SUIT["Play_T"] + 9 * j] = act_0[:, OFFSET_ACT_BY_SUIT["Play_W"] + 9 * src : OFFSET_ACT_BY_SUIT["Play_T"] + 9 * src]
            act_1[:, OFFSET_ACT_BY_SUIT["Peng"] + 9 * j : OFFSET_ACT_BY_SUIT["Peng"] + 9 * (j + 1)] = act_0[:, OFFSET_ACT_BY_SUIT["Peng"] + 9 * src : OFFSET_ACT_BY_SUIT["Peng"] + 9 * (src + 1)]
            act_1[:, OFFSET_ACT_BY_SUIT["Gang"] + 9 * j : OFFSET_ACT_BY_SUIT["Gang"] + 9 * (j + 1)] = act_0[:, OFFSET_ACT_BY_SUIT["Gang"] + 9 * src : OFFSET_ACT_BY_SUIT["Gang"] + 9 * (src + 1)]
            act_1[:, OFFSET_ACT_BY_SUIT["AnGang"] + 9 * j : OFFSET_ACT_BY_SUIT["AnGang"] + 9 * (j + 1)] = act_0[:, OFFSET_ACT_BY_SUIT["AnGang"] + 9 * src : OFFSET_ACT_BY_SUIT["AnGang"] + 9 * (src + 1)]
            act_1[:, OFFSET_ACT_BY_SUIT["BuGang"] + 9 * j : OFFSET_ACT_BY_SUIT["BuGang"] + 9 * (j + 1)] = act_0[:, OFFSET_ACT_BY_SUIT["BuGang"] + 9 * src : OFFSET_ACT_BY_SUIT["BuGang"] + 9 * (src + 1)]

        mask_augment.append(mask_1)
        act_augment.append(act_1)

    obs = np.concatenate(tuple(obs_augment), axis=0)
    mask = np.concatenate(tuple(mask_augment), axis=0)
    act_one_hot = np.concatenate(tuple(act_augment), axis=0)
    act = np.argmax(act_one_hot, axis=1).astype(np.int32)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / f"{index}_augmented_{FEATURE_NUM}.npz", obs=obs, mask=mask, act=act)
    if index % 1024 == 0:
        print("data %d augmented and saved" % index)


def augment_all(count_path, input_dir, output_dir):
    with Path(count_path).open(encoding="utf-8") as f:
        match_samples = json.load(f)
    for i in range(len(match_samples)):
        augment_match(i, input_dir, output_dir)


def build_parser():
    parser = argparse.ArgumentParser(description="Augment preprocessed Mahjong samples by permuting suits.")
    parser.add_argument("--count-path", default=str(DATA_DIR / "count.json"))
    parser.add_argument("--input-dir", default=str(DATA_DIR / "preprocessed"))
    parser.add_argument("--output-dir", default=str(DATA_DIR / "augmented"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    augment_all(args.count_path, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
