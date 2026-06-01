import argparse
import json
from pathlib import Path

import numpy as np

from .paths import DATA_DIR


def filter_data(obs, actions):
    new_obs = [[] for _ in range(4)]
    new_actions = [[] for _ in range(4)]
    for i in range(4):
        for j, o in enumerate(obs[i]):
            if o["action_mask"].sum() > 1:
                new_obs[i].append(o)
                new_actions[i].append(actions[i][j])
    return new_obs, new_actions


def save_match(output_dir, match_id, obs, actions):
    assert [len(x) for x in obs] == [len(x) for x in actions], "obs actions not matching!"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / f"{match_id}.npz",
        obs=np.stack([x["observation"] for i in range(4) for x in obs[i]]).astype(np.int8),
        mask=np.stack([x["action_mask"] for i in range(4) for x in obs[i]]).astype(np.int8),
        act=np.array([x for i in range(4) for x in actions[i]]),
    )
    return sum(len(x) for x in obs)


def preprocess(input_path, output_dir, count_path):
    from .feature import FeatureAgent

    obs = [[] for _ in range(4)]
    actions = [[] for _ in range(4)]
    counts = []
    match_id = -1
    cur_tile = None
    agents = None

    with Path(input_path).open(encoding="utf-8") as f:
        line = f.readline()
        while line:
            t = line.split()
            if len(t) == 0:
                line = f.readline()
                continue
            if t[0] == "Match":
                agents = [FeatureAgent(i) for i in range(4)]
                match_id += 1
                if match_id % 128 == 0:
                    print("Processing match %d %s..." % (match_id, t[1]))
            elif t[0] == "Wind":
                for agent in agents:
                    agent.request2obs(line)
            elif t[0] == "Player":
                p = int(t[1])
                if t[2] == "Deal":
                    agents[p].request2obs(" ".join(t[2:]))
                elif t[2] == "Draw":
                    for i in range(4):
                        if i == p:
                            obs[p].append(agents[p].request2obs(" ".join(t[2:])))
                            actions[p].append(0)
                        else:
                            agents[i].request2obs(" ".join(t[:3]))
                elif t[2] == "Play":
                    actions[p].pop()
                    actions[p].append(agents[p].response2action(" ".join(t[2:])))
                    for i in range(4):
                        if i == p:
                            agents[p].request2obs(line)
                        else:
                            obs[i].append(agents[i].request2obs(line))
                            actions[i].append(0)
                    cur_tile = t[3]
                elif t[2] == "Chi":
                    actions[p].pop()
                    actions[p].append(agents[p].response2action("Chi %s %s" % (cur_tile, t[3])))
                    for i in range(4):
                        if i == p:
                            obs[p].append(agents[p].request2obs("Player %d Chi %s" % (p, t[3])))
                            actions[p].append(0)
                        else:
                            agents[i].request2obs("Player %d Chi %s" % (p, t[3]))
                elif t[2] == "Peng":
                    actions[p].pop()
                    actions[p].append(agents[p].response2action("Peng %s" % t[3]))
                    for i in range(4):
                        if i == p:
                            obs[p].append(agents[p].request2obs("Player %d Peng %s" % (p, t[3])))
                            actions[p].append(0)
                        else:
                            agents[i].request2obs("Player %d Peng %s" % (p, t[3]))
                elif t[2] == "Gang":
                    actions[p].pop()
                    actions[p].append(agents[p].response2action("Gang %s" % t[3]))
                    for i in range(4):
                        agents[i].request2obs("Player %d Gang %s" % (p, t[3]))
                elif t[2] == "AnGang":
                    actions[p].pop()
                    actions[p].append(agents[p].response2action("AnGang %s" % t[3]))
                    for i in range(4):
                        if i == p:
                            agents[p].request2obs("Player %d AnGang %s" % (p, t[3]))
                        else:
                            agents[i].request2obs("Player %d AnGang" % p)
                elif t[2] == "BuGang":
                    actions[p].pop()
                    actions[p].append(agents[p].response2action("BuGang %s" % t[3]))
                    for i in range(4):
                        if i == p:
                            agents[p].request2obs("Player %d BuGang %s" % (p, t[3]))
                        else:
                            obs[i].append(agents[i].request2obs("Player %d BuGang %s" % (p, t[3])))
                            actions[i].append(0)
                elif t[2] == "Hu":
                    actions[p].pop()
                    actions[p].append(agents[p].response2action("Hu"))

                if t[2] in ["Peng", "Gang", "Hu"]:
                    for k in range(5, 15, 5):
                        if len(t) <= k:
                            break
                        p = int(t[k + 1])
                        if t[k + 2] == "Chi":
                            actions[p].pop()
                            actions[p].append(agents[p].response2action("Chi %s %s" % (cur_tile, t[k + 3])))
                        elif t[k + 2] == "Peng":
                            actions[p].pop()
                            actions[p].append(agents[p].response2action("Peng %s" % t[k + 3]))
                        elif t[k + 2] == "Gang":
                            actions[p].pop()
                            actions[p].append(agents[p].response2action("Gang %s" % t[k + 3]))
                        elif t[k + 2] == "Hu":
                            actions[p].pop()
                            actions[p].append(agents[p].response2action("Hu"))
            elif t[0] == "Score":
                obs, actions = filter_data(obs, actions)
                counts.append(save_match(Path(output_dir), match_id, obs, actions))
                obs = [[] for _ in range(4)]
                actions = [[] for _ in range(4)]
            line = f.readline()

    with Path(count_path).open("w", encoding="utf-8") as f:
        json.dump(counts, f)


def build_parser():
    parser = argparse.ArgumentParser(description="Convert Botzone match logs into supervised-learning samples.")
    parser.add_argument("--input", default=str(DATA_DIR / "data.txt"), help="Raw match log file.")
    parser.add_argument("--output-dir", default=str(DATA_DIR / "preprocessed"), help="Directory for .npz samples.")
    parser.add_argument("--count-path", default=str(DATA_DIR / "count.json"), help="Output path for per-match sample counts.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    preprocess(args.input, args.output_dir, args.count_path)


if __name__ == "__main__":
    main()
