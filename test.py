import argparse
import numpy as np

from utils.config import *
from agents import *

def main():
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument('config',
                            metavar='config_json_file',
                            default='None',
                            help='The Configuration file in json format')

    args = arg_parser.parse_args()

    config = process_config(args.config)

    # 에이전트 생성
    agent_class = globals()[config.agent]
    agent = agent_class(config)
    agent.load_checkpoint(config.ckpt_file)

    # 테스트 플레이
    score_list = []
    for play_idx in range(10):
        ep_score = agent.play()
        score_list.append(ep_score)
        print("play_idx:{:3d} \t score: {:.3f}".format(play_idx, ep_score))

    print("\n")
    print("Average score: {:.3f}".format(np.mean(score_list)))



if __name__ == "__main__":
    main()