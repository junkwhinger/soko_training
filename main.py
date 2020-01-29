
import argparse

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

    # 학습
    agent.run()

    # 학습 종료
    agent.finalize()


if __name__ == "__main__":
    main()