import timeit
from tqdm import tqdm
import shutil
import numpy as np
import torch
from mlagents.envs import UnityEnvironment

from graph.models.dqn import DQN
from graph.losses.huber_loss import HuberLoss
from agents.base import BaseAgent
from utils.replay_buffer import ReplayBuffer

from tensorboardX import SummaryWriter
from collections import deque


class DQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        # env
        self.env = UnityEnvironment(file_name=self.config.env_location + self.config.env_name)
        self.default_brain = self.env.brain_names[0]
        self.brain = self.env.brains[self.default_brain]
        self.env_info = self.env.reset(train_mode=self.config.train_mode)[self.default_brain]

        # model
        self.policy_model = DQN(self.config)
        self.target_model = DQN(self.config)
        self.replayBuffer = ReplayBuffer(self.config)
        self.loss = HuberLoss()
        self.optim = torch.optim.Adam(self.policy_model.parameters(), lr=self.config.learning_rate)

        # init counter
        self.current_episode = self.config.start_episode
        self.current_iteration = 0
        self.episode_duration = []
        self.episode_rewards_deque = deque(maxlen=100)

        # cuda
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("Warning: Cuda is not being used.")
        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.logger.info("Running on GPU..")
        else:
            self.device = torch.device("cpu")
            self.logger.info("Running on CPU..")

        # epsilon
        self.epsilon = self.config.epsilon_init

        # model to cuda device
        self.policy_model = self.policy_model.to(self.device)
        self.target_model = self.target_model.to(self.device)
        self.loss = self.loss.to(self.device)

        # Target model의 파라미터를 policy_model의 파라미터로 덮어씌우기
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()

        # 텐서보드
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='DQN')
        self.best_score = -np.inf


    def load_checkpoint(self, file_name):
        filename = self.config.checkpoint_dir / file_name
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_episode = checkpoint['episode']
            self.current_iteration = checkpoint['iteration']
            self.policy_model.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir,
                                     checkpoint['episode'],
                                     checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from {}".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        state = {
            'episode': self.current_episode,
            'iteration': self.current_iteration,
            'state_dict': self.policy_model.state_dict(),
            'optimizer': self.optim.state_dict(),
        }

        torch.save(state, self.config.checkpoint_dir / file_name)

        if is_best:
            shutil.copyfile(self.config.checkpoint_dir / file_name,
                            self.config.checkpoint_dir / 'model_best.pth.tar')

    def get_action(self, state_t):
        if self.epsilon > np.random.rand():
            action = np.random.randint(0, self.config.action_dim)
        else:
            self.policy_model.eval()
            with torch.no_grad():
                predicted_value = self.policy_model(state_t)
            self.policy_model.train()
            action = predicted_value.detach().max(1)[1].item()
        return action

    def run(self):
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("Keyboard Interrupted. Wait to finalize..")

    def optimize_policy_model(self, done):

        # epsilon 값 조정
        if done:
            if self.epsilon > self.config.epsilon_min:
                self.epsilon -= 1 / (self.config.num_episodes - self.config.start_episode)

        # 메모리에서 배치를 꺼내온다.
        batch = self.replayBuffer.sample_batch(self.config.batch_size)
        state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor = batch

        state_tensor = state_tensor.to(self.device)
        action_tensor = action_tensor.to(self.device)
        reward_tensor = reward_tensor.to(self.device)
        next_state_tensor = next_state_tensor.to(self.device)
        done_tensor = done_tensor.to(self.device)

        # 현재 state의 예측과, 타깃(r + discounted 다음 state 타겟 예측)간의 차이를 줄이도록 학습한다.
        # state(t)의 예측 가치 (bsz x action_dim)
        current_state_values = self.policy_model(state_tensor)
        # state-action(t)의 예측 가치 (bsz x 1) - 선택한 action을 이용
        current_state_action_prediction = current_state_values.gather(1, action_tensor)

        # state(t+1)의 타겟 가치 (bsz x action_dim)
        with torch.no_grad():
            next_state_values = self.target_model(next_state_tensor).detach()
        # 여러 action-value 중 최대값
        max_next_state_values = next_state_values.max(1)[0].unsqueeze(1)
        # 게임이 Done이면 이건 의미없으므로 0으로 바꾼다. 1-True를 쓰자.
        next_state_prediction = (max_next_state_values * (1-done_tensor))
        # expected Q values
        expected_state_action_values = (reward_tensor + (next_state_prediction * self.config.gamma))

        # loss: HuberLoss(current prediction, estimated target)
        loss_batch = self.loss(current_state_action_prediction, expected_state_action_values)

        self.optim.zero_grad()
        loss_batch.backward()
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()
        self.summary_writer.add_scalar("TD_Loss", loss_batch.detach().item(), self.current_iteration)


    def train(self):

        for episode in tqdm(range(self.current_episode, self.config.num_episodes + 1)):
            self.current_episode = episode

            self.train_one_epoch(episode)

        self.env.close()

    def play(self):

        self.load_checkpoint(self.config.ckpt_file)
        self.epsilon = 0.05

        state = np.array(self.env_info.visual_observations[0])
        episode_rewards = 0
        done = False

        # 에피소드 진행
        while not done:
            state_t = torch.from_numpy(state).permute(0, 3, 1, 2).float()
            action = self.get_action(state_t)

            env_info = self.env.step(action)[self.default_brain]
            reward = env_info.rewards[0]
            next_state = np.array(env_info.visual_observations[0])
            done = env_info.local_done[0]

            state = next_state

            episode_rewards += reward

            if done:
                break

        return episode_rewards



    def train_one_epoch(self, ep_num):

        # 리셋-
        state = np.array(self.env_info.visual_observations[0])
        episode_rewards = 0
        episode_duration = 0

        done = False

        # 에피소드 진행
        while not done:

            episode_duration += 1
            self.current_iteration += 1

            # 행동 결정
            state_t = torch.from_numpy(state).permute(0, 3, 1, 2).float()
            action = self.get_action(state_t)
            env_info = self.env.step(action)[self.default_brain]
            next_state = np.array(env_info.visual_observations[0])
            next_state_t = torch.from_numpy(next_state).permute(0, 3, 1, 2).float()
            reward = env_info.rewards[0]
            episode_rewards += reward
            done = env_info.local_done[0]

            # 버퍼에 기록
            self.replayBuffer.append_sample(state_t, action, reward, next_state_t, done)

            # 상태 정보 업데이트
            state = next_state

            # 학습 수행
            if ep_num > self.config.start_train_episode:
                self.optimize_policy_model(done)

                # 타겟 네트워크 업데이트
                if self.current_iteration % self.config.target_update_iteration == 0:
                    print("target nn update at {}".format(self.current_iteration))
                    self.target_model.load_state_dict(self.policy_model.state_dict())

            # 게임 오버되면 루프 종료
            if done:
                break

        self.episode_rewards_deque.append(episode_rewards)
        episode_rewards_running_average = np.mean(self.episode_rewards_deque)
        if episode_rewards_running_average > self.best_score:
            self.save_checkpoint(is_best=True)
            self.best_score = episode_rewards_running_average

        self.summary_writer.add_scalar("Training_epsilon", self.epsilon, self.current_episode)
        self.summary_writer.add_scalar("Training_Ep_duration", episode_duration, self.current_episode)
        self.summary_writer.add_scalar("Training_Ep_rewards", episode_rewards, self.current_episode)
        self.summary_writer.add_scalar("Training_Ep_rewards_running_average", episode_rewards_running_average, self.current_episode)

        if ep_num % (self.config.save_every) == 0:
            self.save_checkpoint(file_name="ckpt_{:06d}.pth.tar".format(ep_num))


    def finalize(self):
        self.logger.info("Please wait while finalizing the operation.")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()




