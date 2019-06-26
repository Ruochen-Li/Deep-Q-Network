
import copy, os
from typing import List

import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter

from modules.policy.policy_rl import RLPolicy
from modules.policy.rl.common import DEVICE
from modules.policy.rl.experience_buffer import UniformBuffer, NaivePrioritizedBuffer
from modules.policy.rl.dqn import DQN, DuelingDQN, NetArchitecture
from utils.useract import UserActionType, UserAct
from utils.sysact import SysAct, SysActionType
from utils.beliefstate import BeliefState
from utils.logger import DiasysLogger
from utils import Goal, common
logger = DiasysLogger()

MAX_TURNS = 25



class DQNPolicy(RLPolicy):

    def __init__(self, domain,
                 architecture:NetArchitecture=NetArchitecture.DUELING,
                 hidden_layer_sizes=[300, 700, 700],  # vanilla architecture
                 shared_layer_sizes=[300], value_layer_sizes=[300,300],
                 advantage_layer_sizes = [400, 400],  # dueling architecture
                 lr=0.0001, discount_gamma=0.99,
                 target_update_rate = 5,
                 replay_buffer_size=16384, batch_size=64,
                 buffer_cls=NaivePrioritizedBuffer,
                 eps_start=0.3, eps_end=0.0,
                 l2_regularisation=0.0, gradient_clipping=0.0,
                 p_dropout=0.0, training_frequency=1, train_dialogs=1000,
                 include_confreq=False):
        """
        Args:
            target_update_rate: if 1, vanilla dqn update
                                if > 1, double dqn with specified target update
                                rate
        """
        super(DQNPolicy, self).__init__(domain, buffer_cls=buffer_cls,
                 buffer_size=replay_buffer_size, batch_size=batch_size,
                 discount_gamma=discount_gamma, include_confreq=include_confreq)

        self.training_frequency = training_frequency
        self.train_dialogs = train_dialogs
        self.lr = lr
        self.gradient_clipping = gradient_clipping
        if gradient_clipping > 0.0:
            logger.info("Gradient Clipping: " + str(gradient_clipping))
        self.target_update_rate = target_update_rate

        self.epsilon_start = eps_start
        self.epsilon_end = eps_end

        # Select network architecture
        if architecture == NetArchitecture.VANILLA:
            logger.info("Architecture: Vanilla")
            self.model = DQN(self.state_dim, self.action_dim,
                            hidden_layer_sizes=hidden_layer_sizes,
                            dropout_rate=p_dropout)
        else:
            logger.info("Architecture: Dueling")
            self.model = DuelingDQN(self.state_dim, self.action_dim,
                                    shared_layer_sizes=shared_layer_sizes,
                                    value_layer_sizes=value_layer_sizes,
                                    advantage_layer_sizes=advantage_layer_sizes,
                                    dropout_rate=p_dropout)
        # Select network update
        self.target_model = None
        if target_update_rate > 1:
            logger.info("Update: Double")
            if architecture == NetArchitecture.VANILLA:
                self.target_model = copy.deepcopy(self.model)
        else:
            logger.info("Update: Vanilla")

        self.optim = optim.Adam(self.model.parameters(), lr=lr,
                                weight_decay=l2_regularisation)
        self.loss_fun = nn.SmoothL1Loss(reduction='none')
        #self.loss_fun = nn.MSELoss(reduction='none')

        self.train_call_count = 0
        self.total_train_dialogs = 0
        self.epsilon = self.epsilon_start


    def select_action_eps_greedy(self, state_vector):
        """ Epsilon-greedy policy """
        self.eps_scheduler()

        # epsilon greedy exploration
        if self.is_training and common.random.random() < self.epsilon:
            next_action_idx = common.random.randint(0, self.action_dim-1)
            #print("rand act", self.action_name(next_action_idx))
        else:
            torch.autograd.set_grad_enabled(False)
            q_values = self.model(state_vector)
            next_action_idx = q_values.squeeze(dim=0).max(dim=0)[1].item()
            torch.autograd.set_grad_enabled(True)
        return next_action_idx


    def end_dialog(self, sim_goal):
        super(DQNPolicy, self).end_dialog(sim_goal)
        if self.is_training:
            self.total_train_dialogs += 1
        self.train_batch()


    def forward(self, dialog_graph, user_acts: List[UserAct] = None,
                beliefstate: BeliefState = None, sim_goal : Goal = None,
                writer : SummaryWriter = None, **kwargs) -> dict(sys_act=SysAct):
        super(DQNPolicy, self).forward(dialog_graph, user_acts=user_acts,
                                       beliefstate=beliefstate, sim_goal=sim_goal,
                                       writer=writer)
        self.num_dialogs = dialog_graph.num_dialogs % self.train_dialogs
        if dialog_graph.num_dialogs == 0 and self.target_model is not None:
            # start with same weights for target and online net when a new epoch begins
            self.target_model.load_state_dict(self.model.state_dict())
        if dialog_graph.num_turns == 0:
            # first turn of dialog: say hello & don't record
            return self._expand_hello()

        if dialog_graph.num_turns > MAX_TURNS:
            # reached turn limit -> terminate dialog
            bye_action = SysAct()
            bye_action.type = SysActionType.Bye
            self.last_sys_act = bye_action
            #self.end_dialog(sim_goal)
            logger.dialog_turn("system action > " + str(bye_action))
            return {'sys_act': bye_action}

        # intermediate or closing turn
        #logger.dialog_turn("   last informed venue" + beliefstate['system']['lastInformedPrimKeyVal'])
        state_vector = self.beliefstate_dict_to_vector(beliefstate)
        next_action_idx = -1

        # check if user ended dialog
        if user_acts is not None:
            for user_act in user_acts:
                if user_act.type == UserActionType.Bye:
                # user terminated current dialog -> say bye
                    next_action_idx = self.action_idx(SysActionType.Bye.value)
        if next_action_idx == -1:
            # dialog continues
            next_action_idx = self.select_action_eps_greedy(state_vector)

        self.turn_end(beliefstate, state_vector, next_action_idx)

        # if next_action_idx == self.action_idx(SysActionType.Bye.value):
        #     # system ended current dialog
        #     self.end_dialog(sim_goal)
        # logger.dialog_turn("Belief State: " + str(list(filter(lambda x: x[0] in ['area', 'pricerange'], beliefstate['beliefs'].items()))))
        return {'sys_act': self.last_sys_act}


    def _forward(self, state, action):
        q_values = self.model(state)
        return q_values.gather(1, action)


    def _forward_target(self, state, reward, terminal, gamma):
        target_q_values = self.model(state)
        greedy_actions = target_q_values.max(1)[1].unsqueeze(1)
        target_q_values = reward + (1.0 - terminal) * gamma * target_q_values.gather(1, greedy_actions)
        return target_q_values


    def _forward_target_ddqn(self, state, reward, terminal, gamma):
        greedy_actions = self.model(state).max(1)[1].unsqueeze(1)
        target_q_values = self.target_model(state).gather(1, greedy_actions)
        target_q_values = reward + (1.0 - terminal) * gamma * target_q_values
        return target_q_values


    def loss(self, s_batch, a_batch, s2_batch, r_batch, t_batch, gamma):
        # forward value
        torch.autograd.set_grad_enabled(True)
        q_val = self._forward(s_batch, a_batch)

        # forward target
        torch.autograd.set_grad_enabled(False)
        if self.target_model is None:
            q_target = self._forward_target(s2_batch, r_batch, t_batch, gamma)
        else:
            q_target = self._forward_target_ddqn(s2_batch, r_batch, t_batch,
                                                 gamma)
        torch.autograd.set_grad_enabled(True)

        # loss
        loss = self.loss_fun(q_val, q_target)
        return loss


    def train_batch(self):
        """ Train on a minibatch drawn from the experience buffer. """
        if not self.is_training:
            return

        if len(self.buffer) >= self.batch_size * 10 and \
                self.total_train_dialogs % self.training_frequency == 0:
            self.train_call_count += 1

            s_batch, a_batch, r_batch, s2_batch, t_batch, indices, importance_weights = \
                self.buffer.sample()

            self.optim.zero_grad()
            torch.autograd.set_grad_enabled(True)
            s_batch.requires_grad_()
            gamma = torch.tensor([self.discount_gamma] * self.batch_size,
                     dtype=torch.float, device=DEVICE).view(self.batch_size, 1)

            # calculate loss
            loss = self.loss(s_batch, a_batch, s2_batch, r_batch, t_batch,
                             gamma)
            if importance_weights is not None:
                loss = loss * importance_weights
                for i in range(self.batch_size):
                    # importance weighting
                    # update priorities
                    self.buffer.update(i, loss[i].item())
            loss = loss.mean()
            loss.backward()

            # clip gradients
            if self.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.gradient_clipping)

            # update weights
            self.optim.step()
            current_loss = loss.item()
            torch.autograd.set_grad_enabled(False)

            if self.writer is not None:
                # plot loss
                self.writer.add_scalar('train/loss', current_loss,
                                       self.train_call_count)
                # plot min/max gradients
                max_grad_norm = -1.0
                min_grad_norm = 1000000.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        # TODO decide on norm
                        current_grad_norm = torch.norm(param.grad, 2)
                        if current_grad_norm > max_grad_norm:
                            max_grad_norm = current_grad_norm
                        if current_grad_norm < min_grad_norm:
                            min_grad_norm = current_grad_norm
                self.writer.add_scalar('train/min_grad', min_grad_norm,
                                       self.train_call_count)
                self.writer.add_scalar('train/max_grad', max_grad_norm,
                                       self.train_call_count)

            # update target net
            if self.target_model is not None and \
               self.train_call_count % self.target_update_rate == 0:
                self.target_model.load_state_dict(self.model.state_dict())

    def eps_scheduler(self):
        """ Linear epsilon decay """
        if self.is_training:
            self.epsilon = max(0, self.epsilon_start - (self.epsilon_start - self.epsilon_end) * float(self.num_dialogs) / float(self.train_dialogs))
            if self.writer is not None:
                self.writer.add_scalar('train/eps', self.epsilon, self.total_train_dialogs)

    def save(self):
        torch.save(self.model, 'model.pt')

    def load(self):
        if not os.path.isfile("./model.pt"):
            raise FileNotFoundError("There is no model for the RL policy which could be loaded.")
        self.model = torch.load('model.pt')
        if self.target_model is not None:
            self.target_model.load_state_dict(self.model.state_dict())


    def train(self):
        """ Sets module and its subgraph to training mode """
        super(DQNPolicy, self).train()
        self.model.train()
        if self.target_model is not None:
            self.target_model.train()

    def eval(self):
        """ Sets module and its subgraph to eval mode """
        super(DQNPolicy, self).eval()
        self.model.eval()
        if self.target_model is not None:
            self.target_model.eval()


