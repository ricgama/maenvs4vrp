"""
PPO implementation adapted from: https://pettingzoo.farama.org/tutorials/cleanrl/
and https://github.com/vwxyzjn/cleanrl

"""

import os
import sys
sys.path.insert(0, '../')

import argparse
from distutils.util import strtobool
import yaml
from tqdm import tqdm

import numpy as np

import time
import random
import os.path as osp
import os
import torch
from tensordict import TensorDict

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F

import wandb

#from ml_collections import config_dict
import importlib

from attention_model.policy_net_am import ActionCriticNet

def save_model_state_dict(save_path, model_policy):
    # save the policy state dict
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state_dict = model_policy.to("cpu").state_dict()
    torch.save(state_dict, save_path)

def set_random_seed(seed, torch_deterministic):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def train(args, writer):

    """ ENV SETUP """

    #for CVRP
    if args.vrp_env == 'cvrp':
        feature_list = yaml.safe_load("""
                                    nodes_static:
                                        x_coordinate:
                                            feat: x_coordinate
                                            norm:
                                        y_coordinate:
                                            feat: y_coordinate
                                            norm:
                                        demand:
                                            feat: demand
                                            norm:
                                        is_depot:
                                            feat: is_depot
                                            norm:
                                    agent:
                                        - remaining_capacity
                                                            """)
    #for TOP
    elif args.vrp_env == 'top':
        feature_list = yaml.safe_load("""
                                    nodes_static:
                                        x_coordinate:
                                            feat: x_coordinate
                                            norm:
                                        y_coordinate:
                                            feat: y_coordinate
                                            norm:
                                        profits:
                                            feat: profits
                                            norm:
                                        is_depot:
                                            feat: is_depot
                                            norm:
                                    agent:
                                        - frac_current_profit
                                        - frac_current_time
                                                            """)
    else:
        raise Warning("define feature_list for this environment")

    num_agents = args.num_agents
    num_nodes = args.num_nodes
    num_steps = args.num_steps
    n_envs = args.batch_size

    env_agent_selector_module_name = f'maenvs4vrp.environments.{args.vrp_env}.env_agent_selector'
    if args.selection == 'rand':
        env_agent_selector = importlib.import_module(env_agent_selector_module_name).RandomSelector()
    elif args.selection == 'single':
        env_agent_selector = importlib.import_module(env_agent_selector_module_name).AgentSelector()
    elif args.selection == 'stime':
        env_agent_selector = importlib.import_module(env_agent_selector_module_name).SmallestTimeAgentSelector()

    observations_module_name = f'maenvs4vrp.environments.{args.vrp_env}.observations'
    observations = importlib.import_module(observations_module_name).Observations(feature_list)

    generator_module_name = f'maenvs4vrp.environments.{args.vrp_env}.instances_generator'
    generator = importlib.import_module(generator_module_name).InstanceGenerator(device=args.env_device)

    environment_module_name = f'maenvs4vrp.environments.{args.vrp_env}.env'
    environment_module = importlib.import_module(environment_module_name)

    env_agent_reward_module_name = f'maenvs4vrp.environments.{args.vrp_env}.env_agent_reward'
    reward_evaluator = importlib.import_module(env_agent_reward_module_name).SparseReward()

    env = environment_module.Environment(instance_generator_object=generator,
                    obs_builder_object=observations,
                    agent_selector_object=env_agent_selector,
                    reward_evaluator=reward_evaluator,
                    device=args.env_device,
                    batch_size = args.batch_size,
                    seed=args.seed)

    if args.val_set == 'None':
        eval_generator = importlib.import_module(generator_module_name).InstanceGenerator(device=args.env_device)
    else:
        set_of_instances = set(generator.get_list_of_benchmark_instances()[args.val_set]['validation'])
        eval_generator = importlib.import_module(generator_module_name).InstanceGenerator(list_of_instances=set_of_instances,
                                                                                          device=args.env_device)
        args.eval_batch_size = None

    eval_env = environment_module.Environment(instance_generator_object=eval_generator,
                    obs_builder_object=observations,
                    agent_selector_object=env_agent_selector,
                    reward_evaluator=reward_evaluator,
                    device=args.env_device,
                    batch_size = args.eval_batch_size,
                    seed=args.eval_seed)

    nodes_static_obs_dim = env.obs_builder.get_nodes_static_feat_dim()
    nodes_dynamic_obs_dim = env.obs_builder.get_nodes_dynamic_feat_dim()
    agent_obs_dim = env.obs_builder.get_agent_feat_dim()
    agents_obs_dim = env.obs_builder.get_other_agents_feat_dim()
    global_obs_dim = env.obs_builder.get_global_feat_dim()

    """ ALGO LOGIC: EPISODE STORAGE"""
    start_time = time.time()

    policy_critic_net = ActionCriticNet(nodes_feat_dim=nodes_static_obs_dim,
                                        agent_feat_dim=agent_obs_dim,
                                        agents_feat_dim=agents_obs_dim,
                                        global_feat_dim=global_obs_dim,
                                        hidden_dim=128).to(args.device)
    optimizer = optim.Adam(policy_critic_net.parameters(), lr=args.learning_rate, eps=1e-5)

    best_lb_total_return = -10000000

    """ TRAINING LOGIC """
    # epoch-level accumulators — reset every iter_count episodes
    ep_loss = ep_pg_loss = ep_vloss = ep_ent = ep_rew = ep_nvnodes = ep_nagent = 0

    # train for n number of episodes
    pbar = tqdm(range(args.total_episodes))
    for episode in pbar:

        if args.anneal_lr:
            frac = 1.0 - (episode - 1.0) / args.total_episodes
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # collect an episode
        with torch.no_grad():

            # collect episodes
            td = env.reset_agent_select_observe(num_agents=num_agents,
                           num_nodes=num_nodes,
                           sample_type='random',
                           force_visit=args.force_visit,
                           seed=args.seed+episode,
                           obs_list=['agent_cur_node_idx', 'nodes_static', 'action_mask', 'agent'])

            rb_node_dyn_obs = torch.zeros((num_steps, n_envs, num_nodes, nodes_dynamic_obs_dim)).to(args.device) if nodes_dynamic_obs_dim > 0 else None
            rb_nodes_static_obs = torch.zeros((num_steps, n_envs, num_nodes, nodes_static_obs_dim)).to(args.device)
            rb_actions_mask = torch.zeros((num_steps, n_envs, num_nodes), dtype=torch.bool).to(args.device)
            rb_self_obs = torch.zeros((num_steps, n_envs, agent_obs_dim)).to(args.device)
            rb_global_obs = torch.zeros((num_steps, n_envs, global_obs_dim)).to(args.device) if global_obs_dim > 0 else None
            rb_cur_node_idx = torch.zeros((num_steps, n_envs), dtype=torch.long).to(args.device)
            rb_step_mask = torch.zeros((num_steps, n_envs), dtype=torch.bool).to(args.device)

            rb_actions = torch.zeros((num_steps, n_envs), dtype=torch.long).to(args.device)
            rb_logprobs = torch.zeros((num_steps, n_envs)).to(args.device)
            rb_rewards = torch.zeros((num_steps, n_envs)).to(args.device)
            rb_entropy = torch.zeros((num_steps, n_envs)).to(args.device)

            rb_values = torch.zeros((num_steps, n_envs)).to(args.device)
            final_reward =  torch.zeros(n_envs).to(args.device)
            step_mask = torch.ones(n_envs, dtype=torch.bool).to(args.device)

            node_stat_obs = td['observations']['nodes_static_obs'].to(args.device)

            policy_critic_net.policy.make_cache_(nodes_obs=node_stat_obs)

            step_idx = 0
            while not td["done"].all():

                # rollover the observation
                try:
                    node_dyn_obs = td['observations']['node_dynamic_obs'].to(args.device)
                except Exception:
                    node_dyn_obs = None
                try:
                    action_mask = td['observations']['action_mask'].to(args.device)
                except Exception:
                    action_mask = None
                try:
                    self_obs = td['observations']['agent_obs'].to(args.device)
                except Exception:
                    self_obs = None
                try:
                    global_obs = td['observations']['global_obs'].to(args.device)
                except Exception:
                    global_obs = None
                cur_node_idx = td['observations']['agent_cur_node_idx'].to(args.device)

                # get action from the agent
                action, logprobs, entropy, values = policy_critic_net.get_action_and_logs(nodes_dyn_obs=node_dyn_obs,
                                                                    self_obs=self_obs,
                                                                    global_obs=global_obs,
                                                                    cur_node_idx=cur_node_idx,
                                                                    action_mask=action_mask)

                td['next_action'] = action.unsqueeze(1).to(args.env_device)

                # execute the environment and log data
                td = env.step_agent_select_observe(td, obs_list=['agent_cur_node_idx', 'nodes_static', 'action_mask', 'agent'])

                if rb_node_dyn_obs is not None:
                    rb_node_dyn_obs[step_idx] = node_dyn_obs
                rb_nodes_static_obs[step_idx] = node_stat_obs
                rb_actions_mask[step_idx] = action_mask
                rb_self_obs[step_idx] = self_obs
                if rb_global_obs is not None:
                    rb_global_obs[step_idx] = global_obs
                rb_cur_node_idx[step_idx] = cur_node_idx.squeeze(-1)

                rb_step_mask[step_idx] = step_mask
                rb_rewards[step_idx] = (td['reward'].squeeze(1) + td['penalty'].squeeze(1)).to(args.device)

                rb_actions[step_idx] = action.to(torch.long)
                rb_logprobs[step_idx] = logprobs
                rb_entropy[step_idx] = entropy
                rb_values[step_idx] = values.squeeze(1)
                step_mask = (~td['done']).to(args.device)
                step_idx += 1

            final_reward = rb_rewards.detach().sum(0)
            not_visited_nodes = env.td_state['nodes']['active_nodes_mask'].sum(-1).float() - 1
            number_used_agents = env.td_state['agents']['visited_nodes'].sum(-1).gt(1).sum(-1).float()

        # compute advantages
        with torch.no_grad():
            if args.gae:
                gae = 0
                rb_advantages = torch.zeros_like(rb_rewards).to(args.device)
                for step in reversed(range(rb_rewards.shape[0]-1)):
                    delta = rb_rewards[step] + args.gamma * rb_values[step + 1] * rb_step_mask[step + 1] - \
                            rb_values[step]
                    rb_advantages[step] = gae = delta + args.gamma * args.gae_lambda * rb_step_mask[step + 1] * gae

                rb_returns = rb_advantages + rb_values
            else:
                rb_returns = torch.zeros_like(rb_rewards).to(args.device)
                for step in reversed(range(rb_rewards.shape[0]-1)):
                    rb_returns[step] = rb_returns[step + 1] * args.gamma * rb_step_mask[step + 1] + rb_rewards[step]
                rb_advantages = rb_returns - rb_values

        # For explained variance after the update
        b_values_flat = rb_values[rb_step_mask]
        b_returns_flat = rb_returns[rb_step_mask]
        n_samples = b_values_flat.size(0)
        assert n_envs >= args.num_minibatches, (
            "num_envs ({}) must be >= num_minibatches ({}).".format(n_envs, args.num_minibatches))

        # Environment-based mini-batches (consistent with RLOR):
        # encode the graph ONCE per env-group, then expand cache across that group's transitions.
        # This avoids re-running the encoder for every shuffled mini-batch of mixed transitions.
        envsperbatch = n_envs // args.num_minibatches
        envinds = np.arange(n_envs)
        env_static_obs = rb_nodes_static_obs[0]  # static obs are identical for all steps

        clip_fracs = []
        policy_critic_net.train()
        for repeat in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, n_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]                    # [envsperbatch]
                mb_step_mask = rb_step_mask[:, mbenvinds]         # [num_steps, envsperbatch]

                # r_inds[i] = local env index for transition i (maps transitions → env cache slot)
                r_inds = torch.arange(envsperbatch, device=args.device).unsqueeze(0)\
                               .repeat(num_steps, 1)[mb_step_mask]  # [n_active]

                # Encode graph ONCE for this env-group (gradients flow through encoder)
                policy_critic_net.policy.make_cache_(nodes_obs=env_static_obs[mbenvinds])
                # Expand cache from [envsperbatch, ...] to [n_active, ...]
                policy_critic_net.policy.expand_cache_(r_inds)

                # Gather all valid transitions for these envs
                batch_node_dyn_obs   = rb_node_dyn_obs[:, mbenvinds][mb_step_mask] if rb_node_dyn_obs is not None else None
                batch_actions_mask   = rb_actions_mask[:, mbenvinds][mb_step_mask]
                batch_self_obs       = rb_self_obs[:, mbenvinds][mb_step_mask]
                batch_global_obs     = rb_global_obs[:, mbenvinds][mb_step_mask] if rb_global_obs is not None else None
                batch_cur_node_idx   = rb_cur_node_idx[:, mbenvinds][mb_step_mask]
                batch_actions        = rb_actions[:, mbenvinds][mb_step_mask]
                batch_logprobs       = rb_logprobs[:, mbenvinds][mb_step_mask]
                batch_values         = rb_values[:, mbenvinds][mb_step_mask]
                batch_returns        = rb_returns[:, mbenvinds][mb_step_mask]
                batch_advantages     = rb_advantages[:, mbenvinds][mb_step_mask]

                # Evaluate OLD actions under the CURRENT policy (critical for PPO ratio)
                _, newlogprob, entropy, value = policy_critic_net.get_action_and_logs(
                    nodes_dyn_obs=batch_node_dyn_obs,
                    self_obs=batch_self_obs,
                    global_obs=batch_global_obs,
                    cur_node_idx=batch_cur_node_idx,
                    action_mask=batch_actions_mask,
                    action=batch_actions,
                )
                logratio = newlogprob - batch_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                advantages = batch_advantages
                if args.norm_adv:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - batch_returns) ** 2
                v_clipped = batch_values + torch.clamp(
                    value - batch_values, -args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - batch_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_critic_net.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values_flat.cpu().numpy(), b_returns_flat.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], episode)
        writer.add_scalar("losses/loss", loss.item(), episode)
        writer.add_scalar("losses/value_loss", v_loss.item(), episode)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), episode)
        writer.add_scalar("losses/entropy", entropy_loss.item(), episode)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), episode)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), episode)
        writer.add_scalar("losses/clipfrac", np.mean(clip_fracs), episode)
        writer.add_scalar("losses/explained_variance", explained_var, episode)
        #print("SPS:", int(episode / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(episode / (time.time() - start_time)), episode)

        #av_total_episodic_rew = torch.mean(total_episodic_rew).item()
        av_total_episodic_return = torch.mean(final_reward).item()
        av_not_visited_nodes = torch.mean(not_visited_nodes).item()
        av_number_used_agents = torch.mean(number_used_agents).item()

        #writer.add_scalar("train/episodic_reward", av_total_episodic_rew, episode)
        writer.add_scalar("train/episodic_return", av_total_episodic_return, episode)
        writer.add_scalar("train/episodic_not_visited_nodes", av_not_visited_nodes, episode)
        writer.add_scalar("train/episodic_number_used_agents", av_number_used_agents, episode)

        pbar.set_description("Episodic Return: {: .2f}, Not visited nodes: {}, Used agents: {}, Policy Loss: {:3.3f}, Value Loss: {:3.3f}, loss: {:3.3f}".format(av_total_episodic_return, av_not_visited_nodes, av_number_used_agents, pg_loss.item(), v_loss.item(), loss))

        # accumulate for epoch averages
        ep_loss    += loss.item()
        ep_pg_loss += pg_loss.item()
        ep_vloss   += v_loss.item()
        ep_ent     += entropy_loss.item()
        ep_rew     += av_total_episodic_return
        ep_nvnodes += av_not_visited_nodes
        ep_nagent  += av_number_used_agents

        if (episode + 1) % args.iter_count == 0:
            ep_num = episode // args.iter_count
            n      = args.iter_count
            writer.add_scalar("epoch/loss",               ep_loss    / n, ep_num)
            writer.add_scalar("epoch/policy_loss",        ep_pg_loss / n, ep_num)
            writer.add_scalar("epoch/value_loss",         ep_vloss   / n, ep_num)
            writer.add_scalar("epoch/entropy",            ep_ent     / n, ep_num)
            writer.add_scalar("epoch/return",             ep_rew     / n, ep_num)
            writer.add_scalar("epoch/not_visited_nodes",  ep_nvnodes / n, ep_num)
            writer.add_scalar("epoch/number_used_agents", ep_nagent  / n, ep_num)
            ep_loss = ep_pg_loss = ep_vloss = ep_ent = ep_rew = ep_nvnodes = ep_nagent = 0

        if episode % args.eval_num_print == 0 and args.val_set != 'None':
            print("\n-------------------------------------------\n")

            print (f'Running eval on validation set')
            latest_episodic_return, not_visited_nodes, number_used_agents = evaluate(args, writer, eval_env, policy_critic_net, episode)
            latest_not_visited_nodes = torch.mean(not_visited_nodes).item()
            latest_episodic_return =  torch.mean(latest_episodic_return).item()
            latest_number_used_agents =  torch.mean(number_used_agents).item()

            print (f'number not visited nodes: {latest_not_visited_nodes}')
            print (f'number of used agents: {latest_number_used_agents}')

            if latest_episodic_return > best_lb_total_return:
                print ('Old best model: {: .2f}'.format(best_lb_total_return))
                best_lb_total_return = latest_episodic_return
                print ('New best model: {: .2f}'.format(latest_episodic_return))
                print ('Saving new best model')
                save_model_state_dict(osp.join(args.log_path, "models/best_model_"+args.run_name+".zip"), policy_critic_net)
                policy_critic_net.to(args.device)
                print ('done')
            else:
                print ('No improvement')
                print (f'Latest model: {latest_episodic_return}')
                print (f'Current best model: {best_lb_total_return}')

            #writer.add_scalar("eval/latest_model_lb_total_reward", latest_lb_total_rew, episode)
            writer.add_scalar("eval/best_model_lb_total_reward", best_lb_total_return, episode)
            #writer.add_scalar("eval/episodic_reward:", latest_lb_total_rew, episode)
            writer.add_scalar("eval/episodic_return", latest_episodic_return, episode)
            writer.add_scalar("eval/episodic_not_visited_nodes", latest_not_visited_nodes, episode)
            writer.add_scalar("eval/episodic_number_used_agents", latest_number_used_agents, episode)
            print("\n-------------------------------------------\n")

    print ('saving latest model')
    save_model_state_dict(osp.join(args.log_path, "models/latest_model_"+args.run_name+".zip"), policy_critic_net)
    policy_critic_net.to(args.device)
    print ('done')
    writer.close()

def evaluate(args, writer, eval_env, policy, ep):
    policy.eval()

    total_reward = []
    not_visited_nodes = []
    number_used_agents = []

    with torch.no_grad():
        for instance_name in eval_env.inst_generator.list_of_instances:

            td = eval_env.reset_agent_select_observe(num_agents=args.num_agents,
                                num_nodes=args.num_nodes,
                                force_visit=args.force_visit,
                                sample_type='saved',
                                instance_name=instance_name,
                                seed=0,
                                obs_list=['agent_cur_node_idx', 'nodes_static', 'action_mask', 'agent'])

            f_reward = []
            node_stat_obs = td['observations']['nodes_static_obs'].to(args.device)
            policy.policy.make_cache_(nodes_obs=node_stat_obs)

            while not td["done"].all():

                # rollover the observation
                try:
                    node_dyn_obs = td['observations']['node_dynamic_obs'].to(args.device)
                except Exception:
                    node_dyn_obs = None
                try:
                    action_mask = td['observations']['action_mask'].to(args.device)
                except Exception:
                    action_mask = None
                try:
                    self_obs = td['observations']['agent_obs'].to(args.device)
                except Exception:
                    self_obs = None
                try:
                    global_obs = td['observations']['global_obs'].to(args.device)
                except Exception:
                    global_obs = None
                cur_node_idx = td['observations']['agent_cur_node_idx'].to(args.device)

                # get action from the agent
                action, _, _ = policy.policy.get_action_and_logs(nodes_obs=node_dyn_obs,
                                                            self_obs=self_obs,
                                                            global_obs=global_obs,
                                                            cur_node_idx=cur_node_idx,
                                                            action_mask=action_mask,
                                                            deterministic=True)

                # execute the environment and log data
                td['next_action'] = action.unsqueeze(1).to(args.env_device)
                td = eval_env.step_agent_select_observe(td, obs_list=['agent_cur_node_idx', 'nodes_static', 'action_mask', 'agent'])

                f_reward.append(td['reward'] + td['penalty'])

            total_reward.append(torch.cat(f_reward, dim=1).sum(-1))
            not_visited_nodes.append(eval_env.td_state['nodes']['active_nodes_mask'].sum(-1).float() - 1)
            number_used_agents.append(eval_env.td_state['agents']['visited_nodes'].sum(-1).gt(1).sum(-1).float())

        total_reward = torch.cat(total_reward).mean()
        not_visited_nodes = torch.cat(not_visited_nodes).mean()
        number_used_agents = torch.cat(number_used_agents).mean()

        writer.add_scalar("eval/episodic_return", total_reward, ep)
        writer.add_scalar("eval/episodic_not_visited_nodes", not_visited_nodes, ep)
        writer.add_scalar("eval/episodic_number_used_agents", number_used_agents, ep)

    print("Reward on test dataset: {:5.2f}".format(total_reward))
    return total_reward, not_visited_nodes, number_used_agents


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vrp_env", type=str, default="cvrp", help="select the vrp environment to train on")
    parser.add_argument("--num_agents", type=int, default=3, help="number of agents")
    parser.add_argument("--num_nodes", type=int, default=51, help="number of nodes")
    parser.add_argument("--selection", type=str, default="single", choices=['rand', 'single', 'stime'], help="next agent selection strategy")
    parser.add_argument("--val_set", type=str, default='None', help="validation set")
    args = parser.parse_args()
    return args


def get_args():
    args = parse_args()
    args.model_name = 'am_ppo_model'
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.env_device = torch.device("cpu")  # env runs on CPU; policy/obs on args.device
    args.ent_coef = 0.01
    args.vf_coef = 0.5
    args.clip_coef = 0.05
    args.gae = True
    args.gamma = 0.99
    args.gae_lambda = 0.95
    args.batch_size = 512
    args.eval_batch_size = 512

    args.hidden_dim = 128
    args.n_envs = 128

    args.force_visit = False
    args.num_steps = args.num_nodes + args.num_agents + 1
    args.epoch_count = 100
    args.iter_count = 2500

    args.total_episodes = args.epoch_count*args.iter_count+1

    args.learning_rate = 1e-4
    args.update_epochs = 2

    args.anneal_lr = False
    args.max_grad_norm = 10
    args.norm_adv = False
    args.torch_deterministic = True
    args.seed = 2297
    args.log_path = 'runs'
    args.eval_seed = 9875

    args.num_minibatches = 8
    args.eval_num_episodes = 1
    args.eval_num_print = 2500
    args.time = time.strftime("%Y_%m_%d_%Hh%Mm")
    args.run_name = f"{args.model_name}_{args.vrp_env}_{args.selection}_{args.num_nodes}n_{args.num_agents}a_{args.time}"
    args.debug = True
    return args


def main(args):
    print("Training with args", args)

    if args.seed != None:
        set_random_seed(args.seed, args.torch_deterministic)

    if not args.debug:
        wandb.init(project="your project", entity="your entity",
                sync_tensorboard=True,
                config=vars(args),
                monitor_gym=False,
                save_code=False,
            )

    writer = SummaryWriter(f"{args.log_path}/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    train(args, writer)

if __name__ == "__main__":
    # main(parse_args())
    main(get_args())
