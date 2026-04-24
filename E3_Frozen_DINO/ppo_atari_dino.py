"""
PPO Atari — Frozen DINO ViT-S/16 Encoder (E3)
===============================================
Based on CleanRL's ppo_atari.py (Huang et al., 2022).
Modified to replace NatureCNN with a frozen DINO ViT-S/16 encoder.

Usage:
    python ppo_atari_dino.py --seed 1
    python ppo_atari_dino.py --seed 1 --track  # with W&B logging
"""

import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# 1. ARGUMENT PARSING
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--exp-name", type=str, default="ppo_dino_frozen",
        help="name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=True)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)),
        default=False, nargs="?", const=True,
        help="toggle Weights & Biases logging")
    parser.add_argument("--wandb-project-name", type=str, default="xai-vision-rl",
        help="W&B project name")

    # Environment
    parser.add_argument("--env-id", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--total-timesteps", type=int, default=500_000,
        help="total environment steps (matched budget)")
    parser.add_argument("--num-envs", type=int, default=8,
        help="number of parallel envs")

    # PPO core
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="policy head LR")
    parser.add_argument("--num-steps", type=int, default=128,
        help="rollout length per env")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=True)
    parser.add_argument("--gamma", type=float, default=0.99,
        help="discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="GAE lambda")
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="PPO clip epsilon (proposal uses 0.1 or 0.2)")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=True)
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=True)
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="entropy bonus coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="value loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)

    # DINO specific
    parser.add_argument("--dino-model", type=str, default="dino_vits16",
        help="DINO model variant from torch.hub")
    parser.add_argument("--projection-dim", type=int, default=512,
        help="projection output dim (matches NatureCNN output)")

    # Checkpointing
    parser.add_argument("--save-interval", type=int, default=50,
        help="save checkpoint every N updates")
    parser.add_argument("--save-dir", type=str, default="checkpoints")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size
    return args


# ============================================================
# 2. ATARI WRAPPERS (standard from CleanRL / Gymnasium)
# ============================================================
def make_env(env_id, seed, idx, run_name):
    """Create a single Atari env with standard preprocessing."""
    def thunk():
        env = gym.make(env_id, render_mode=None)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Standard Atari wrappers
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env
    return thunk


# --- Standard Atari Wrappers ---
class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking random no-ops on reset."""
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Return max over last 2 frames and repeat action for `skip` frames."""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only for training."""
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Take FIRE action on reset for envs that require it."""
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class ClipRewardEnv(gym.Wrapper):
    """Clip rewards to {-1, 0, +1}."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, np.sign(reward), terminated, truncated, info


# ============================================================
# 3. DINO ENCODER + AGENT  ← THIS IS YOUR KEY CONTRIBUTION
# ============================================================
class DINOEncoder(nn.Module):
    """
    Frozen DINO ViT-S/16 encoder with input adapter for Atari frames.

    Input:  (batch, 4, 84, 84)  — 4 stacked grayscale Atari frames, uint8
    Output: (batch, 384)         — CLS token embedding from DINO

    The adapter handles:
      1. uint8 → float32 normalization
      2. 4-channel grayscale → 3-channel pseudo-RGB
      3. 84×84 → 224×224 bilinear resize
      4. ImageNet mean/std normalization (what DINO was trained with)
    """
    def __init__(self, model_name="dino_vits16"):
        super().__init__()

        # Load pretrained DINO from Facebook's repo
        self.dino = torch.hub.load(
            "facebookresearch/dino:main",
            model_name,
            pretrained=True,
        )

        # FREEZE all parameters — this is the whole point of E3
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()

        # ImageNet normalization constants
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Output dimensionality (ViT-S → 384)
        self.embed_dim = self.dino.embed_dim  # 384 for ViT-S/16

    def adapt_input(self, x):
        """
        Convert Atari frames to DINO-compatible input.

        Strategy for 4ch → 3ch: use the 3 most recent frames.
        Frame stack is [t-3, t-2, t-1, t], so indices [1, 2, 3] are most recent.
        Each becomes one RGB channel, preserving temporal info.
        """
        # (batch, 4, 84, 84) uint8 → float [0, 1]
        x = x.float() / 255.0

        # Take last 3 frames as 3 "channels" — preserves motion info
        x = x[:, 1:, :, :]  # (batch, 3, 84, 84)

        # Resize 84×84 → 224×224 (what DINO expects)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Normalize with ImageNet stats
        x = (x - self.img_mean) / self.img_std

        return x

    @torch.no_grad()
    def forward(self, x):
        """Extract frozen features. No gradients flow into DINO."""
        x = self.adapt_input(x)
        features = self.dino(x)  # (batch, 384) — CLS token output
        return features


class AgentDINO(nn.Module):
    """
    PPO Agent with frozen DINO encoder.

    Architecture:
        Atari (4,84,84) → DINOEncoder(frozen) → 384-d
                         → Projection(trainable) → 512-d
                         → Actor head  → action logits
                         → Critic head → state value
    """
    def __init__(self, envs, dino_model="dino_vits16", projection_dim=512):
        super().__init__()
        num_actions = envs.single_action_space.n

        # Frozen encoder
        self.encoder = DINOEncoder(model_name=dino_model)

        # Trainable projection: maps DINO features → shared representation
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.embed_dim, projection_dim),
            nn.ReLU(),
        )

        # Trainable policy head
        self.actor = nn.Linear(projection_dim, num_actions)

        # Trainable value head
        self.critic = nn.Linear(projection_dim, 1)

        # Initialize projection and heads with orthogonal init (CleanRL convention)
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for trainable layers."""
        for module in [self.projection, self.actor, self.critic]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                        nn.init.constant_(layer.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01 if module == self.actor else 1.0)
                nn.init.constant_(module.bias, 0.0)

    def get_features(self, x):
        """Encode observations → projected features."""
        with torch.no_grad():
            raw_features = self.encoder(x)       # (batch, 384) — frozen
        projected = self.projection(raw_features) # (batch, 512) — trainable
        return projected

    def get_value(self, x):
        """Get state value estimate."""
        return self.critic(self.get_features(x))

    def get_action_and_value(self, x, action=None):
        """
        Sample action (or evaluate given action) and return:
          - action, log_prob, entropy, value
        """
        features = self.get_features(x)
        logits = self.actor(features)
        probs = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(features)


# ============================================================
# 4. PPO TRAINING LOOP (standard CleanRL structure)
# ============================================================
def main():
    args = parse_args()
    run_name = f"{args.env_id.replace('/', '_')}__{args.exp_name}__seed{args.seed}__{int(time.time())}"

    # --- Logging setup ---
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(
            [f"|{k}|{v}|" for k, v in vars(args).items()]
        ),
    )

    # --- Seeding ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device}")

    # --- Environment ---
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)
    print(f"Action space: {envs.single_action_space.n} actions")

    # --- Agent ---
    agent = AgentDINO(
        envs,
        dino_model=args.dino_model,
        projection_dim=args.projection_dim,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Frozen params:    {frozen_params:,}")

    # Only optimize trainable parameters (projection + actor + critic)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, agent.parameters()),
        lr=args.learning_rate,
        eps=1e-5,
    )

    # --- Rollout storage ---
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # --- Start training ---
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting training: {args.total_timesteps} steps, {args.num_updates} updates")
    print(f"Batch size: {args.batch_size}, Minibatch size: {args.minibatch_size}")
    print(f"{'='*60}\n")

    for update in range(1, args.num_updates + 1):
        # --- Learning rate annealing ---
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / args.num_updates
            lr_now = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lr_now

        # ===========================
        # ROLLOUT PHASE — collect data
        # ===========================
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Agent selects action
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Step environment
            next_obs_np, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done).to(device)

            # Log completed episodes
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None and "episode" in info:
                        ep_return = info["episode"]["r"].item()
                        ep_length = info["episode"]["l"].item()
                        print(f"  step={global_step:>7d}  return={ep_return:.1f}  len={ep_length}")
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)

        # ===========================
        # GAE — compute advantages
        # ===========================
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # ===========================
        # OPTIMIZATION PHASE — PPO update
        # ===========================
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Debug: approx KL
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss (clipped surrogate)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef, args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, agent.parameters()),
                    args.max_grad_norm,
                )
                optimizer.step()

            # Early stopping on KL
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # --- Logging ---
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)

        if update % 10 == 0:
            print(
                f"Update {update}/{args.num_updates}  "
                f"step={global_step}  SPS={sps}  "
                f"pg_loss={pg_loss.item():.4f}  v_loss={v_loss.item():.4f}  "
                f"entropy={entropy_loss.item():.4f}"
            )

        # --- Checkpointing ---
        if update % args.save_interval == 0 or update == args.num_updates:
            ckpt_path = os.path.join(
                args.save_dir, f"{run_name}_update{update}.pt"
            )
            torch.save(
                {
                    "update": update,
                    "global_step": global_step,
                    "agent_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  → Saved checkpoint: {ckpt_path}")

    envs.close()
    writer.close()
    print(f"\nTraining complete. Total time: {(time.time()-start_time)/60:.1f} min")


if __name__ == "__main__":
    main()
