import numpy as np

from envs.multiagentenv import MultiAgentEnv


class MPEEnvWrapper(MultiAgentEnv):
    def __init__(
        self,
        scenario_name,
        seed=None,
        max_cycles=25,
        predator_agents=3,
        predator_prey=1,
        predator_obstacles=2,
        navigation_agents=3,
        navigation_landmarks=3,
        keep_away_agents=2,
        keep_away_landmarks=2,
        keep_away_opponents=1,
        continuous_actions=False,
        render_mode=None,
    ):
        self.scenario_name = scenario_name
        self.seed_value = seed
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)

        self._scenario_kwargs = {
            "predator_prey": {
                "num_good": predator_prey,
                "num_adversaries": predator_agents,
                "num_obstacles": predator_obstacles,
                "max_cycles": max_cycles,
                "continuous_actions": continuous_actions,
                "render_mode": render_mode,
            },
            "cooperative_navigation": {
                "N": navigation_agents,
                "local_ratio": 0.5,
                "max_cycles": max_cycles,
                "continuous_actions": continuous_actions,
                "render_mode": render_mode,
            },
            "keep_away": {
                "N": keep_away_agents,
                "max_cycles": max_cycles,
                "continuous_actions": continuous_actions,
                "render_mode": render_mode,
            },
        }

        self.env = None
        self.scripted_agents = []
        self.learning_agents = []
        self.all_agents = []
        self.n_agents = 0
        self.episode_limit = max_cycles
        self._last_obs = None
        self._last_state = None
        self._obs_size = None
        self._state_size = None
        self.n_actions = 5
        self.reset()

    def _build_env(self):
        if self.scenario_name == "predator_prey":
            from pettingzoo.mpe import simple_tag_v3

            env = simple_tag_v3.parallel_env(**self._scenario_kwargs["predator_prey"])
            learning_prefixes = ("adversary_",)
        elif self.scenario_name == "cooperative_navigation":
            from pettingzoo.mpe import simple_spread_v3

            env = simple_spread_v3.parallel_env(**self._scenario_kwargs["cooperative_navigation"])
            learning_prefixes = ("agent_",)
        elif self.scenario_name == "keep_away":
            from pettingzoo.mpe import simple_adversary_v3

            env = simple_adversary_v3.parallel_env(**self._scenario_kwargs["keep_away"])
            learning_prefixes = ("agent_",)
        else:
            raise ValueError(f"Unsupported MPE scenario: {self.scenario_name}")

        return env, learning_prefixes

    def _bootstrap(self):
        self.env, learning_prefixes = self._build_env()
        obs, _ = self.env.reset(seed=self.seed_value)
        self.all_agents = list(self.env.agents)
        self.learning_agents = [agent for agent in self.all_agents if agent.startswith(learning_prefixes)]
        self.scripted_agents = [agent for agent in self.all_agents if agent not in self.learning_agents]
        self.n_agents = len(self.learning_agents)
        self.n_actions = int(self.env.action_space(self.learning_agents[0]).n)
        self._last_obs = obs
        self._last_state = self._build_state(obs)
        self._obs_size = int(np.asarray(obs[self.learning_agents[0]], dtype=np.float32).size)
        self._state_size = int(self._last_state.size)

    def _build_state(self, obs_dict):
        parts = [np.asarray(obs_dict[agent], dtype=np.float32).reshape(-1) for agent in self.learning_agents]
        return np.concatenate(parts, axis=0)

    def _scripted_action(self, agent_name, obs):
        if self.scenario_name == "predator_prey":
            return int(self._rng.integers(self.n_actions))
        if self.scenario_name == "keep_away":
            return int(self._rng.integers(self.n_actions))
        return 0

    def step(self, actions):
        joint_actions = {}
        for idx, agent in enumerate(self.learning_agents):
            action = actions[idx]
            if isinstance(action, np.ndarray):
                action = int(action.item())
            joint_actions[agent] = int(action)

        for agent in self.scripted_agents:
            joint_actions[agent] = self._scripted_action(agent, self._last_obs[agent])

        next_obs, rewards, terminations, truncations, infos = self.env.step(joint_actions)
        self._last_obs = next_obs
        self._last_state = self._build_state(next_obs)

        coop_rewards = [float(rewards[agent]) for agent in self.learning_agents]
        reward = float(np.mean(coop_rewards))
        terminated = any(terminations.get(agent, False) or truncations.get(agent, False) for agent in self.learning_agents)
        info = {
            "individual_rewards_mean": reward,
            "episode_limit": all(truncations.get(agent, False) for agent in self.learning_agents),
        }
        return reward, terminated, info

    def get_obs(self):
        return [np.asarray(self._last_obs[agent], dtype=np.float32) for agent in self.learning_agents]

    def get_obs_agent(self, agent_id):
        return np.asarray(self._last_obs[self.learning_agents[agent_id]], dtype=np.float32)

    def get_obs_size(self):
        return self._obs_size

    def get_state(self):
        return self._last_state.copy()

    def get_state_size(self):
        return self._state_size

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(agent_id) for agent_id in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return [1] * self.n_actions

    def get_total_actions(self):
        return self.n_actions

    def reset(self):
        if self.env is None:
            self._bootstrap()
        else:
            obs, _ = self.env.reset(seed=self.seed_value)
            self._last_obs = obs
            self._last_state = self._build_state(obs)
        return self.get_obs(), self.get_state()

    def render(self):
        if self.render_mode is not None:
            return self.env.render()
        return None

    def close(self):
        if self.env is not None:
            self.env.close()

    def seed(self, seed=None):
        self.seed_value = seed
        self._rng = np.random.default_rng(seed)

    def save_replay(self):
        return None

    def get_stats(self):
        return {}
