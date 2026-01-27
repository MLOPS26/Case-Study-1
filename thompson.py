import chex
import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from functools import partial
from consts import EPS
from globals import UserInfo, Char, State


def norm_playtime(arr: chex.Array, cid: int) -> chex.Array:
    max_playtime = jnp.max(arr) + EPS
    norm = arr[cid] / max_playtime
    return norm


@jit
def construct_feats(user: UserInfo, char: Char, char_id: int) -> chex.Array:
    feats = [
        user.skill_level,
        jnp.log1p(user.games_played),
        char.difficulty,
        char.execution_level,
        char.neutral_required,
        char.tier,
    ]
    feats.append(char.archetype_vec)
    skill_match = 1.0 - jnp.abs(user.skill_level - (1.0 - char.difficulty))

    feats.append(jnp.array([skill_match]))

    archetype_sim = jnp.dot(user.pref_archetype, char.archetype_vec)
    feats.append(jnp.array([archetype_sim]))

    tried_before = user.chars_attempted_mask[char_id]
    novelty_bonus = 1.0 - tried_before
    feats.append(jnp.array([novelty_bonus]))

    past_perf = jnp.where(tried_before > 0.5, user.wr[char_id], 0.5)

    feats.append(jnp.array([past_perf]))

    norm = norm_playtime(user.playtime, char_id)
    feats.append(jnp.array([norm]))

    return jnp.concatenate([jnp.atleast_1d(feat) for feat in feats])


@partial(jit, static_argnums=(2,))
def build_feats(user: UserInfo, chars: Char, n_chars: int):
    def build_single(cid: int):
        char = jax.tree.map(lambda x: x[cid], chars)
        return construct_feats(user, char, cid)

    return vmap(build_single)(jnp.arange(n_chars))


@jit
def sample_params(key: chex.PRNGKey, mu: chex.Array, Sigma: chex.Array) -> chex.Array:
    d = mu.shape[0]
    Lambda = Sigma + EPS * jnp.eye(d)
    theta = random.multivariate_normal(key, mu, Lambda)
    return theta


@jit
def compute_expected_rewards(thetas: chex.Array, feats: chex.Array) -> chex.Array:
    return vmap(jnp.dot)(thetas, feats)


@jit
def thompson_sample(
    key: chex.PRNGKey, state: State, feats: chex.Array
) -> tuple[chex.Array, chex.Array]:
    num_chars = feats.shape[0]
    keys = random.split(key, num_chars)

    thetas = vmap(sample_params)(keys, state.mu, state.Sigma)
    rewards = compute_expected_rewards(thetas, feats)
    return rewards, thetas


@jit
def update_posterior(
    state: State,
    char_id: int,
    feats: chex.Array,
    reward: float,
    noise_var: float = 1.0,
    use_adaptive_noise: bool = True,
) -> State:
    x = feats
    d = x.shape[0]
    mu_old = state.mu[char_id]
    sigma_old = state.Sigma[char_id]

    # might be numerically unstable, not sure... for noninvertivle matrices should check this later when not lazy
    # ugly and hacky but idk how to approx this outside of inv, solve and do op, then inv to undo

    Sigma_old_inv = jnp.linalg.inv(sigma_old + EPS * jnp.eye(d))
    Sigma_new_inv = Sigma_old_inv + (1.0 / noise_var) * jnp.outer(x, x)
    Sigma_new = jnp.linalg.inv(Sigma_new_inv)

    mu_new = Sigma_new @ (Sigma_old_inv @ mu_old + (reward / noise_var) * x)

    new_mu = state.mu.at[char_id].set(mu_new)
    new_Sigma = state.Sigma.at[char_id].set(Sigma_new)

    # TODO: figure out whether adaptive noise in gp is needed
    new_beta = None

    if use_adaptive_noise:
        new_beta = state.beta.at[char_id].add(1)
    return State(
        mu=new_mu,
        Sigma=new_Sigma,
        alpha=state.alpha,
        beta=new_beta if new_beta is not None else state.beta,
    )


@partial(jit, static_argnums=(2, 3))
def select_top_k_diverse(
    scores: chex.Array, archetypes: chex.Array, k: int, diversity_thresh: float
) -> chex.Array:
    n_chars = scores.shape[0]
    sorted_idx = jnp.argsort(-scores)

    def selection_step(carry, cand_idx):
        select, cnt = carry
        cand_idx = sorted_idx[cand_idx]

        done = cnt > k

        cand_arch = archetypes[cand_idx]

        def check_item_diversity(sel_idx):
            # may need a max bound here
            is_valid = sel_idx >= 0
            sel_arch = archetypes[sel_idx]
            # cos_sim w little eps to avoid div 0

            sim = jnp.dot(cand_arch, sel_arch) / (
                jnp.linalg.norm(cand_arch) * jnp.linalg.norm(sel_arch) + 1e-8
            )
            return jnp.where(is_valid, sim < diversity_thresh, True)

        all_diverse = jnp.all(vmap(check_item_diversity)(select))

        add_op = jnp.logical_and(jnp.logical_not(done), all_diverse)

        new_sel = jnp.where(add_op, select.at[cnt].set(cand_idx), select)
        new_cnt = jnp.where(add_op, cnt + 1, cnt)
        return (new_sel, new_cnt), None

    init = jnp.full(k, -1, dtype=jnp.int32)
    init = init.at[0].set(sorted_idx[0])

    (final_sel, null), null = jax.lax.scan(
        selection_step, (init, 1), jnp.arange(1, n_chars)
    )
    return final_sel


@jit
def compute_reward(
        won: bool, completed: bool, rating: float, playtime_mins: float, weights:chex.Array = jnp.array([0.3, 0.15, 0.25, 0.3])
) -> float:
    win_reward = jnp.where(won, weights[0], 0.0)
    completion_reward = jnp.where(completed, weights[1], 0.0)
    rating_reward = weights[2] * jnp.clip(rating / 5.0, 0.0, 1.0)

    engagement_reward = weights[3] * jnp.clip(jnp.log1p(playtime_mins) / 5.0, 0.0, 1.0)
    return win_reward + completion_reward + rating_reward + engagement_reward


@partial(jit, static_argnums=(4, 5))
def recommend_characters(
    key: chex.PRNGKey,
    state: State,
    user: UserInfo,
    characters: Char,
    n_chars: int,
    top_k: int = 3,
    diversity_threshold: float = 0.75,
) -> tuple[chex.Array, chex.Array]:
    features = build_feats(user, characters, n_chars)
    sampled_rewards, sampled_thetas = thompson_sample(key, state, features)

    selected = select_top_k_diverse(
        sampled_rewards, characters.archetype_vec, top_k, diversity_threshold
    )

    return selected, sampled_rewards


def init_thompson(n_chars: int, feature_dim: int, prior_var: float = 1.0) -> State:
    return State(
        mu=jnp.zeros((n_chars, feature_dim)),
        Sigma=jnp.tile(prior_var * jnp.eye(feature_dim), (n_chars, 1, 1)),
        alpha=jnp.ones(n_chars),
        beta=jnp.ones(n_chars),
    )


@jit
def batch_update_posterior(
    state: State,
    char_ids: chex.Array,
    features: chex.Array,
    rewards: chex.Array,
    noise_var: float = 1.0,
) -> State:
    def single_update(s, data):
        char_id, feat, reward = data
        return update_posterior(s, char_id, feat, reward, noise_var), None

    final_state, _ = jax.lax.scan(single_update, state, (char_ids, features, rewards))
    return final_state


