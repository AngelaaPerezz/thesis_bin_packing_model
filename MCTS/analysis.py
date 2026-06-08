import json
import numpy as np
from collections import defaultdict
from itertools import product
from scipy import stats

from env import make_env
from mcts import mcts_search


# ─────────────────────────────────────────────
#  LIKELIHOOD
# ─────────────────────────────────────────────

def fit_kde(values):
    """Fit a KDE to a list of values, return a callable pdf."""
    values = np.array(values)
    values = values[~np.isnan(values)]
    if len(values) < 2 or np.std(values) < 1e-9:
        mu = np.mean(values)
        return lambda x: stats.norm.pdf(x, mu, 1e-3)
    return stats.gaussian_kde(values)


def log_likelihood_trial(human_coverage, human_corr, sim_results):
    """
    Log likelihood of one human trial given simulated distribution.
    Fits independent KDEs on bin_coverage and size_order_corr.
    """
    coverages = [r['bin_coverage'] for r in sim_results]
    corrs     = [r['size_order_corr'] for r in sim_results
                 if not np.isnan(r['size_order_corr'])]

    kde_cov = fit_kde(coverages)
    ll_cov  = np.log(max(kde_cov(human_coverage)[0], 1e-10))

    if np.isnan(human_corr) or len(corrs) < 2:
        ll_corr = 0.0
    else:
        kde_corr = fit_kde(corrs)
        ll_corr  = np.log(max(kde_corr(human_corr)[0], 1e-10))

    return ll_cov + ll_corr


def log_likelihood_condition(human_trials, sim_results_per_stimulus):
    """Total log likelihood for a condition across all trials."""
    total_ll = 0.0
    for trial in human_trials:
        sid = trial['stimulus_id']
        if sid not in sim_results_per_stimulus:
            continue
        ll = log_likelihood_trial(
            trial['bin_coverage'],
            trial['size_order_corr'],
            sim_results_per_stimulus[sid]
        )
        total_ll += ll
    return total_ll


# ─────────────────────────────────────────────
#  AIC / BIC
# ─────────────────────────────────────────────

def compute_aic_bic(log_likelihood, num_params, num_observations):
    aic = 2 * num_params - 2 * log_likelihood
    bic = num_params * np.log(num_observations) - 2 * log_likelihood
    return aic, bic


MODEL_PARAMS = {
    'full':     0,   # no free parameters
    'depth':    1,   # depth only
    'breadth':  1,   # breadth (item_limit) only
    'combined': 2,   # depth + breadth
}


# ─────────────────────────────────────────────
#  GRID SEARCH
# ─────────────────────────────────────────────

def grid_search(env, human_trials, stimuli_by_id, depth_values, item_limit_values,
                num_simulations=200, num_runs=50):
    """
    Grid search over (depth, item_limit) combinations.
    Returns dict (depth, item_limit) -> log_likelihood.
    """
    unique_stim_ids = list({t['stimulus_id'] for t in human_trials})
    n_obs           = len(human_trials)
    results         = {}
    total           = len(depth_values) * len(item_limit_values)
    done            = 0

    for depth, item_limit in product(depth_values, item_limit_values):
        print(f"  [{done+1}/{total}] depth={depth}, item_limit={item_limit}")

        sim_results_per_stimulus = {}
        for sid in unique_stim_ids:
            stimulus = stimuli_by_id[sid]
            sim_results_per_stimulus[sid] = simulate_stimulus(
                env, stimulus,
                num_simulations=num_simulations,
                max_depth=depth,
                max_items=item_limit,
                num_runs=num_runs,
            )

        ll              = log_likelihood_condition(human_trials, sim_results_per_stimulus)
        results[(depth, item_limit)] = ll
        done += 1

    return results


# ─────────────────────────────────────────────
#  HELD-OUT EVALUATION
# ─────────────────────────────────────────────

def evaluate_success_rate(env, human_trials, stimuli_by_id, depth, item_limit,
                           num_simulations=200, num_runs=50):
    """Compare human vs MCTS success rate for best-fitting parameters."""
    unique_stim_ids = list({t['stimulus_id'] for t in human_trials})

    sim_results_per_stimulus = {}
    for sid in unique_stim_ids:
        sim_results_per_stimulus[sid] = simulate_stimulus(
            env, stimuli_by_id[sid],
            num_simulations=num_simulations,
            max_depth=depth,
            max_items=item_limit,
            num_runs=num_runs,
        )

    human_success = np.mean([t['success'] for t in human_trials])
    mcts_successes = [r['success']
                      for sid in unique_stim_ids
                      for r in sim_results_per_stimulus[sid]]
    mcts_success  = np.mean(mcts_successes)

    return human_success, mcts_success


# ─────────────────────────────────────────────
#  MAIN ANALYSIS
# ─────────────────────────────────────────────

def load_data(human_data_path, stimuli_path):
    """
    human_data: list of trials
    [
      {
        "stimulus_id": "stim_01",
        "condition": {"time_pressure": "high", "difficulty": "hard"},
        "bin_coverage": 0.72,
        "size_order_corr": 0.45,
        "success": true
      }, ...
    ]

    stimuli: dict of stimulus_id -> stimulus dict
    {
      "stim_01": {
        "bin_size": [12, 12],
        "items": [{"id": 1, "width": 3, "height": 4}, ...]
      }, ...
    }
    """
    with open(human_data_path) as f:
        human_data = json.load(f)
    with open(stimuli_path) as f:
        stimuli = json.load(f)
    return human_data, stimuli


def split_by_condition(human_data):
    conditions = defaultdict(list)
    for trial in human_data:
        key = (trial['condition']['time_pressure'], trial['condition']['difficulty'])
        conditions[key].append(trial)
    return conditions


def run_analysis(human_data_path, stimuli_path,
                 max_bin_size=[13, 13],
                 max_num_items=16,
                 depth_values=[1, 2, 3, 5, 7, 10, None],
                 item_limit_values=[2, 3, 4, 5, 7, 10, None],
                 num_simulations=200,
                 num_runs=50):

    env        = make_env(max_bin_size, max_num_items)
    human_data, stimuli_by_id = load_data(human_data_path, stimuli_path)
    conditions = split_by_condition(human_data)

    all_results = {}

    for condition, trials in conditions.items():
        print(f"\n=== Condition: {condition} ===")
        print(f"  {len(trials)} human trials")
        n_obs = len(trials)

        ll_grid  = grid_search(env, trials, stimuli_by_id,
                               depth_values, item_limit_values,
                               num_simulations=num_simulations,
                               num_runs=num_runs)

        # ── Model comparison with AIC/BIC ──
        # Full model: depth=None, item_limit=None
        ll_full = ll_grid.get((None, None), float('-inf'))

        # Best depth-only model (item_limit=None)
        best_depth_key = max(
            [(d, None) for d in depth_values],
            key=lambda k: ll_grid.get(k, float('-inf'))
        )
        ll_depth = ll_grid.get(best_depth_key, float('-inf'))

        # Best breadth-only model (depth=None)
        best_breadth_key = max(
            [(None, b) for b in item_limit_values],
            key=lambda k: ll_grid.get(k, float('-inf'))
        )
        ll_breadth = ll_grid.get(best_breadth_key, float('-inf'))

        # Best combined model
        best_combined_key = max(
            [(d, b) for d in depth_values for b in item_limit_values
             if d is not None and b is not None],
            key=lambda k: ll_grid.get(k, float('-inf'))
        )
        ll_combined = ll_grid.get(best_combined_key, float('-inf'))

        model_comparison = {}
        for name, ll, key in [
            ('full',     ll_full,     (None, None)),
            ('depth',    ll_depth,    best_depth_key),
            ('breadth',  ll_breadth,  best_breadth_key),
            ('combined', ll_combined, best_combined_key),
        ]:
            k        = MODEL_PARAMS[name]
            aic, bic = compute_aic_bic(ll, k, n_obs)
            model_comparison[name] = {
                'params': key, 'log_likelihood': ll, 'AIC': aic, 'BIC': bic
            }
            print(f"  {name:10s}  params={key}  LL={ll:.2f}  AIC={aic:.2f}  BIC={bic:.2f}")

        best_model = min(model_comparison, key=lambda m: model_comparison[m]['BIC'])
        print(f"\n  Best model by BIC: {best_model}")

        best_depth, best_item_limit = model_comparison[best_model]['params']
        human_sr, mcts_sr = evaluate_success_rate(
            env, trials, stimuli_by_id,
            best_depth, best_item_limit,
            num_simulations=num_simulations,
            num_runs=num_runs,
        )
        print(f"  Human success rate: {human_sr:.3f}")
        print(f"  MCTS success rate:  {mcts_sr:.3f}")

        all_results[str(condition)] = {
            'model_comparison': model_comparison,
            'best_model':       best_model,
            'human_success_rate': human_sr,
            'mcts_success_rate':  mcts_sr,
            'll_grid': {str(k): v for k, v in ll_grid.items()},
        }

    with open('mcts_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to mcts_results.json")

    return all_results



if __name__ == '__main__':
    results = run_analysis(
        human_data_path='human_data.json',
        stimuli_path='stimuli.json',
        max_bin_size=[13, 13],
        max_num_items=16,
        depth_values=[1, 2, 3, 5, 7, 10, None],
        item_limit_values=[2, 3, 4, 5, 7, 10, None],
        num_simulations=200,
        num_runs=50,
    )
