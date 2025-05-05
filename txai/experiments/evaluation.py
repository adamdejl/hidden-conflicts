import captum
import gc
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as stats
import warnings

from copy import deepcopy
from collections import defaultdict
from contextlib import nullcontext
from tqdm.auto import tqdm
from txai.utils import (
    infidelity_perturb_tabular_const,
    infidelity_perturb_image_patch_const,
)
from txai.utils.models import ResidualBlock

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Metric:
    def __init__(
        self,
        name,
        callback,
        metric_evaluator=None,
        args={},
        is_summary_metric=False,
        aggregate="mean",
        higher_is_better=True,
        run_stat_tests=False,
    ):
        self.name = name
        self.callback = callback
        self.args = args
        self.is_summary_metric = is_summary_metric
        self.metric_evaluator = metric_evaluator
        if not is_summary_metric:
            # Options only used for non-summary metrics
            self.aggregate = aggregate
            self.higher_is_better = higher_is_better
            self.run_stat_tests = run_stat_tests

    def __call__(
        self,
        model,
        data_loader,
        data_x=None,
        targets=None,
        attributions=None,
        exp_method=None,
        **kwargs,
    ):
        return self.callback(
            metric_evaluator=self.metric_evaluator,
            model=model,
            data_loader=data_loader,
            data_x=data_x,
            targets=targets,
            attributions=attributions,
            exp_method=exp_method,
            **self.args,
            **kwargs,
        )


class RuntimeMetric(Metric):
    def __init__(self):
        # Runtime tracking implemented in ExplanationMetricEvaluator
        super().__init__("Runtime", lambda: -1, higher_is_better=False)


class MlpConflictPrevalenceMetric(Metric):
    def __init__(self):
        def eval_conflict_units(model, x):
            assert isinstance(model, nn.Sequential), "Model must be a Sequential model"

            total_units = 0
            inactive_units = 0
            current_as = x
            for i in range(len(model)):
                curr_layer = model[i]
                next_layer = model[i + 1] if i + 1 < len(model) else None

                if isinstance(curr_layer, ResidualBlock):
                    block_total_units, block_inactive_units = eval_conflict_units(
                        curr_layer.block, current_as
                    )
                    total_units += block_total_units
                    inactive_units += block_inactive_units
                elif isinstance(curr_layer, nn.Linear) and (
                    isinstance(next_layer, nn.ReLU) or isinstance(next_layer, nn.GELU)
                ):
                    # The considered kinds of conflict can only occur in hidden layers
                    pos_as = (
                        current_as @ F.relu(curr_layer.weight.T)
                        + current_as @ F.relu(-curr_layer.weight.T)
                        + F.relu(curr_layer.bias)
                    ) >= 1e-5
                    neg_as = (
                        current_as @ F.relu(-curr_layer.weight.T)
                        + current_as @ F.relu(curr_layer.weight.T)
                        + F.relu(curr_layer.bias)
                    ) >= 1e-5
                    next_as = curr_layer(current_as)
                    total_units += torch.numel(next_as)

                    if isinstance(next_layer, nn.ReLU):
                        inactive_as = next_as < 0
                        inactive_units += (inactive_as & pos_as & neg_as).sum().item()
                    elif isinstance(next_layer, nn.GELU):
                        inactive_as = next_as < -0.4
                        inactive_units += (inactive_as & pos_as & neg_as).sum().item()

                current_as = curr_layer(current_as)

            assert torch.allclose(
                current_as, model(x)
            ), "Model output does not match the expected final layer output"
            return total_units, inactive_units

        def conflict_prevalence_callback(model, data_loader, **kwargs):
            total_units = 0
            inactive_units = 0
            i = 0
            for x, y in (batch_progress := tqdm(data_loader, leave=False)):
                i += 1
                batch_progress.set_description(f"Batch {i}")
                x, y = x.to(DEVICE), y.to(DEVICE)

                batch_total_units, batch_inactive_units = eval_conflict_units(model, x)
                total_units += batch_total_units
                inactive_units += batch_inactive_units

            return inactive_units / total_units

        super().__init__(
            name="MLP Conflict Prevalence",
            callback=conflict_prevalence_callback,
            is_summary_metric=True,
            aggregate="mean",
            higher_is_better=True,
            run_stat_tests=False,
        )


class TabularInfidelityMetric(Metric):
    def __init__(
        self,
        normalize=True,
        *,
        perturbation_scale,
        cat_resample_proba,
        categorical_variables_spans,
        n_perturb_samples,
        test_dl,
    ):
        def infidelity_callback(model, data_x, targets, attributions, **kwargs):
            return captum.metrics.infidelity(
                model,
                infidelity_perturb_tabular_const(
                    perturbation_scale,
                    cat_resample_proba=cat_resample_proba,
                    categorical_variables_spans=categorical_variables_spans,
                    test_dl=test_dl,
                ),
                data_x,
                attributions,
                target=targets,
                normalize=normalize,
                n_perturb_samples=n_perturb_samples,
            )

        super().__init__(
            name=f"Infidelity, S={perturbation_scale}, P={cat_resample_proba}",
            callback=infidelity_callback,
            is_summary_metric=False,
            aggregate="mean",
            higher_is_better=False,
            run_stat_tests=True,
        )


class InputStructuralInfidelityMetric(Metric):
    def __init__(
        self,
        *,
        neuron_selection_fun,
        n_neuron_samples,
        perturbation_scale,
        cat_resample_proba,
        categorical_variables_spans,
        n_perturb_samples,
        test_dl,
        neuron_aggregate="mean",
        normalize=True,
    ):
        self.selected_neurons = {}

        def structural_infidelity_callback(
            model, data_x, exp_method, seed=None, **kwargs
        ):
            assert seed is not None, "Seed must be provided for structural infidelity"

            if seed not in self.selected_neurons:
                self.selected_neurons[seed] = neuron_selection_fun(
                    model, data_x, n_neuron_samples, seed=seed
                )

            all_neuron_results = []
            for neuron_model in self.selected_neurons[seed]:
                neuron_model = deepcopy(neuron_model)

                torch.manual_seed(seed)
                attributions = exp_method(neuron_model, data_x, 0)

                torch.manual_seed(seed)
                neuron_results = captum.metrics.infidelity(
                    neuron_model,
                    infidelity_perturb_tabular_const(
                        perturbation_scale,
                        cat_resample_proba=cat_resample_proba,
                        categorical_variables_spans=categorical_variables_spans,
                        test_dl=test_dl,
                    ),
                    data_x,
                    attributions,
                    normalize=normalize,
                    n_perturb_samples=n_perturb_samples,
                )

                all_neuron_results.append(neuron_results)

            if neuron_aggregate == "mean":
                return sum(all_neuron_results) / len(all_neuron_results)
            elif neuron_aggregate == "sum":
                return sum(all_neuron_results)
            else:
                raise ValueError(
                    f"Invalid neuron aggregation method {neuron_aggregate}"
                )

        super().__init__(
            name=f"Input Structural Infidelity, S={perturbation_scale}",
            callback=structural_infidelity_callback,
            is_summary_metric=False,
            aggregate="mean",
            higher_is_better=False,
            run_stat_tests=True,
        )


class OutputStructuralInfidelityMetric(Metric):
    def __init__(
        self,
        *,
        layer_selection_fun,
        perturbation_scale,
        n_perturb_samples,
        layer_aggregate="mean",
        normalize=True,
    ):
        self.selected_layers = {}

        def structural_infidelity_callback(
            model, data_x, exp_method, targets, seed=None, **kwargs
        ):
            assert seed is not None, "Seed must be provided for structural infidelity"

            selected_layers = layer_selection_fun(model, data_x)

            all_layer_results = []
            for hidden_model, output_model in selected_layers:
                hidden_model = deepcopy(hidden_model)
                output_model = deepcopy(output_model)

                hidden_activations = hidden_model(data_x)

                torch.manual_seed(seed)
                attributions = exp_method(output_model, hidden_activations, targets)

                torch.manual_seed(seed)
                neuron_results = captum.metrics.infidelity(
                    output_model,
                    infidelity_perturb_tabular_const(
                        perturbation_scale,
                    ),
                    hidden_activations,
                    attributions,
                    target=targets,
                    normalize=normalize,
                    n_perturb_samples=n_perturb_samples,
                )

                all_layer_results.append(neuron_results)

            if layer_aggregate == "mean":
                return sum(all_layer_results) / len(all_layer_results)
            elif layer_aggregate == "sum":
                return sum(all_layer_results)
            else:
                raise ValueError(f"Invalid neuron aggregation method {layer_aggregate}")

        super().__init__(
            name=f"Output Structural Infidelity, S={perturbation_scale}",
            callback=structural_infidelity_callback,
            is_summary_metric=False,
            aggregate="mean",
            higher_is_better=False,
            run_stat_tests=True,
        )


class SensitivityMetric(Metric):
    def __init__(self, *, n_perturb_samples):
        def sensitivity_callback(model, data_x, targets, exp_method, **kwargs):
            return captum.metrics.sensitivity_max(
                lambda x, target: exp_method(model, x, target),
                data_x,
                n_perturb_samples=n_perturb_samples,
                target=targets,
            )

        super().__init__(
            name="Sensitivity",
            callback=sensitivity_callback,
            is_summary_metric=False,
            aggregate="mean",
            higher_is_better=False,
            run_stat_tests=False,
        )


class QuantusMetric(Metric):
    def __init__(
        self,
        name,
        metric_evaluator,
        args={},
        aggregate="mean",
        higher_is_better=True,
        run_stat_tests=False,
        ignore_warnings=False,
    ):
        def quantus_callback(
            metric_evaluator, model, data_x, targets, attributions, **kwargs
        ):
            with (
                warnings.catch_warnings(action="ignore")
                if ignore_warnings
                else nullcontext()
            ):
                return metric_evaluator(
                    model=model.cpu(),
                    x_batch=data_x.detach().cpu().numpy(),
                    y_batch=targets.detach().cpu().numpy(),
                    a_batch=attributions.detach().cpu().numpy(),
                    softmax=False,
                )

        super().__init__(
            name=name,
            callback=quantus_callback,
            metric_evaluator=metric_evaluator,
            args=args,
            is_summary_metric=False,
            aggregate=aggregate,
            higher_is_better=higher_is_better,
            run_stat_tests=run_stat_tests,
        )


class ExplanationMetricEvaluator:
    def __init__(self, model_loader, exp_methods, metrics, name=None):
        self.model_loader = model_loader
        self.exp_methods = exp_methods
        self.metrics = metrics
        self.summary_results = defaultdict(list)
        self.method_results = defaultdict(lambda: defaultdict(list))
        self.runtime_metric = RuntimeMetric()
        self.name = name

        # Prepare summary metrics
        processed_metrics = []
        for metric in self.metrics:
            if metric.is_summary_metric:
                train_metric = deepcopy(metric)
                train_metric.name += " (Train)"
                test_metric = deepcopy(metric)
                test_metric.name += " (Test)"
                processed_metrics.append(train_metric)
                processed_metrics.append(test_metric)
            else:
                processed_metrics.append(metric)
        self.metrics = processed_metrics

    def evaluate(self, seeds, data_loader, limit=None, train_dl=None):
        for seed in (seed_progress := tqdm(seeds, leave=False)):
            seed_progress.set_description(f"Seed {seed}")
            model = self.model_loader(seed)
            model.to(DEVICE)
            model.eval()
            for metric in [m for m in self.metrics if m.is_summary_metric]:
                if train_dl is not None and "(Train)" in metric.name:
                    result = metric(model, train_dl)
                else:
                    result = metric(model, data_loader)
                self.summary_results[metric].append(result)

            for exp_method in (exp_progress := tqdm(self.exp_methods, leave=False)):
                exp_progress.set_description(f"Method {exp_method.name}")

                # Prepare model for explanation method application
                torch.manual_seed(seed)
                np.random.seed(seed)
                model.to(DEVICE)
                exp_model = deepcopy(model)
                exp_model.to(DEVICE)
                exp_model.eval()

                start_time = time.time()
                xs = []
                targets = []
                attributions = []
                for i, (x, y) in enumerate(data_loader):
                    if limit is not None and i * data_loader.batch_size > limit:
                        # Reached sample limit
                        break
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    predictions = model(x).argmax(dim=-1)
                    method_results = exp_method(exp_model, x, predictions)
                    xs.append(x)
                    targets.append(predictions)
                    attributions.append(method_results)

                data_x = torch.cat(xs)[:limit]
                targets = torch.cat(targets)[:limit]
                attributions = torch.cat(attributions)[:limit]

                end_time = time.time()
                self.method_results[self.runtime_metric][exp_method].append(
                    end_time - start_time
                )

                exp_metrics = [m for m in self.metrics if not m.is_summary_metric]
                for metric in (metric_progress := tqdm(exp_metrics, leave=False)):
                    metric_progress.set_description(f"Metric {metric.name}")

                    # Reset seeds for better reproducibility, as metrics may be stochastic
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    exp_model.to(DEVICE)
                    exp_model.eval()
                    exp_model.zero_grad()

                    results = []
                    for i, (x, y) in enumerate(data_loader):
                        x, y = x.to(DEVICE), targets[i * len(x) : (i + 1) * len(x)]
                        b_attrs = attributions[i * len(x) : (i + 1) * len(x)]
                        result = metric(
                            model,
                            None,  # No need for data loader for explanation metrics
                            x,
                            y,
                            b_attrs,
                            exp_method.__call__,
                            seed=seed,
                        )
                        results.append(result)

                    if isinstance(results[0], torch.Tensor):
                        results = torch.cat(results)
                        results = results.detach().cpu().numpy()
                    elif isinstance(results[0], np.ndarray):
                        results = np.concatenate(results)
                    else:
                        results = [elem for result in results for elem in result]
                        results = np.array(results)

                    if not (
                        results.ndim == 1
                        or (results.ndim == 2 and results.shape[1] == 1)
                    ):
                        warnings.warn(
                            f"Output of {metric.name} has unexpected shape {results.shape}, check for correctness"
                        )
                    if metric.aggregate == "mean":
                        results = sum(results) / len(results)
                    elif metric.aggregate == "sum":
                        results = sum(results)
                    else:
                        raise ValueError(
                            f"Invalid metric aggregation method {metric.aggregate}"
                        )
                    self.method_results[metric][exp_method].append(results)

                del exp_model
                del data_x
                del targets
                del attributions
                gc.collect()
                torch.cuda.empty_cache()

        # Move runtime to be the last item in the results dict
        self.method_results[self.runtime_metric] = self.method_results.pop(
            self.runtime_metric
        )

    def print_report(self, run_stat_tests=False):
        header_suffix = "" if self.name is None else f" for {self.name}"
        print(f"———————————————[ Evaluation results{header_suffix} ]———————————————")
        print()

        print(f"———————[ Summary metrics ]———————")
        for metric, results in self.summary_results.items():
            results = np.array(results)
            mean = round(results.mean(axis=0), 3)
            std = round(results.std(axis=0), 4)
            print(f"{metric.name}: {'{:.3f}'.format(mean)}±{'{:.4f}'.format(std)}")
        print()

        for metric, metric_results in self.method_results.items():
            header_text = f"{metric.name} ({'↑' if metric.higher_is_better else '↓'})"
            print(f"———————[ {header_text} ]———————")
            for exp_method, method_results in metric_results.items():
                results = np.array(method_results)
                mean = round(results.mean(axis=0), 3)
                std = round(results.std(axis=0), 4)
                print(
                    f"{exp_method.name}: {'{:.3f}'.format(mean)}±{'{:.4f}'.format(std)}"
                )
            print()

            if run_stat_tests and metric.run_stat_tests:
                print(f"———————[ {header_text} statistical test results ]———————")
                cafe_methods = [
                    method for method in self.exp_methods if "CAFE" in method.name
                ]
                baseline_methods = [
                    method for method in self.exp_methods if "CAFE" not in method.name
                ]
                for cafe_method in cafe_methods:
                    for baseline_method in baseline_methods:
                        cafe_results = np.array(metric_results[cafe_method])
                        baseline_results = np.array(metric_results[baseline_method])

                        less_result = stats.ttest_rel(
                            cafe_results, baseline_results, alternative="less"
                        ).pvalue
                        greater_result = stats.ttest_rel(
                            baseline_results, cafe_results, alternative="less"
                        ).pvalue

                        if less_result < greater_result:
                            significant = "" if less_result >= 0.05 else " [S]"
                            print(
                                f"{cafe_method.name} < {baseline_method.name}: {'{:.3f}'.format(round(less_result, 3))}{significant}"
                            )
                        else:
                            significant = "" if greater_result >= 0.05 else " [S]"
                            print(
                                f"{cafe_method.name} > {baseline_method.name}: {'{:.3f}'.format(round(greater_result, 3))}{significant}"
                            )
                print()
