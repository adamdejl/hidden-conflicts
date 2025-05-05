import os
import torch

from abc import ABC, abstractmethod
from txai.utils import construct_feedforward_nn, eval_nn, eval_continuous_nn, train_nn
from txai.experiments import Metric, ExplanationMetricEvaluator
from txai.explainability import (
    ExplanationMethod,
    explain_cafe,
    explain_ixg,
    explain_lrp,
    explain_dl,
    explain_ig,
    explain_sg,
    explain_gs,
    explain_ks,
    explain_svs,
    explain_lime,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Experimenter(ABC):
    @abstractmethod
    def __init__(
        self,
        force_retrain=False,
        *,
        experiment_name,
        seeds,
        num_epochs,
        checkpoint_path_base,
        loss,
        optimizer_constructor=None,
        use_checkpoint=True,
    ):
        self.experiment_name = experiment_name
        self.seeds = seeds
        self.num_epochs = num_epochs
        self.checkpoint_path_base = checkpoint_path_base
        self.loss = loss
        self.optimizer_constructor = optimizer_constructor
        self.evaluator = None
        self.force_retrain = force_retrain
        self.training_confirmed = False
        self.use_checkpoint = use_checkpoint
        self.train_dl = None

    @abstractmethod
    def construct_model(self):
        pass

    def construct_checkpoint_path(self, seed):
        return f"{self.checkpoint_path_base}{seed}.pth"

    def get_model_loader(self, report_training_progress=False, val_dl=None):
        def model_loader(seed):
            torch.manual_seed(seed)
            model = self.construct_model()
            if self.use_checkpoint:
                checkpoint_path = self.construct_checkpoint_path(seed)
                if os.path.isfile(checkpoint_path) and not self.force_retrain:
                    # Load the saved checkpoint
                    model.load_state_dict(torch.load(checkpoint_path))
                    model.to(DEVICE)
                else:
                    # Train and save the model
                    if not self.training_confirmed:
                        checkpoint_exists = os.path.isfile(checkpoint_path)
                        warning_exists = (
                            " No existing checkpoints were found."
                            if not checkpoint_exists
                            else " WARNING: Existing checkpoint will be overwritten!"
                        )
                        confirmation = ""
                        while confirmation.strip().lower() not in ["y", "n"]:
                            confirmation = input(
                                f"This will train and save new models — continue?{warning_exists} (y/n): "
                            )
                            if confirmation.strip().lower() == "n":
                                raise KeyboardInterrupt("Aborted by user.")
                            elif confirmation.strip().lower() != "y":
                                print("Invalid input. Please enter 'y' or 'n'.")
                        self.training_confirmed = True

                    train_nn(
                        model,
                        self.train_dl,
                        self.num_epochs,
                        loss=self.loss,
                        optimizer_constructor=self.optimizer_constructor,
                        report_progress=report_training_progress,
                        val_dl=val_dl,
                    )
                    torch.save(model.state_dict(), checkpoint_path)
            return model

        return model_loader

    def get_explanation_methods(
        self, use_perturb_methods=True, feature_mask=None, method_filter=None
    ):
        if use_perturb_methods and feature_mask is None:
            raise ValueError(
                "Need to provide a feature mask when using perturbation methods."
            )

        grad_methods = [
            ExplanationMethod("Gradient x Input", explain_ixg),
            ExplanationMethod("LRP", explain_lrp),
            ExplanationMethod(
                "DeepLIFT Rescale", explain_dl, args={"multiply_by_inputs": True}
            ),
            ExplanationMethod(
                "GradientSHAP", explain_gs, args={"multiply_by_inputs": True}
            ),
            ExplanationMethod(
                "Integrated Gradients", explain_ig, args={"multiply_by_inputs": True}
            ),
            ExplanationMethod(
                "SmoothGrad", explain_sg, args={"multiply_by_inputs": True}
            ),
        ]
        perturb_methods = [
            ExplanationMethod(
                "KernelSHAP",
                explain_ks,
                args={"multiply_by_inputs": False, "feature_mask": feature_mask},
            ),
            ExplanationMethod(
                "Shapley Value Sampling",
                explain_svs,
                args={"multiply_by_inputs": False, "feature_mask": feature_mask},
            ),
            ExplanationMethod(
                "LIME",
                explain_lime,
                args={"multiply_by_inputs": False, "feature_mask": feature_mask},
            ),
        ]
        cafe_methods = [
            ExplanationMethod("CAFE (c = 0.0)", explain_cafe, args={"c": 0.0}),
            ExplanationMethod("CAFE (c = 0.25)", explain_cafe, args={"c": 0.25}),
            ExplanationMethod("CAFE (c = 0.5)", explain_cafe, args={"c": 0.5}),
            ExplanationMethod("CAFE (c = 0.75)", explain_cafe, args={"c": 0.75}),
            ExplanationMethod("CAFE (c = 1.0)", explain_cafe, args={"c": 1.0}),
        ]

        used_methods = grad_methods
        if use_perturb_methods:
            used_methods += perturb_methods
        used_methods += cafe_methods

        if method_filter is not None:
            used_methods = [m for m in used_methods if m.name not in method_filter]

        return used_methods

    def get_summary_metrics(self, binary=False, continuous=False):
        def eval_fun_for(metric, eval_fun=eval_nn):
            return lambda model, data_loader, **kwargs: getattr(
                eval_fun(model, data_loader, loss=self.loss), metric
            )

        if not continuous:
            metrics = [
                Metric("Loss", eval_fun_for("loss"), is_summary_metric=True),
                Metric("F1", eval_fun_for("f1"), is_summary_metric=True),
                Metric("Accuracy", eval_fun_for("accuracy"), is_summary_metric=True),
            ]

            if binary:
                metrics += [Metric("AUC", eval_fun_for("auc"), is_summary_metric=True)]

            return metrics

        metrics = [
            Metric(
                "Loss",
                eval_fun_for("loss", eval_continuous_nn),
                is_summary_metric=True,
            ),
            Metric(
                "RMSE", eval_fun_for("rmse", eval_continuous_nn), is_summary_metric=True
            ),
        ]

        return metrics

    def run_experiment(self, limit=None, rerun=False, *, exp_methods, metrics, test_dl):
        print(f"Running experiment {self.experiment_name}...")
        if not rerun and self.evaluator is not None:
            raise ValueError("Experiment already run. Set rerun=True to rerun.")

        self.evaluator = ExplanationMetricEvaluator(
            self.get_model_loader(), exp_methods, metrics, name=self.experiment_name
        )
        self.evaluator.evaluate(
            self.seeds, test_dl, limit=limit, train_dl=self.train_dl
        )

        print("Experiment complete!")

    def report_experiment_results(self, run_stat_tests=False):
        if self.evaluator is None:
            raise ValueError("No results to report. Run the experiment first.")

        self.evaluator.print_report(run_stat_tests=run_stat_tests)


class FeedforwardExperimenter(Experimenter):
    def __init__(
        self,
        force_retrain=False,
        *,
        experiment_name,
        seeds,
        num_epochs,
        checkpoint_path_base,
        loss,
        optimizer_constructor=None,
        data_dim,
        num_classes,
        num_layers,
        hidden_size,
        activation_fun,
        train_dl,
    ):
        super().__init__(
            experiment_name=experiment_name,
            seeds=seeds,
            num_epochs=num_epochs,
            checkpoint_path_base=checkpoint_path_base,
            loss=loss,
            optimizer_constructor=optimizer_constructor,
            force_retrain=force_retrain,
        )
        self.data_dim = data_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.activation_fun = activation_fun
        self.train_dl = train_dl

    def construct_model(self):
        network_architecture = (
            [self.data_dim] + [self.hidden_size] * self.num_layers + [self.num_classes]
        )
        model = construct_feedforward_nn(
            nn_dims=network_architecture,
            activation_fun=self.activation_fun,
        )
        return model


class GenericExperimenter(Experimenter):
    def __init__(
        self,
        force_retrain=False,
        *,
        experiment_name,
        seeds,
        num_epochs,
        checkpoint_path_base=None,
        loss,
        optimizer_constructor=None,
        model_constructor,
        train_dl,
        use_checkpoint=True,
    ):
        super().__init__(
            experiment_name=experiment_name,
            seeds=seeds,
            num_epochs=num_epochs,
            checkpoint_path_base=checkpoint_path_base,
            loss=loss,
            optimizer_constructor=optimizer_constructor,
            force_retrain=force_retrain,
            use_checkpoint=use_checkpoint,
        )
        self.model_constructor = model_constructor
        self.train_dl = train_dl

    def construct_model(self):
        return self.model_constructor()
