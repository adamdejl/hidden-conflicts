import numpy as np
import torch

from captum.metrics import infidelity_perturb_func_decorator


def infidelity_perturb_tabular_const(
    scale,
    categorical_variables_spans=None,
    test_dl=None,
    cat_resample_proba=0.25,
):
    @infidelity_perturb_func_decorator(True)
    def infidelity_perturb_func(inputs):
        # Construct masks for noise and resampling categorical variables
        noise_mask = torch.ones(inputs.shape, device=inputs.device)
        if categorical_variables_spans is not None:
            cat_resample_masks = []
            for i, (s, e) in enumerate(categorical_variables_spans):
                cat_resample_mask = torch.zeros(inputs.shape, device=inputs.device)
                probabilities = torch.full(
                    (inputs.size(0), 1), cat_resample_proba, device=inputs.device
                )
                resample_tensor = torch.bernoulli(probabilities)
                noise_mask[:, s:e] = 0.0
                cat_resample_mask[:, s:e] = resample_tensor
                cat_resample_masks.append(cat_resample_mask)

        # Add noise to continuous features only
        noise = (
            torch.tensor(
                np.random.normal(0, scale, inputs.shape), device=inputs.device
            ).float()
            * noise_mask
        )
        perturbed_inputs = inputs - noise

        # Randomly resample categorical variables
        if categorical_variables_spans is not None:
            if test_dl is None:
                raise ValueError("test_dl must be provided for categorical resampling")
            test_dataset = test_dl.dataset
            for cat_resample_mask in cat_resample_masks:
                random_samples = []
                for _ in range(len(inputs)):
                    random_samples.append(
                        test_dataset[np.random.randint(len(test_dataset))][0]
                    )
                if not isinstance(random_samples[0], torch.Tensor):
                    random_samples = list(map(lambda a: torch.tensor(a).to(inputs.device), random_samples))
                random_samples = torch.stack(random_samples).to(inputs.device)
                perturbed_inputs = (
                    perturbed_inputs * (1 - cat_resample_mask)
                    + random_samples * cat_resample_mask
                )

        return perturbed_inputs

    return infidelity_perturb_func
