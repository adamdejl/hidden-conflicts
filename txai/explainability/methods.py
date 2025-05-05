import captum
import torch
import warnings

from txai.explainability.cafe_ng import CafeNgExplainer


class ExplanationMethod:
    def __init__(self, name, callback, args={}):
        self.name = name
        self.callback = callback
        self.args = args

    def __call__(self, model, x, target):
        return self.callback(model, x, target, **self.args)


def explain_cafe(model, x, target, c=0.5):
    return_tuple = False
    if isinstance(x, tuple):
        x = x[0]
        return_tuple = True
    cafe = CafeNgExplainer(model, c=c)
    ((s_feat_plus, s_feat_minus), _) = cafe.attribute(
        x, ref=torch.zeros_like(x), target=target
    )
    attribution = s_feat_plus - s_feat_minus
    if return_tuple:
        return (attribution,)
    return attribution


def explain_ixg(model, x, target):
    with warnings.catch_warnings():
        # Suppress annoying warnings about setting activation hooks & time consuming operations
        # IMPORTANT: Re-enable warnings when changing code to ensure things still work
        warnings.simplefilter("ignore")
        ixg_captum = captum.attr.InputXGradient(model)
        attribution = ixg_captum.attribute(x, target=target)
    return attribution


def explain_lrp(model, x, target):
    with warnings.catch_warnings():
        # Suppress annoying warnings about setting activation hooks & time consuming operations
        # IMPORTANT: Re-enable warnings when changing code to ensure things still work
        warnings.simplefilter("ignore")
        lrp_captum = captum.attr.LRP(model)
        attribution = lrp_captum.attribute(x, target=target)
    return attribution


def explain_dl(model, x, target, multiply_by_inputs=True):
    with warnings.catch_warnings():
        # Suppress annoying warnings about setting activation hooks & time consuming operations
        # IMPORTANT: Re-enable warnings when changing code to ensure things still work
        warnings.simplefilter("ignore")
        dl_captum = captum.attr.DeepLift(model, multiply_by_inputs=multiply_by_inputs)
        attribution, delta = dl_captum.attribute(
            x,
            baselines=torch.zeros_like(x[0] if isinstance(x, tuple) else x),
            return_convergence_delta=True,
            target=target,
        )
    return attribution


def explain_ig(model, x, target, multiply_by_inputs=True):
    with warnings.catch_warnings():
        # Suppress annoying warnings about setting activation hooks & time consuming operations
        # IMPORTANT: Re-enable warnings when changing code to ensure things still work
        warnings.simplefilter("ignore")
        ig_captum = captum.attr.IntegratedGradients(
            model, multiply_by_inputs=multiply_by_inputs
        )
        attribution, delta = ig_captum.attribute(
            x,
            baselines=torch.zeros_like(x[0] if isinstance(x, tuple) else x),
            return_convergence_delta=True,
            target=target,
            # Reduced internal batch size to speed up the computation
            internal_batch_size=64000,
        )
    return attribution


def explain_sg(model, x, target, multiply_by_inputs=True):
    with warnings.catch_warnings():
        # Suppress annoying warnings about setting activation hooks & time consuming operations
        # IMPORTANT: Re-enable warnings when changing code to ensure things still work
        warnings.simplefilter("ignore")
        return_tuple = False
        if isinstance(x, tuple):
            x = x[0]
            return_tuple = True
        nt_captum = captum.attr.NoiseTunnel(captum.attr.Saliency(model))
        attribution = nt_captum.attribute(x, target=target)
        if multiply_by_inputs:
            attribution *= x
        if return_tuple:
            return (attribution,)
    return attribution


def explain_gs(model, x, target, multiply_by_inputs=True):
    with warnings.catch_warnings():
        # Suppress annoying warnings about setting activation hooks & time consuming operations
        # IMPORTANT: Re-enable warnings when changing code to ensure things still work
        warnings.simplefilter("ignore")
        gs_captum = captum.attr.GradientShap(
            model, multiply_by_inputs=multiply_by_inputs
        )
        attribution = gs_captum.attribute(
            x,
            baselines=torch.zeros_like(x[0] if isinstance(x, tuple) else x),
            target=target,
        )
    return attribution


def explain_ks(model, x, target, multiply_by_inputs=False, feature_mask=None):
    with warnings.catch_warnings():
        # Suppress annoying warnings about setting activation hooks & time consuming operations
        # IMPORTANT: Re-enable warnings when changing code to ensure things still work
        warnings.simplefilter("ignore")
        ks_captum = captum.attr.KernelShap(model)
        attribution = ks_captum.attribute(
            x,
            baselines=torch.zeros_like(x[0] if isinstance(x, tuple) else x),
            feature_mask=feature_mask,
            target=target,
        )
        if multiply_by_inputs:
            attribution *= x
    return attribution


def explain_svs(model, x, target, multiply_by_inputs=False, feature_mask=None):
    with warnings.catch_warnings():
        # Suppress annoying warnings about setting activation hooks & time consuming operations
        # IMPORTANT: Re-enable warnings when changing code to ensure things still work
        warnings.simplefilter("ignore")
        svs_captum = captum.attr.ShapleyValueSampling(model)
        attribution = svs_captum.attribute(
            x,
            baselines=torch.zeros_like(x[0] if isinstance(x, tuple) else x),
            feature_mask=feature_mask,
            target=target,
        )
        if multiply_by_inputs:
            attribution *= x
    return attribution


def explain_lime(model, x, target, multiply_by_inputs=False, feature_mask=None):
    with warnings.catch_warnings():
        # Suppress annoying warnings about setting activation hooks & time consuming operations
        # IMPORTANT: Re-enable warnings when changing code to ensure things still work
        warnings.simplefilter("ignore")
        lime_captum = captum.attr.Lime(model)
        attribution = lime_captum.attribute(
            x,
            baselines=torch.zeros_like(x[0] if isinstance(x, tuple) else x),
            feature_mask=feature_mask,
            target=target,
        )
        if multiply_by_inputs:
            attribution *= x
    return attribution
