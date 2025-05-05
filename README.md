# Hidden Conflicts in Neural Networks
This is a code repository for the paper "Hidden Conflicts in Neural Networks and their Implications for Explainability".

## Usage Instructions
To replicate our experiments, please set up a new [Conda](https://docs.conda.io/en/latest/) environment using the provided `environment.yml` file and install the `txai` package from the `cafe` folder using either conda or pip (e.g., by running `pip install -e .` in the given folder). You will also need to install an editable version of Captum from the source (using `pip install -e .` in the cloned [Captum repo](https://github.com/pytorch/captum) after checking out the `v0.7.0` version tag) and make the following changes to the code:

* In the file `captum/attr/_core/lrp.py`, add `nn.Flatten: EpsilonRule` to `SUPPORTED_LAYERS_WITH_RULES` and add `nn.GELU` to `SUPPORTED_NON_LINEAR_LAYERS`. This enables the application of LRP to some of our MLP models.
* In the file `captum/metrics/_core/sensitivity.py`, remove `with torch.no_grad():` at line 309, as it interferes with the computation of CAFE scores when evaluating sensitivity.

Taking the above steps should provide you with the basic environment necessary for running the main scripts in the `cafe/experiments` subfolder (though there are some additional requirements for running certain experiments — see the details below):

* Running the experiments on the OpenXAI models requires an additional installation of the [OpenXAI](https://github.com/AI4LIFE-GROUP/OpenXAI) package in the environment.
* To run the MIMIC-IV experiments, you will first need to regenerate our version of the MIMIC-IV data using the provided `MIMIC.ipynb` notebook and the specific `mimic.yml` environment after placing the MIMIC-IV files downloaded from [PhysioNet](https://physionet.org/content/mimiciv/2.2/) at the locations expected by the code. Unfortunately, we are unable to directly share our version of the MIMIC-IV data due to the restrictions associated with the Physionet Credentialed Health Data License 1.5.0 under which the data is distributed.
* Running the Covertype experiments requires the associated data. You should not need to download it manually, as it can be automatically downloaded by the scikit-learn library.
