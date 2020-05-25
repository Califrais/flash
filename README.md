# lights
_ligths_ is a generalized joint model for high-dimensional multivariate longitudinal data and censored durations

## Quick description
_ligths_ is a Python 3 package to deal with the problem of joint modeling of longitudinal data and censored durations, where a large number of longitudinal features are available. Features extracted from the longitudinal processes are included as potential risk factor in the survival model, which is a group-specific Cox model with high- dimensional shared associations.

To allow flexibility in modeling the dependency between the longitudinal features and the event time, we use appropriate penalties : elastic net for feature selection in the latent class membership, and sparse group lasso in the survival model, as well as for the fixed effect (allowing flexible representations of time).

Inference is achieved using an efficient and novel Quasi-Newton Monte Carlo Expectation Maximization algorithm.

## Use cases

_lights_ can be used for many statistical learning applications where one wants to predict the risk for an event of interest to occur quickly, taking into account simultaneously a huge number of longitudinal signals in a high-dimensional context. It provides powerful interpretability by automatically pinpointing significant time-dependent and time-independent features. Hence, it can be used for real-time decision support, for instance:
 
- In a medical context, where the event of interest could be survival time, re-hospitalization, relapse or disease progression, and the longitudinal data represents biomarkers or vital parameters measurements.

- In a customer's satisfaction monitoring context, where the event of interest could be the time when a client stops using a company's product or service (churn), and the longitudinal data represents the client's activity recorded from account opening throughout the duration of the business relationship.

## Installation
Clone the repository, then inside the folder, use a `virtualenv` to install the requirements
```shell script
git clone git@github.com:Califrais/lights.git
cd lights

# If your default interpreter is Python3:
virtualenv .env
# If your default interpreter is Python2, you can explicitly target Python3 with:
virtualenv -p python3 .env

source .env/bin/activate
```
Then, to download all required modules and initialize the project run the following commands:
```shell script
pip install -r requirements.txt
```