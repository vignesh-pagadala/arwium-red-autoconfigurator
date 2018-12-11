# Arwium

Arwium - a neural-network-based RED autoconfigurator.

---

## About

Arwium was a project done as a part of the Advanced Networking course at Colorado State University, during the Fall of 2018. The primary use-case of Arwium is to automatically find the best parametric configuration for network gateways that use the Random Early Detection algorithm for congestion avoidance. Automatic configuration of RED parameters will help taking the burden off network administrators, who might otherwise have to configure it manually. Arwium uses reinforcement learning based on observed network behavior to find optimum RED parameters, for a given network state.

The full report can be found <a href = "https://github.com/vignesh-pagadala/arwium-red-autoconfigurator/blob/main/RED-Autoconfigurator-Paper-Final.pdf">here</a>.

Arwium finds best values for the following RED parameters:

1. p(a, b, c, ...): Random drop probability of a packet.
2. maxP: Maximum drop probability.
3. Qmax: Maximum queue length threshold.
4. Qmin: Minimum queue length threshold.
5. Weight associated with the queue (for calculating average queue length).

For calculating the queue length, we use Exponential Weighted Moving Average (EWMA), for reasons stated in report. Q-Reinforcement Learning is used for the training process. For simulating the network environment, we use a discreet-events simulation package by <a href="https://www.grotto-networking.com/DiscreteEventPython.html">Grotto Networking</a>. More information can be found in the <a href = "https://github.com/vignesh-pagadala/arwium-red-autoconfigurator/blob/main/RED-Autoconfigurator-Paper-Final.pdf">report</a>.

### Built With

1. Python 3
2. Discreet-events simulation package by Grotto Networking: https://www.grotto-networking.com/DiscreteEventPython.html


## Getting Started

### Prerequisites

1. Anaconda Distribution: https://www.anaconda.com/products/individual

### Installation

To install, clone the repo:

`git clone https://github.com/vignesh-pagadala/arwium-red-autoconfigurator.git`

## Usage

To reproduce, compile the Python files in the /src directory:

`cd /src`

`python simulation-complete.py`

## Roadmap

See the [open issues](https://github.com/vignesh-pagadala/arwium-red-autoconfigurator/issues) for a list of proposed features (and known issues).

- [Top Feature Requests](https://github.com/vignesh-pagadala/arwium-red-autoconfigurator/issues?q=label%3Aenhancement+is%3Aopen+sort%3Areactions-%2B1-desc) (Add your votes using the 👍 reaction)
- [Top Bugs](https://github.com/vignesh-pagadala/arwium-red-autoconfigurator/issues?q=is%3Aissue+is%3Aopen+label%3Abug+sort%3Areactions-%2B1-desc) (Add your votes using the 👍 reaction)
- [Newest Bugs](https://github.com/vignesh-pagadala/arwium-red-autoconfigurator/issues?q=is%3Aopen+is%3Aissue+label%3Abug)

## Support

Reach out to the maintainer at one of the following places:

- [GitHub issues](https://github.com/vignesh-pagadala/arwium-red-autoconfigurator/issues/new?assignees=&labels=question&template=04_SUPPORT_QUESTION.md&title=support%3A+)
- The email which is located [in GitHub profile](https://github.com/vignesh-pagadala)

## Project assistance

If you want to say **thank you** or/and support active development of Arwium RED Autoconfigurator:

- Add a [GitHub Star](https://github.com/vignesh-pagadala/arwium-red-autoconfigurator) to the project.
- Tweet about the Arwium RED Autoconfigurator on your Twitter.
- Write interesting articles about the project on [Dev.to](https://dev.to/), [Medium](https://medium.com/) or personal blog.

Together, we can make Arwium RED Autoconfigurator **better**!

## Contributing

First off, thanks for taking the time to contribute! Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make will benefit everybody else and are **greatly appreciated**.

We have set up a separate document containing our [contribution guidelines](docs/CONTRIBUTING.md).

Thank you for being involved!

## Authors & contributors

The original setup of this repository is by [Vignesh Pagadala](https://github.com/vignesh-pagadala).

For a full list of all authors and contributors, check [the contributor's page](https://github.com/vignesh-pagadala/arwium-red-autoconfigurator/contributors).

## Security

Arwium RED Autoconfigurator follows good practices of security, but 100% security can't be granted in software.
Arwium RED Autoconfigurator is provided **"as is"** without any **warranty**. Use at your own risk.

_For more info, please refer to the [security](docs/SECURITY.md)._

## License

This project is licensed under the **MIT license**.

See [LICENSE](LICENSE) for more information.