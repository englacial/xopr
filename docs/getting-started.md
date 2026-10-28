# Getting started

`xOPR` is a Python package with relatively minimal depdencies. It is designed to be integrated with the `xarray` ecosystem, so, if you're familiar with `xarray`, we hope you'll feel at home.

If you've got a Python environment setup already, you can install `xOPR` using your choice of `pip` or `uv` (recommended). (If you haven't come across `uv` yet, you can [learn more here](https://docs.astral.sh/uv/getting-started/).)

:::{code}
pip install xopr
:::

:::{code}
uv add xopr
:::

Our documentation notebooks use a few extra packages. If you want to be able to run all of our notebooks, you can install `xOPR` with our `docs` extras:

:::{code}
pip install xopr[docs] # or xopr[all] for absolutely everything
:::

:::{code}
uv add xopr[docs] # or xopr[all] for absolutely everything
:::

You do **not** need to be working on the CReSIS servers to use `xOPR`. Unlike the rest of the OPR tooling, `xOPR` relies only on public data access, so you can run it wherever you like.

## Data access credentials

Due to an increase in bot activity, the OPR data site (`https://data.openpolarradar.org/`) currently requires HTTP basic authentication. OPR publishes shared credentials that can be found at [on GitLab](https://gitlab.com/openpolarradar/opr/-/wikis/Downloading-OPR-data-with-Authentication). `OPRConnection` looks for credentials in this order:

1. The `OPR_USERNAME` and `OPR_PASSWORD` environment variables
2. A `machine data.openpolarradar.org` entry in `~/.netrc` (or the file named by `$NETRC`)
3. An interactive prompt (skipped in non-interactive sessions)

You can also pass credentials directly with `xopr.OPRConnection(auth=("username", "password"))`, or pick one method with `auth="environment"`, `auth="netrc"` or `auth="interactive"`. If no credentials are found, `xOPR` warns and continues. Reads from the data site then raise a `PermissionError` that explains how to provide credentials.

## Working on a StratusGeo/CryoCloud/any JupyterHub

Working on StratusGeo (formerly known as [CryoCloud](https://cryointhecloud.com/)) is a great option is you're an existing CryoCloud user.

If you're working in any JupyterHub environment, you can install `xOPR` by running this in a cell at the top of your notebook:

:::{code}
%pip install xopr # or xopr[docs]
:::

## Quick-start with Binder

If you just want to play around, you can launch a temporary JupyterHub instance to experiment. This will not save any of your work and is not a good way to do real work, but it's a quick way to get started. To do this, just press the "Launch on Binder" button in the top right of this page.

# What's next?

If you're a "read the instructions first" person, start by reading through the [design notes](design.md) and then start with [our demo notebook](notebooks/demo_notebook.ipynb).

If you like to skip the instructions, click on "Notebooks" on the left and pick an example that looks interesting!