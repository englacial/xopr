# xOPR

## Overview

xOPR is a Python library designed to make accessing [Open Polar Radar's](https://ops.cresis.ku.edu/) data archives easy, scalable, and reproducible.

:::{tip}
xOPR is a work in progress! We hope to keep the API relatively stable, but it's still early days and it may evolve.

We welcome your feedback and contributions. If you run into problems or have ideas for how this could be better, please consider [opening an issue](https://github.com/englacial/xopr/issues/new/choose). We also welcome pull requests!

If you're using or thinking about using xOPR, please reach out to thomas.teisberg@astera.org. Even just a one sentence email with what you're interested in using xOPR for or what you'd like it to do for you is helpful!
:::

xOPR offers access to most of the OPR data catalog, but not absolutely every line. Check out our [availability maps](https://docs.englacial.org/xopr/map/) for details.

## Installing xOPR

To install xOPR, use:

:::{code}
pip install xopr
:::

Or, using [uv](https://docs.astral.sh/uv/) (our recommendation!):

:::{code}
uv add xopr
:::

## Getting Started

Minimal example of loading and plotting a single frame of radar data:

```python
import numpy as np
import xopr

opr = xopr.OPRConnection()

stac_items = opr.query_frames(collections=["2022_Antarctica_BaslerMKB"], segment_paths=["20221228_01"], max_items=1)
frames = opr.load_frames(stac_items)

(10*np.log10(frames[0].Data)).plot.imshow(x='slow_time', y='twtt', cmap='gray', yincrease=False)
```

To learn more, we recommend looking through the notebooks on the left side navigation.

## Design

For details on the initial design planning of xopr, please see [this OPR wiki page](https://gitlab.com/openpolarradar/opr/-/wikis/OPR-Data-Access-Tool-Planning).

For current design and terminology, see the [design notes](https://docs.englacial.org/xopr/design/) page.

## API Reference

Full API documentation for all xopr modules is available in the [API Reference](api/xopr.html).

:::{figure} img/opr-data-access-infra-1.png
:align: center
:width: 80%

xOPR acts as an interface to OPR data. It has two primary roles: helping create queries to the OPR STAC catalog to find data and returning radar data in the form of an Xarray Dataset.
:::

