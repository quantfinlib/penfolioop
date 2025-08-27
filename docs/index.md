# PenFolioOp


[🌐 **GitHub**](https://github.com/quantfinlib/penfolioop)
&nbsp;&nbsp;&nbsp; [🔗 **API**](penfolioop)
&nbsp;&nbsp;&nbsp; [📖 **Docs**](https://quantfinlib.github.io/penfolioop/)


## Getting Started

* [Basic example with US Asset Classes](Example_US_Asset_Classes.html)


## Documentation

The documentation is available at [githubpages](https://quantfinlib.github.io/penfolioop/).
The [🔗 API documentation](penfolioop) is generated using [pdoc3](https://pdoc3.github.io/pdoc/).

To manually generate the documentation, first, install the penfolioop package with the doc dependencies using `uv`:

```bash
$ uv pip install -e .[docs]
```

Then
```bash
$ uv run pdoc --html  -c latex_math=True --output-dir docs --force penfolioop
```