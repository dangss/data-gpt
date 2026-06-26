from pybloqs.block.image import PlotlyPlotBlock
from plotly.graph_objs import Figure as PlotlyFigure
from pybloqs.static import JScript
import plotly.offline as po


def __custom_init__(self, contents, plotly_kwargs=None, **kwargs):
    """
    Writes out the content as raw text or HTML.

    :param contents: Plotly graphics object figure.
    :param plotly_kwargs: Kwargs that are passed to plotly plot function.
    :param kwargs: Optional styling arguments. The `style` keyword argument has special
                    meaning in that it allows styling to be grouped as one argument.
                    It is also useful in case a styling parameter name clashes with a standard
                    block parameter.
    """
    self.resource_deps = [JScript(script_string=po.offline.get_plotlyjs(), name='plotly')]

    super(PlotlyPlotBlock, self).__init__(**kwargs)

    if not isinstance(contents, PlotlyFigure):
        raise ValueError("Expected plotly.graph_objs.graph_objs.Figure type but got %s", type(contents))

    plotly_kwargs = plotly_kwargs or {}
    
    # prefix = "<script>if (typeof require !== 'undefined' && Plotly) {var Plotly = require('plotly')}</script>"
    prefix = "<script type='text/javascript'> require.config({ paths: { 'plotly': 'https://cdn.plot.ly/plotly-latest.min'}, waitSeconds: 40}); </script>"

    data_html = po.plot(contents, include_plotlyjs=False, output_type='div', **plotly_kwargs)
    data_html = prefix + data_html.replace("window.PLOTLYENV=window.PLOTLYENV", "require(['plotly'], function (Plotly){ window.PLOTLYENV = window.PLOTLYENV").replace("</script>", "})</script>")

    self._contents = data_html


PlotlyPlotBlock.__init__ = __custom_init__

# fix jupyterlab not showing chart
from IPython.display import display, HTML
js = '<script defer="defer" src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>'
display(HTML(js))