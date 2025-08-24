import typer

from camvidlog2.bioclip.cli import app as bioclip_app
from camvidlog2.moondream.cli import app as moondream_app
from camvidlog2.yoloe.cli import app as yoloe_app

app = typer.Typer()
app.add_typer(bioclip_app, name="bioclip")
app.add_typer(yoloe_app, name="yoloe")
app.add_typer(moondream_app, name="moondream")

if __name__ == "__main__":
    app()
