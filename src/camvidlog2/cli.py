import typer

from camvidlog2.bioclip.cli import app as bioclip_app
from camvidlog2.yoloe.cli import app as yoloe_app

app = typer.Typer()
app.add_typer(bioclip_app, name="bioclip")
app.add_typer(yoloe_app, name="yoloe")

if __name__ == "__main__":
    app()
