import typer

from camvidlog2.bioclip.cli import app as bioclip_app

app = typer.Typer()
app.add_typer(bioclip_app, name="bioclip")

if __name__ == "__main__":
    app()
