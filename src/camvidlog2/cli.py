from camvidlog2.bioclip.cli import app as bioclip_app

import typer

app = typer.Typer()
app.add_typer(bioclip_app, name="bioclip")

if __name__ == "__main__":
    app()
