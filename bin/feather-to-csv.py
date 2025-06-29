#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(help="Convert feather files to human-readable CSV.")


@app.command()
def convert(
    input_file: Path = typer.Argument(..., help="Input feather file"),
    output: Path = typer.Option(
        None, "-o", "--output", help="Output CSV file (default: stdout)"
    ),
):
    """Convert a feather file to CSV."""
    df = pd.read_feather(input_file)
    if output:
        df.to_csv(output, index=False)
        typer.echo(f"CSV written to {output}")
    else:
        df.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    app()
