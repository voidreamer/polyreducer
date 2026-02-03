"""CLI for poly-reducer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress
    HAS_CLI = True
except ImportError:
    HAS_CLI = False

if HAS_CLI:
    from poly_reducer import PolyReducer, LODChain

    app = typer.Typer(
        name="polyreduce",
        help="Smart headless polygon reduction",
        add_completion=False,
    )
    console = Console()
    
    @app.command()
    def reduce(
        source: Path = typer.Argument(..., help="Source mesh file"),
        target_tris: Optional[int] = typer.Option(None, "--target-tris", "-t", help="Target triangle count"),
        ratio: Optional[float] = typer.Option(None, "--ratio", "-r", help="Reduction ratio (0-1)"),
        output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
        preserve_uvs: bool = typer.Option(True, "--preserve-uvs/--no-preserve-uvs"),
        preserve_normals: bool = typer.Option(True, "--preserve-normals/--no-preserve-normals"),
        analyze_only: bool = typer.Option(False, "--analyze", "-a", help="Analyze only, don't reduce"),
    ) -> None:
        """Reduce polygon count of a mesh."""
        if not source.exists():
            console.print(f"[red]Error:[/red] File not found: {source}")
            raise typer.Exit(1)
        
        try:
            reducer = PolyReducer()
        except RuntimeError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
        
        if analyze_only:
            # Analyze mode
            with console.status("Analyzing mesh..."):
                analysis = reducer.analyze(source)
            
            table = Table(title=f"Analysis: {source.name}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value")
            
            table.add_row("Triangles", f"{analysis.triangle_count:,}")
            table.add_row("Vertices", f"{analysis.vertex_count:,}")
            table.add_row("Edges", f"{analysis.edge_count:,}")
            table.add_row("Has UVs", "✓" if analysis.has_uvs else "✗")
            table.add_row("Sharp Edges", f"{analysis.sharp_edge_count:,}")
            table.add_row("Boundary Edges", f"{analysis.boundary_edge_count:,}")
            table.add_row("Materials", str(analysis.material_count))
            
            console.print(table)
            
            # Suggestions
            if analysis.suggested_lods:
                console.print("\n[bold]Suggested LOD targets:[/bold]")
                for lod in analysis.suggested_lods:
                    console.print(f"  LOD{lod.level}: {lod.target_tris:,} triangles ({lod.ratio:.0%})")
            
            return
        
        # Reduce mode
        if target_tris is None and ratio is None:
            console.print("[red]Error:[/red] Specify --target-tris or --ratio")
            raise typer.Exit(1)
        
        with console.status("Reducing mesh..."):
            result = reducer.reduce(
                source,
                target_tris=target_tris,
                ratio=ratio,
                output=output,
                preserve_uvs=preserve_uvs,
                preserve_normals=preserve_normals,
            )
        
        if result.success:
            console.print(f"[green]✓ Reduced successfully[/green]")
            console.print(f"  Original: {result.original_tris:,} triangles")
            console.print(f"  Final: {result.final_tris:,} triangles ({result.reduction_ratio:.1%})")
            console.print(f"  Output: {result.output_path}")
        else:
            console.print(f"[red]✗ Reduction failed:[/red] {result.error_message}")
            raise typer.Exit(1)
    
    @app.command("lod-chain")
    def lod_chain(
        source: Path = typer.Argument(..., help="Source mesh file"),
        targets: str = typer.Option(
            None, "--targets", "-t",
            help="Comma-separated target triangle counts (e.g., 100000,50000,10000)",
        ),
        output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    ) -> None:
        """Generate LOD chain from a mesh."""
        if not source.exists():
            console.print(f"[red]Error:[/red] File not found: {source}")
            raise typer.Exit(1)
        
        # Parse targets
        target_list = None
        if targets:
            target_list = [int(t.strip()) for t in targets.split(",")]
        
        chain = LODChain(
            source=source,
            targets=target_list,
            output_dir=output_dir,
        )
        
        with Progress() as progress:
            task = progress.add_task("Generating LODs...", total=None)
            
            def on_progress(current, total):
                progress.update(task, total=total, completed=current)
            
            results = chain.generate(on_progress=on_progress)
        
        # Display results
        table = Table(title="LOD Chain Results")
        table.add_column("Level", style="cyan")
        table.add_column("Triangles")
        table.add_column("Ratio")
        table.add_column("Output")
        table.add_column("Status")
        
        for lod in results:
            status = "[green]✓[/green]" if lod.success else f"[red]✗[/red] {lod.error_message}"
            table.add_row(
                f"LOD{lod.level}",
                f"{lod.triangle_count:,}",
                f"{lod.reduction_ratio:.1%}",
                str(lod.output_path.name) if lod.success else "-",
                status,
            )
        
        console.print(table)

    @app.command("serve")
    def serve(
        host: str = typer.Option("0.0.0.0", "--host", "-h"),
        port: int = typer.Option(8000, "--port", "-p"),
    ) -> None:
        """Start the REST API server."""
        try:
            from poly_reducer.api import run_server
            console.print(f"[green]Starting API server at http://{host}:{port}[/green]")
            run_server(host=host, port=port)
        except ImportError:
            console.print("[red]API dependencies not installed. Run: pip install poly-reducer[api][/red]")
            raise typer.Exit(1)
    
    @app.command("worker")
    def worker(
        queue_url: str = typer.Option(..., "--queue", "-q", help="Redis URL or SQS queue"),
    ) -> None:
        """Start a background worker for processing jobs."""
        try:
            from poly_reducer.worker import run_worker
            console.print(f"[green]Starting worker, listening to {queue_url}[/green]")
            run_worker(queue_url)
        except ImportError:
            console.print("[red]Worker dependencies not installed. Run: pip install poly-reducer[worker][/red]")
            raise typer.Exit(1)

else:
    def app():
        print("CLI dependencies not installed. Run: pip install poly-reducer[cli]")


def main() -> None:
    if HAS_CLI:
        app()
    else:
        print("CLI dependencies not installed. Run: pip install poly-reducer[cli]")


if __name__ == "__main__":
    main()
