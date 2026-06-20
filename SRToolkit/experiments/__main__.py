"""
CLI entry point for the SRToolkit runner.

Four subcommands:

    run_job   -- Execute one experiment identified by (grid, dataset, approach, seed).
    adapt     -- Pre-adapt "once"-scope approaches: all missing pairs, or a single
                 (dataset, approach) pair when both --dataset and --approach are given.
                 Add --force to re-adapt and overwrite an existing state (single pair only).
    commands  -- Write a commands file of run_job calls for HPC/terminal use.
    progress  -- Print a dataset × approach table of completed experiments.

Usage examples::

    python -m SRToolkit.experiments run_job \\
        --grid /out/run1/grid.json --dataset NG-1 --approach ProGED --seed 42

    python -m SRToolkit.experiments adapt --grid /out/run1/grid.json            # all pairs
    python -m SRToolkit.experiments adapt \\
        --grid /out/run1/grid.json --dataset NG-1 --approach ProGED             # one pair
    python -m SRToolkit.experiments adapt \\
        --grid /out/run1/grid.json --dataset NG-1 --approach ProGED --force     # re-adapt one pair

    python -m SRToolkit.experiments commands \\
        --grid /out/run1/grid.json \\
        --out /out/run1/commands.txt \\
        --python python3

    python -m SRToolkit.experiments progress --grid /out/run1/grid.json
"""

import argparse

from .experiment_grid import ExperimentGrid


def _cmd_run_job(args: argparse.Namespace) -> None:
    grid = ExperimentGrid.load(args.grid)
    job = grid.build_job(args.dataset, args.approach, args.seed)
    job.run()
    print(f"[run_job] Saved result to {job.result_path}")


def _cmd_adapt(args: argparse.Namespace) -> None:
    if (args.dataset is None) != (args.approach is None):
        raise SystemExit("[adapt] --dataset and --approach must be given together (or neither).")
    if args.force and args.dataset is None:
        raise SystemExit("[adapt] --force requires --dataset and --approach (it only applies to a single pair).")
    grid = ExperimentGrid.load(args.grid)
    if args.dataset is not None:
        grid.adapt_one(args.approach, args.dataset, force=args.force)
        print(f"[adapt] Finished adapting {args.approach!r} on {args.dataset!r}.")
    else:
        grid.adapt_if_missing()
        print("[adapt] Finished adapting all approaches.")


def _cmd_commands(args: argparse.Namespace) -> None:
    grid = ExperimentGrid.load(args.grid)
    skip = not args.all
    prepare_path = grid.save_commands(
        path=args.out,
        python_executable=args.python,
        skip_completed=skip,
    )
    if prepare_path is not None:
        print(f"[commands] Prepare commands written to {prepare_path} — run them to completion FIRST.")
    print(f"[commands] Experiment commands written to {args.out}")


def _cmd_progress(args: argparse.Namespace) -> None:
    ExperimentGrid.load(args.grid).progress()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m SRToolkit.experiments",
        description="SRToolkit experiment runner CLI.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # ---- run_job ----
    p_run = subparsers.add_parser(
        "run_job",
        help="Execute a single experiment identified by (grid, dataset, approach, seed).",
    )
    p_run.add_argument("--grid", required=True, metavar="PATH", help="Path to the grid.json file.")
    p_run.add_argument("--dataset", required=True, metavar="NAME", help="Dataset name within the grid.")
    p_run.add_argument("--approach", required=True, metavar="NAME", help="Approach name within the grid.")
    p_run.add_argument("--seed", required=True, type=int, metavar="N", help="Random seed for this run.")

    # ---- adapt ----
    p_adapt = subparsers.add_parser(
        "adapt",
        help='Pre-adapt "once"-scope approaches (all missing pairs, or one --dataset/--approach pair).',
    )
    p_adapt.add_argument("--grid", required=True, metavar="PATH", help="Path to the grid.json file.")
    p_adapt.add_argument("--dataset", metavar="NAME", help="Adapt only this dataset (requires --approach).")
    p_adapt.add_argument("--approach", metavar="NAME", help="Adapt only this approach (requires --dataset).")
    p_adapt.add_argument(
        "--force",
        action="store_true",
        help="Re-adapt and overwrite even if the state file exists (single --dataset/--approach pair only).",
    )

    # ---- commands ----
    p_cmds = subparsers.add_parser(
        "commands",
        help="Write a commands file of run_job calls for HPC/terminal use.",
    )
    p_cmds.add_argument("--grid", required=True, metavar="PATH", help="Path to the grid.json file.")
    p_cmds.add_argument("--out", required=True, metavar="PATH", help="Output path for the commands file.")
    p_cmds.add_argument(
        "--python",
        default="python",
        metavar="EXECUTABLE",
        help='Python executable to use in commands (default: "python").',
    )
    p_cmds.add_argument("--all", action="store_true", help="Include already-completed jobs (default: skip them).")

    # ---- progress ----
    p_prog = subparsers.add_parser(
        "progress",
        help="Print a dataset × approach table of completed experiments.",
    )
    p_prog.add_argument("--grid", required=True, metavar="PATH", help="Path to the grid.json file.")

    args = parser.parse_args()

    if args.subcommand == "run_job":
        _cmd_run_job(args)
    elif args.subcommand == "adapt":
        _cmd_adapt(args)
    elif args.subcommand == "commands":
        _cmd_commands(args)
    elif args.subcommand == "progress":
        _cmd_progress(args)


if __name__ == "__main__":
    main()
