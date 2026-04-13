"""
CLI entry point for the SRToolkit runner.

Three subcommands:

    run_job   -- Execute a single experiment given three JSON file paths.
    adapt     -- Pre-adapt all "once"-scope approaches where state is missing.
    commands  -- Write a commands file of run_job calls for HPC/terminal use.

Usage examples::

    python -m SRToolkit.experiments run_job \\
        --dataset /out/run1/_datasets/NG-1/NG-1.json \\
        --approach /out/run1/_approaches/ProGED_config.json \\
        --info /out/run1/NG-1/ProGED/exp_42/info.json

    python -m SRToolkit.experiments adapt --grid /out/run1/grid.json

    python -m SRToolkit.experiments commands \\
        --grid /out/run1/grid.json \\
        --out /out/run1/commands.txt \\
        --python python3
"""

import argparse

from .experiment_grid import ExperimentGrid, ExperimentJob


def _cmd_run_job(args: argparse.Namespace) -> None:
    job = ExperimentJob(args.dataset, args.approach, args.info)
    job.run()
    print(f"[run_job] Saved result to {job.result_path}")


def _cmd_adapt(args: argparse.Namespace) -> None:
    grid = ExperimentGrid.load(args.grid)
    grid.adapt_if_missing()
    print("[adapt] Finished adapting all approaches.")


def _cmd_commands(args: argparse.Namespace) -> None:
    grid = ExperimentGrid.load(args.grid)
    skip = not args.all
    grid.save_commands(
        path=args.out,
        python_executable=args.python,
        skip_completed=skip,
    )
    print(f"[commands] Commands file written to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m SRToolkit.experiments",
        description="SRToolkit experiment runner CLI.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # ---- run_job ----
    p_run = subparsers.add_parser(
        "run_job",
        help="Execute a single experiment from three JSON file paths.",
    )
    p_run.add_argument("--dataset", required=True, metavar="PATH", help="Path to SR_dataset.to_dict() JSON file.")
    p_run.add_argument("--approach", required=True, metavar="PATH", help="Path to ApproachConfig.to_dict() JSON file.")
    p_run.add_argument("--info", required=True, metavar="PATH", help="Path to ExperimentInfo.to_dict() JSON file.")

    # ---- adapt ----
    p_adapt = subparsers.add_parser(
        "adapt",
        help='Pre-adapt all "once"-scope approaches where state files are missing.',
    )
    p_adapt.add_argument("--grid", required=True, metavar="PATH", help="Path to the grid.json file.")

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

    args = parser.parse_args()

    if args.subcommand == "run_job":
        _cmd_run_job(args)
    elif args.subcommand == "adapt":
        _cmd_adapt(args)
    elif args.subcommand == "commands":
        _cmd_commands(args)


if __name__ == "__main__":
    main()
