#!/usr/bin/env python3
"""tools/mcp_cli.py - Direct CLI bridge for Strata MCP server tools.

Usage:
    python tools/mcp_cli.py status [--hardware]
    python tools/mcp_cli.py start [model_name] [--port 8080] [--wait 60]
    python tools/mcp_cli.py stop [--force]
    python tools/mcp_cli.py logs [--source server|engine|setup] [--lines 50]
    python tools/mcp_cli.py benchmark [--tokens 64]
    python tools/mcp_cli.py models
    python tools/mcp_cli.py connect
"""
import sys
import json
import argparse
from pathlib import Path

# Add current folder to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from strata_mcp import Strata, Tools

def main():
    parser = argparse.ArgumentParser(description="Direct CLI execution for Strata MCP tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # status
    p_status = subparsers.add_parser("status", help="Check server status and installed models")
    p_status.add_argument("--hardware", action="store_true", help="Include full hardware details")
    p_status.add_argument("--port", type=int, default=8080)

    # start
    p_start = subparsers.add_parser("start", help="Start model in background")
    p_start.add_argument("model", nargs="?", default=None, help="Model name (e.g. q2_0 or coder-iq1_m)")
    p_start.add_argument("--port", type=int, default=8080)
    p_start.add_argument("--wait", type=int, default=60, help="Seconds to wait for READY")

    # stop
    p_stop = subparsers.add_parser("stop", help="Stop running model server")
    p_stop.add_argument("--force", action="store_true", help="Force stop even if busy")
    p_stop.add_argument("--port", type=int, default=8080)

    # logs
    p_logs = subparsers.add_parser("logs", help="Get engine/server logs")
    p_logs.add_argument("--source", choices=["server", "engine", "setup"], default="server")
    p_logs.add_argument("--lines", type=int, default=50)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run benchmark on running server")
    p_bench.add_argument("--tokens", type=int, default=64)
    p_bench.add_argument("--port", type=int, default=8080)

    # models
    subparsers.add_parser("models", help="List supported models and hardware fit")

    # connect
    p_conn = subparsers.add_parser("connect", help="Get connection info for agents / clients")
    p_conn.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    strata_instance = Strata(ROOT)
    tools = Tools(strata_instance)

    if args.command == "status":
        res = tools.call("strata_status", {"hardware": args.hardware, "port": args.port})
    elif args.command == "start":
        kwargs = {"port": args.port, "wait_seconds": args.wait}
        if args.model:
            kwargs["model"] = args.model
        res = tools.call("strata_start", kwargs)
    elif args.command == "stop":
        res = tools.call("strata_stop", {"force": args.force, "port": args.port})
    elif args.command == "logs":
        res = tools.call("strata_logs", {"source": args.source, "lines": args.lines})
    elif args.command == "benchmark":
        res = tools.call("strata_benchmark", {"port": args.port, "max_tokens": args.tokens})
    elif args.command == "models":
        res = tools.call("strata_models", {})
    elif args.command == "connect":
        res = tools.call("strata_connect_info", {"port": args.port})
    else:
        print(f"Unknown command: {args.command}")
        return 1

    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
