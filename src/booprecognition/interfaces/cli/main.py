"""Entrega por linha de comando."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from booprecognition.config.container import build_container
from booprecognition.domain.errors import DomainError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="booprecognition")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("identify", help="Identificar pessoa").add_argument("image", type=Path)
    sub.add_parser("register", help="Cadastrar pessoa").add_argument("image", type=Path)
    sub.add_parser("remove", help="Remover pessoa").add_argument("person_id", type=int)
    serve = sub.add_parser("serve", help="Subir API HTTP")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        from booprecognition.interfaces.http.fastapi_app import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)
        return 0

    container = build_container()
    try:
        if args.command == "identify":
            result = container.identify_person.execute(args.image.read_bytes())
            output = result.to_dict() if result else None
        elif args.command == "register":
            output = container.register_person.execute(args.image.read_bytes()).to_dict()
        else:
            output = container.remove_person.execute(args.person_id).to_dict()
    except DomainError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
