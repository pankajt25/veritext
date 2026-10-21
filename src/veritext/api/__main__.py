"""CLI interface for Veritext REST API service and key management."""

import argparse
import sys

import uvicorn

from veritext.api.auth import create_api_key, load_api_keys, revoke_api_key


def main():
    parser = argparse.ArgumentParser(
        prog="python -m veritext.api",
        description="Veritext REST API Service & Key Management",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: create-key
    create_parser = subparsers.add_parser("create-key", help="Generate a new API key")
    create_parser.add_argument(
        "--name",
        type=str,
        default="default",
        help="Identifier name for the key (e.g., 'moodle_integration', 'instructor_bob')",
    )

    # Command: list-keys
    subparsers.add_parser("list-keys", help="List registered API keys")

    # Command: revoke-key
    revoke_parser = subparsers.add_parser("revoke-key", help="Revoke an existing API key")
    revoke_parser.add_argument("name", type=str, help="Name or prefix of the key to revoke")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Run local REST API server")
    serve_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Binding host address (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8600,
        help="Port number to bind (default: 8600)",
    )

    args = parser.parse_args()

    if args.command == "create-key":
        key, khash = create_api_key(args.name)
        print("\n" + "=" * 60)
        print("🎉 Veritext API Key Created Successfully")
        print("=" * 60)
        print(f"Key Name : {args.name}")
        print(f"API Key  : {key}")
        print("-" * 60)
        print("⚠️  Store this key securely! It will NOT be shown again in plaintext.")
        print("Include in HTTP headers as:  X-API-Key: " + key)
        print("=" * 60 + "\n")

    elif args.command == "list-keys":
        data = load_api_keys()
        keys_map = data.get("keys", {})
        if not keys_map:
            print("No API keys found. Generate one with: python -m veritext.api create-key --name <name>")
            return
        print(f"\nFound {len(keys_map)} registered API key(s):")
        print(f"{'Name':<20} {'Prefix':<16} {'Created':<25} {'Active'}")
        print("-" * 70)
        for khash, meta in keys_map.items():
            print(f"{meta.get('name', 'unnamed'):<20} {meta.get('prefix', '???'):<16} {meta.get('created_at', ''):<25} {meta.get('active', True)}")
        print()

    elif args.command == "revoke-key":
        found = revoke_api_key(args.name)
        if found:
            print(f"✅ Key '{args.name}' successfully revoked.")
        else:
            print(f"❌ Key '{args.name}' not found.")

    elif args.command == "serve":
        print(f"Starting Veritext API on http://{args.host}:{args.port}")
        uvicorn.run("veritext.api.app:app", host=args.host, port=args.port, reload=False)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
