from __future__ import annotations

import argparse
from pathlib import Path

from . import db
from .config import DEFAULT_CAPTION_MODEL, DEFAULT_CLIP_MODEL, DEFAULT_DB_PATH, Settings
from .embedding import ModelBundle
from .indexer import index_directory
from .search import search_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="picfind", description="Local image semantic search")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to the SQLite database")
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL, help="CLIP model id")
    parser.add_argument("--caption-model", default=DEFAULT_CAPTION_MODEL, help="BLIP caption model id")
    parser.add_argument("--no-caption", action="store_true", help="Disable caption generation during indexing")

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize the database")
    init_parser.set_defaults(handler=handle_init)

    index_parser = subparsers.add_parser("index", help="Index a source directory")
    index_parser.add_argument("--source", type=Path, required=True, help="Root directory containing images")
    index_parser.set_defaults(handler=handle_index)

    search_parser = subparsers.add_parser("search", help="Search indexed images")
    search_parser.add_argument("--query", required=True, help="Natural-language search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    search_parser.set_defaults(handler=handle_search)

    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(handler=handle_stats)

    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    return Settings(
        db_path=args.db,
        clip_model=args.clip_model,
        caption_model=args.caption_model,
        enable_captioning=not getattr(args, "no_caption", False),
    )


def handle_init(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    settings.ensure_parent_dirs()
    with db.connect(settings.db_path):
        pass
    print(f"Database ready: {settings.db_path}")
    return 0


def handle_index(args: argparse.Namespace) -> int:
    if not args.source.exists() or not args.source.is_dir():
        raise SystemExit(f"Source directory not found: {args.source}")

    settings = settings_from_args(args)
    models = ModelBundle(settings.clip_model, settings.caption_model)
    print(f"Device: {models.device_summary()}")
    print(f"Captioning: {'enabled' if settings.enable_captioning else 'disabled'}")

    summary = index_directory(args.source, settings)
    print(f"Total discovered: {summary['total']}")
    print(f"Indexed: {summary['indexed']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Failed: {summary['failed']}")
    print(f"Flush size: {summary['flush_every']}")
    if summary['interrupted']:
        print("Interrupted: yes (pending batch saved)")
        return 130
    return 0


def handle_search(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    results = search_images(args.query, settings, limit=args.limit)
    if not results:
        print("No indexed images found.")
        return 0
    for index, result in enumerate(results, start=1):
        print(f"{index}. score={result.score:.4f}")
        print(f"   path: {result.path}")
        print(f"   caption: {result.caption}")
    return 0


def handle_stats(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    settings.ensure_parent_dirs()
    with db.connect(settings.db_path) as connection:
        count = db.count_images(connection)
    print(f"Indexed images: {count}")
    print(f"Database: {settings.db_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
