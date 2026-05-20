#!/usr/bin/env python3
"""Publish a notebook markdown note as a Jekyll post.

Copies the source file to _posts/<date>-<slug>.md, prepends frontmatter,
copies any referenced images from the source's relative ../assets/ paths
into mh-amani.github.io/assets/images/<slug>/, and rewrites the paths.

Usage:
    scripts/publish-post.py <path/to/note.md> [options]

Options:
    --slug NAME          Override slug (default: derived from filename)
    --date YYYY-MM-DD    Override post date (default: today)
    --unlisted           Mark as hidden (skipped from listing, sitemap, keywords)
    --title TEXT         Skip the title prompt
    --subtitle TEXT      Skip the subtitle prompt (pass "" for no subtitle)
    --keywords TEXT      Skip the keywords prompt (comma-separated)
"""

import argparse
import datetime as dt
import re
import shutil
import sys
from pathlib import Path

PREFIX_RE = re.compile(r"^(LITREV|NOTE|IDEA|DEV|REVIEW)-", re.IGNORECASE)
ASSET_PATH_RE = re.compile(r"(?:\.\./)+assets/[A-Za-z0-9._\-]+")
H1_RE = re.compile(r"^# .*$", re.MULTILINE)


def derive_slug(filename: str) -> str:
    stem = PREFIX_RE.sub("", Path(filename).stem)
    return stem.lower()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("--slug", default=None)
    p.add_argument("--date", default=dt.date.today().isoformat())
    p.add_argument("--unlisted", action="store_true")
    p.add_argument("--title", default=None)
    p.add_argument("--subtitle", default=None)
    p.add_argument("--keywords", default=None)
    args = p.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        sys.exit(f"no such file: {source}")

    slug = args.slug or derive_slug(source.name)
    site_root = Path(__file__).resolve().parent.parent
    img_dest = site_root / "assets" / "images" / slug
    post_path = site_root / "_posts" / f"{args.date}-{slug}.md"

    if post_path.exists():
        if input(f"Overwrite {post_path}? [y/N] ").strip().lower() != "y":
            sys.exit("aborted")

    title = args.title if args.title is not None else input("Title: ").strip()
    subtitle = args.subtitle if args.subtitle is not None else input("Subtitle (blank for none): ").strip()
    keywords = args.keywords if args.keywords is not None else input("Keywords (comma-separated): ").strip()

    body = source.read_text(encoding="utf-8")
    body = H1_RE.sub("", body, count=1).lstrip("\n")

    refs = sorted(set(ASSET_PATH_RE.findall(body)))
    if refs:
        img_dest.mkdir(parents=True, exist_ok=True)
    for ref in refs:
        src_img = (source.parent / ref).resolve()
        if not src_img.is_file():
            print(f"  WARN missing image: {ref} -> {src_img}", file=sys.stderr)
            continue
        shutil.copy2(src_img, img_dest / src_img.name)
        print(f"  copied: {src_img.name}")

    body = ASSET_PATH_RE.sub(
        lambda m: f"/assets/images/{slug}/{Path(m.group(0)).name}",
        body,
    )

    fm = ["---", f"title: {title}"]
    if subtitle:
        fm.append(f"subtitle: {subtitle}")
    fm += ["layout: default", f"date: {args.date}", f"keywords: {keywords}"]
    if args.unlisted:
        fm += ["unlisted: true", "sitemap: false"]
    fm += ["published: true", "---", ""]
    frontmatter = "\n".join(fm) + "\n"

    post_path.parent.mkdir(parents=True, exist_ok=True)
    post_path.write_text(frontmatter + body, encoding="utf-8")

    yyyy, mm, dd = args.date.split("-")
    print(f"\nPublished: {post_path.relative_to(site_root)}")
    print(f"URL after deploy: /blog/{yyyy}/{mm}/{dd}/{slug}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
