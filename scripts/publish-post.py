#!/usr/bin/env python3
"""Publish a notebook markdown note as a Jekyll post.

What this does (publish path, given a source markdown file):

  1. Derive a slug by stripping LITREV-/NOTE-/IDEA-/DEV-/REVIEW- prefixes
     from the filename and lowercasing.
  2. Strip the first body H1 (the `default` layout already renders
     `{{ page.title }}` from front matter, so a body H1 would duplicate it).
  3. Find every `(../)+assets/foo.png` reference in the body, resolve each
     relative to the source file's directory, copy the image into
     `assets/images/<slug>/foo.png`, and rewrite the body path to absolute.
  4. Write `_posts/YYYY-MM-DD-<slug>.md` with a frontmatter block. The
     emitted frontmatter always includes `render_with_liquid: false` —
     notebook notes commonly contain `{{ ... }}` and `{% ... %}` literals
     in code samples and prompt templates, which Jekyll would otherwise
     try to parse as Liquid and fail the build. Disabling Liquid in the
     post body is safe: the `default` layout itself still uses Liquid
     (for `{{ content }}`, mathjax include, bibliography), but the post
     content is treated as literal markdown.
  5. With `--unlisted`, also emit `unlisted: true` and `sitemap: false`,
     which the listing template (`_layouts/blog.html`) and the keyword
     plugin (`_plugins/generate_keywords.rb`) filter out.

The list path (`--list-unlisted`) walks every file in `_posts/` and prints
the public URL for posts marked `unlisted: true`. URLs are derived
mechanically from the filename per the permalink scheme in `_config.yml`
(`/blog/:year/:month/:day/:title/`).

Usage:
    scripts/publish-post.py <path/to/note.md> [options]
    scripts/publish-post.py --list-unlisted

Options:
    --slug NAME          Override slug (default: derived from filename)
    --date YYYY-MM-DD    Override post date (default: today)
    --unlisted           Mark as hidden (skipped from listing, sitemap, keywords)
    --title TEXT         Skip the title prompt
    --subtitle TEXT      Skip the subtitle prompt (pass "" for no subtitle)
    --keywords TEXT      Skip the keywords prompt (comma-separated)
    --list-unlisted      Print URLs of all unlisted posts and exit
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
POST_FILENAME_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(.+)\.md$")
UNLISTED_RE = re.compile(r"^unlisted:\s*true\s*$", re.MULTILINE)


def derive_slug(filename: str) -> str:
    stem = PREFIX_RE.sub("", Path(filename).stem)
    return stem.lower()


def list_unlisted(site_root: Path) -> int:
    posts_dir = site_root / "_posts"
    for path in sorted(posts_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        end = text.find("\n---", 3)
        if end == -1:
            continue
        if not UNLISTED_RE.search(text[3:end]):
            continue
        m = POST_FILENAME_RE.match(path.name)
        if not m:
            continue
        yyyy, mm, dd, slug = m.groups()
        print(f"/blog/{yyyy}/{mm}/{dd}/{slug}/")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("source", nargs="?", default=None)
    p.add_argument("--slug", default=None)
    p.add_argument("--date", default=dt.date.today().isoformat())
    p.add_argument("--unlisted", action="store_true")
    p.add_argument("--title", default=None)
    p.add_argument("--subtitle", default=None)
    p.add_argument("--keywords", default=None)
    p.add_argument("--list-unlisted", action="store_true")
    args = p.parse_args()

    site_root = Path(__file__).resolve().parent.parent

    if args.list_unlisted:
        return list_unlisted(site_root)
    if not args.source:
        p.error("source is required unless --list-unlisted is given")

    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        sys.exit(f"no such file: {source}")

    slug = args.slug or derive_slug(source.name)
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
    fm += ["render_with_liquid: false", "published: true", "---", ""]
    frontmatter = "\n".join(fm) + "\n"

    post_path.parent.mkdir(parents=True, exist_ok=True)
    post_path.write_text(frontmatter + body, encoding="utf-8")

    yyyy, mm, dd = args.date.split("-")
    print(f"\nPublished: {post_path.relative_to(site_root)}")
    print(f"URL after deploy: /blog/{yyyy}/{mm}/{dd}/{slug}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
