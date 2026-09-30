#!/usr/bin/env python3
"""Publish a notebook markdown note as a Jekyll post.

What this does (publish path, given a source markdown file):

  1. If the note begins with a `---` YAML block, lift `title`, `subtitle`,
     and `keywords` from it (skipping the interactive prompts for the
     fields it finds). Simple `key: value` scalars only.
  2. Derive a slug by stripping LITREV-/NOTE-/IDEA-/DEV-/REVIEW- prefixes
     from the filename and lowercasing.
  3. Strip the first body H1 (the `default` layout already renders
     `{{ page.title }}` from front matter, so a body H1 would duplicate it).
  4. Rewrite Obsidian image embeds `![[foo.png]]` (with optional `|width`)
     by resolving `foo.png` in the source dir's `assets/`, the source dir,
     the enclosing vault's `assets/`, or vault root (vault = first ancestor
     containing `.obsidian/`). Copy into `assets/images/<slug>/` and rewrite
     as `![](/assets/images/<slug>/foo.png)`, or `<img ... width="N">` when
     a width is given.
  5. Find every `(../)+assets/foo.png` reference, resolve relative to the
     source file, copy into the same `assets/images/<slug>/`, and rewrite
     the path to absolute.
  6. Resolve pandoc citations `[@key]` and `[@k1; @k2]` against
     `_bibliography/references.bib` at publish time (Liquid is off in the
     post body — see below — so `{% cite %}` is not an option). Inline
     text becomes `(Bishop, 2006)` / `(Smith & Doe, 2020)` /
     `(Smith et al., 2024)`. A `## References` section is appended with
     author-year formatted entries, alphabetized by first author.
  7. Warn on unresolved `[[wikilinks]]` (left literal in the output — the
     user manually pastes URLs later if needed).
  8. Write `_posts/YYYY-MM-DD-<slug>.md`. The emitted frontmatter always
     includes `render_with_liquid: false` — notebook notes commonly
     contain `{{ ... }}` and `{% ... %}` literals in code samples and
     prompt templates, which Jekyll would otherwise try to parse as
     Liquid and fail the build.
  9. With `--unlisted`, also emit `unlisted: true` and `sitemap: false`,
     which the listing template (`_layouts/blog.html`) and the keyword
     plugin (`_plugins/generate_keywords.rb`) filter out.

The list path (`--list-unlisted`) walks every file in `_posts/` and prints
the public URL for posts marked `unlisted: true`.

Usage:
    scripts/publish-post.py <path/to/note.md> [options]
    scripts/publish-post.py --list-unlisted

Options:
    --slug NAME          Override slug (default: derived from filename)
    --date YYYY-MM-DD    Override post date (default: today)
    --unlisted           Mark as hidden (skipped from listing, sitemap, keywords)
    --title TEXT         Skip the title prompt / frontmatter lookup
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
OBSIDIAN_IMG_RE = re.compile(r"!\[\[([^\]|]+?)(?:\|(\d+))?\]\]")
WIKILINK_RE = re.compile(r"(?<!!)\[\[([^\]]+)\]\]")
CITE_RE = re.compile(r"\[@([^\]]+)\]")
H1_RE = re.compile(r"^# .*$", re.MULTILINE)
POST_FILENAME_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(.+)\.md$")
UNLISTED_RE = re.compile(r"^unlisted:\s*true\s*$", re.MULTILINE)
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def derive_slug(filename: str) -> str:
    stem = PREFIX_RE.sub("", Path(filename).stem)
    return stem.lower()


def parse_frontmatter(text: str):
    """Return (fields_dict, body_without_frontmatter). Minimal YAML: scalar values only."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fields = {}
    for line in m.group(1).split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        # Skip lines that look like list items or nested keys.
        if line.startswith((" ", "\t", "-")):
            continue
        k, _, v = stripped.partition(":")
        k = k.strip()
        v = v.strip()
        if v.startswith(('"', "'")) and v.endswith(v[0]) and len(v) >= 2:
            v = v[1:-1]
        fields[k] = v
    return fields, text[m.end():]


def find_vault_root(source_dir: Path) -> Path:
    p = source_dir
    while p != p.parent:
        if (p / ".obsidian").is_dir():
            return p
        p = p.parent
    return source_dir


def find_obsidian_image(name: str, source_dir: Path, vault_root: Path):
    candidates = [
        source_dir / "assets" / name,
        source_dir / name,
        vault_root / "assets" / name,
        vault_root / name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def process_obsidian_images(body: str, source_dir: Path, img_dest: Path, slug: str) -> str:
    vault_root = find_vault_root(source_dir)

    def repl(m):
        name = m.group(1).strip()
        width = m.group(2)
        src_img = find_obsidian_image(name, source_dir, vault_root)
        if src_img is None:
            print(f"  WARN missing obsidian image: {name}", file=sys.stderr)
            return m.group(0)
        img_dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, img_dest / src_img.name)
        print(f"  copied: {src_img.name}")
        url = f"/assets/images/{slug}/{src_img.name}"
        if width:
            return f'<img src="{url}" width="{width}">'
        return f"![]({url})"

    return OBSIDIAN_IMG_RE.sub(repl, body)


def process_relative_assets(body: str, source_dir: Path, img_dest: Path, slug: str) -> str:
    refs = sorted(set(ASSET_PATH_RE.findall(body)))
    if refs:
        img_dest.mkdir(parents=True, exist_ok=True)
    for ref in refs:
        src_img = (source_dir / ref).resolve()
        if not src_img.is_file():
            print(f"  WARN missing image: {ref} -> {src_img}", file=sys.stderr)
            continue
        shutil.copy2(src_img, img_dest / src_img.name)
        print(f"  copied: {src_img.name}")
    return ASSET_PATH_RE.sub(
        lambda m: f"/assets/images/{slug}/{Path(m.group(0)).name}",
        body,
    )


def parse_bibliography(path: Path) -> dict:
    """Minimal BibTeX parser. Handles brace-quoted fields with one level of nesting."""
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    entries = {}
    i = 0
    while True:
        m = re.search(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text[i:])
        if not m:
            break
        start = i + m.end()
        key = m.group(2).strip()
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        body = text[start:j - 1]
        fields = {}
        for fm in re.finditer(
            r"(\w+)\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|\"([^\"]*)\"|(\d+))",
            body,
        ):
            name = fm.group(1).lower()
            val = fm.group(2) or fm.group(3) or fm.group(4) or ""
            fields[name] = re.sub(r"\s+", " ", val).strip()
        entries[key] = fields
        i = j
    return entries


def author_last_names(author_field: str):
    if not author_field:
        return []
    names = []
    for a in re.split(r"\s+and\s+", author_field):
        a = a.strip()
        if not a:
            continue
        if "," in a:
            names.append(a.split(",")[0].strip())
        else:
            parts = a.split()
            names.append(parts[-1] if parts else a)
    return names


def abbreviate_author(author: str) -> str:
    """`Bishop, Christopher M` -> `Bishop, C. M.`. `Christopher M Bishop` -> `Bishop, C. M.`."""
    author = author.strip()
    if not author:
        return author
    if "," in author:
        last, first = author.split(",", 1)
        first = first.strip()
        initials = " ".join(f"{p[0]}." for p in first.split() if p)
        return f"{last.strip()}, {initials}".rstrip(", ")
    parts = author.split()
    if len(parts) < 2:
        return author
    last = parts[-1]
    initials = " ".join(f"{p[0]}." for p in parts[:-1] if p)
    return f"{last}, {initials}"


def format_authors_for_reference(author_field: str) -> str:
    authors = [a.strip() for a in re.split(r"\s+and\s+", author_field) if a.strip()]
    if not authors:
        return ""
    formatted = [abbreviate_author(a) for a in authors]
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} & {formatted[1]}"
    return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"


def strip_braces(s: str) -> str:
    return re.sub(r"[{}]", "", s)


def format_inline_citation(entries: dict, keys) -> str:
    parts = []
    for key in keys:
        entry = entries.get(key)
        if entry is None:
            print(f"  WARN unknown citation key: {key}", file=sys.stderr)
            parts.append(f"@{key}")
            continue
        authors = author_last_names(entry.get("author", ""))
        year = entry.get("year", "n.d.")
        if not authors:
            name = "Anonymous"
        elif len(authors) == 1:
            name = authors[0]
        elif len(authors) == 2:
            name = f"{authors[0]} & {authors[1]}"
        else:
            name = f"{authors[0]} et al."
        parts.append(f"{name}, {year}")
    return "(" + "; ".join(parts) + ")"


def format_reference_entry(entry: dict) -> str:
    authors = format_authors_for_reference(entry.get("author", ""))
    year = entry.get("year", "n.d.")
    title = strip_braces(entry.get("title", "")).strip().rstrip(".")
    journal = strip_braces(entry.get("journal", "")).strip()
    booktitle = strip_braces(entry.get("booktitle", "")).strip()
    publisher = strip_braces(entry.get("publisher", "")).strip()
    volume = entry.get("volume", "").strip()
    number = entry.get("number", "").strip()
    pages = entry.get("pages", "").strip().replace("--", "–")

    lead = f"{authors} ({year})." if authors else f"({year})."
    parts = [lead, f"{title}."]
    if journal:
        loc = f"*{journal}*"
        if volume:
            loc += f", {volume}"
            if number:
                loc += f"({number})"
        if pages:
            loc += f", {pages}"
        parts.append(loc + ".")
    elif booktitle:
        parts.append(f"In *{booktitle}*.")
    if publisher:
        parts.append(f"{publisher}.")
    return " ".join(parts)


def process_citations(body: str, bib_entries: dict):
    cited_order = []

    def repl(m):
        raw = m.group(1)
        keys = []
        for chunk in raw.split(";"):
            token = chunk.strip().lstrip("@").strip()
            # Drop pandoc-style locator suffixes like ", p. 42".
            token = token.split(",")[0].strip()
            if token:
                keys.append(token)
        for k in keys:
            if k not in cited_order:
                cited_order.append(k)
        return format_inline_citation(bib_entries, keys)

    new_body = CITE_RE.sub(repl, body)
    return new_body, cited_order


def build_references_section(bib_entries: dict, cited_keys) -> str:
    known = [k for k in cited_keys if k in bib_entries]
    if not known:
        return ""

    def sort_key(k):
        authors = author_last_names(bib_entries[k].get("author", ""))
        first = authors[0].lower() if authors else ""
        return (first, bib_entries[k].get("year", ""))

    known.sort(key=sort_key)
    lines = ["", "## References", ""]
    for k in known:
        lines.append(f"- {format_reference_entry(bib_entries[k])}")
    lines.append("")
    return "\n".join(lines)


def warn_wikilinks(body: str) -> None:
    for m in WIKILINK_RE.finditer(body):
        print(f"  WARN unresolved wikilink: [[{m.group(1)}]]", file=sys.stderr)


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

    raw = source.read_text(encoding="utf-8")
    fm_fields, body = parse_frontmatter(raw)

    def resolve(cli_val, fm_key, prompt_msg):
        if cli_val is not None:
            return cli_val
        if fm_fields.get(fm_key):
            return fm_fields[fm_key]
        return input(prompt_msg).strip()

    title = resolve(args.title, "title", "Title: ")
    subtitle = resolve(args.subtitle, "subtitle", "Subtitle (blank for none): ")
    keywords = resolve(args.keywords, "keywords", "Keywords (comma-separated): ")

    body = H1_RE.sub("", body, count=1).lstrip("\n")
    body = process_obsidian_images(body, source.parent, img_dest, slug)
    body = process_relative_assets(body, source.parent, img_dest, slug)

    bib_entries = parse_bibliography(site_root / "_bibliography" / "references.bib")
    body, cited = process_citations(body, bib_entries)
    warn_wikilinks(body)

    refs_section = build_references_section(bib_entries, cited)
    if refs_section:
        body = body.rstrip() + "\n" + refs_section

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
