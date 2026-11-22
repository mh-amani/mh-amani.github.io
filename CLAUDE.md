# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Local development server (live reload at http://127.0.0.1:4000):
```bash
bundle exec jekyll serve
```

Production build (outputs to `_site/`):
```bash
bundle exec jekyll build
```

First-time setup: `bundle install` (Ruby 3.1, Jekyll 4.0).

Ruby 3.1 is what CI uses, but Homebrew no longer installs it (`ruby@3.1` is disabled). On Ruby 3.3, Jekyll 4.0 crashes with `undefined method '[]' for nil` in `logger.rb`, because Ruby 3.3 ships logger 1.6. To build locally without touching the repo's Gemfile, use a wrapper Gemfile outside the repo containing `eval_gemfile "<repo>/Gemfile"` and `gem "logger", "< 1.6"`, copy `Gemfile.lock` next to it, and run with `BUNDLE_GEMFILE` pointing at the wrapper.

## Architecture

This is a personal academic website (`masani` / Mohammad Hossein Amani) built with **Jekyll 4** and deployed to GitHub Pages via the workflow in `.github/workflows/jekyll.yml` on every push to `main`. The deploy uses native Jekyll (not GitHub Pages' restricted gem set), so non-whitelisted plugins like `jekyll-scholar` work.

### Page model

Top-level `.html` files (`index.html`, `about.html`, `publications.html`, `blog/blog.html`) are entry points with front-matter `permalink`s. Some use `layout: default` (wraps content in the title/byline/bibliography frame from `_layouts/default.html`); others are self-contained HTML that just `{% include nav.html %}`. The nav (`_includes/nav.html`) is the canonical site map: home, about, notebook, publications.

### Blog posts (`_posts/`) and the keywords plugin

Posts use the `default` layout and render math via MathJax (`_includes/mathjax.html`). Write math Obsidian-style: `$...$` inline, `$$...$$` display. `_plugins/protect_math.rb` is what makes this work. kramdown only knows `$$` math, starts a table on any line containing `|`, and reads `_`, `*`, and `\{` inside `$...$` as Markdown. So the plugin's `:pre_render` hook swaps every math span outside code for a `MJXPH<n>END` token, and its `:post_render` hook turns the tokens into `\(...\)` / `\[...\]` in both `content` (used by the feed) and `output`. Inline math follows pandoc's rule: there's no space after the opening `$` or before the closing one, and no digit right after the closing one, so "$5 and $10" stays text. Write `\$` for a literal dollar sign. MathJax 3 can't line-break inline math, so `_includes/mathjax.html` gives inline formulas wider than their paragraph the `mjx-wide` class, which makes them scroll within the line. Each post declares comma-separated `keywords:` in front matter. The `blog/blog.html` page is a 6-line stub that just sets `layout: blog`; the real listing template is `_layouts/blog.html`. `_plugins/generate_keywords.rb` is a `Jekyll::Generator` that walks all posts, collects unique trimmed keywords, and adds one in-memory page per keyword at `/blog/keywords/<slug>/` (`layout: blog`, `keyword:` set). The layout reads `page.keyword` and shows posts whose comma-separated keywords include it exactly. Nothing is written to `blog/keywords/` in the source tree. (An earlier `:post_write` hook wrote stub files there after the build, so on a clean CI checkout only the committed stubs existed and new keywords 404'd.) The permalink scheme for posts is `/blog/:year/:month/:day/:title/` (see `_config.yml`). The date is part of the URL, so changing a deployed post's date breaks its old link. `redirects/` holds small meta-refresh pages, pinned to old URLs by `permalink:`, that point to the new ones. To date a post when the work was done rather than today, pass `--date` to the publish script.

Posts marked `unlisted: true` in their front matter are filtered out of the blog listing, the keyword filter pills, the `<meta keywords>` tag, the keyword-page generator, and the Atom feed. Combined with `sitemap: false`, this gives an unguessable-by-browsing post that still renders at its normal permalink. Filtering happens in three files: the keyword and post loops in `_layouts/blog.html`, the `each` in `_plugins/generate_keywords.rb`, and the `where_exp` in `feed.xml`. Keep them in sync if you add another loop over `site.posts`.

`feed.xml` at the repo root is a hand-written Atom template adapted from jekyll-feed 0.17.0. jekyll-feed skips generating `/feed.xml` when the source already has one, and its own template has no per-post exclude, so this file is what keeps unlisted posts out of the feed. It lists the 10 newest listed posts.

Citations inside posts use `jekyll-scholar` against `_bibliography/references.bib`; `_layouts/default.html` renders `{% bibliography --cited %}` at the bottom of each article.

### Publishing notes from the notebook

`scripts/publish-post.py` converts a markdown note from `~/repos/notebook/` (or anywhere) into a Jekyll post. It:

1. If the note starts with a `---` YAML block, lifts `title` / `subtitle` / `keywords` from it (skipping the prompt for each field it finds). Simple `key: value` scalars only.
2. Derives a slug by stripping `LITREV-`/`NOTE-`/`IDEA-`/`DEV-`/`REVIEW-` from the filename and lowercasing.
3. Strips the first body H1 (the `default` layout already renders `{{ page.title }}` from front matter).
4. Rewrites Obsidian image embeds `![[foo.png]]` and `![[foo.png|width]]` — resolves against the source dir's `assets/`, the source dir, the enclosing vault's `assets/`, then vault root (vault = first ancestor with `.obsidian/`) — copies into `assets/images/<slug>/` and rewrites as `![](/...)` normally, or `<img src="..." width="N">` when a width suffix is given.
5. Finds every `(\.\./)+assets/foo.png` reference, resolves it relative to the source file, copies the image to `assets/images/<slug>/foo.png`, and rewrites the path to absolute.
6. Resolves pandoc citations `[@key]` and `[@k1; @k2]` against `_bibliography/references.bib` **at publish time** (Liquid is off in the post body — see below — so `{% cite %}` is not an option). Emits `(Author, YYYY)` / `(Author & Coauthor, YYYY)` / `(Author et al., YYYY)` inline, and appends a `## References` section with author-year entries alphabetized by first author. Unknown keys warn on stderr and pass through as `@key`.
7. Warns on unresolved `[[wikilinks]]`, which are left literal (paste blog URLs in manually). Adds a blank line after any table followed directly by non-table text, outside code fences. Obsidian ends a table there, but kramdown turns the whole table into a paragraph of raw pipes.
8. Writes `_posts/YYYY-MM-DD-<slug>.md` with frontmatter (`title`, `subtitle`, `layout: default`, `date`, `keywords`, optionally `unlisted: true` + `sitemap: false`, plus `render_with_liquid: false`).

Title/subtitle/keywords are prompted interactively unless passed as flags or lifted from frontmatter. Example:

```bash
scripts/publish-post.py ~/repos/notebook/notes/LLMRL/LITREV-foo.md \
  --unlisted \
  --title "Foo" --subtitle "A review" --keywords "rl, llms"
```

**`render_with_liquid: false` is emitted by default** because notebook notes routinely contain `{{ ... }}` and `{% ... %}` literals (LM prompt templates, code samples) that Jekyll would otherwise try to parse as Liquid and fail the build. This disables Liquid for the post body only — the layout itself still uses Liquid for `{{ content }}`, the mathjax include, and the bibliography. **A past build failure had this exact cause** (literal `{%- if ... -%}` in a code block from the LITREV).

To list the URLs of all unlisted posts:

```bash
scripts/publish-post.py --list-unlisted
```

The script is the source of truth for the publishing pipeline — when adding new conventions (e.g., per-post asset folders, new front-matter flags), update it rather than documenting a manual workflow.

### Publications

`publications.html` iterates `site.data.publications` (loaded from `_data/publications.yml`). To add a publication, append a YAML entry — supported keys: `title`, `authors` (comma-separated string; any author containing "Amani" is auto-bolded), `year`, `journal`, `arxiv`, `code`, `poster` (filename under `assets/pdf/`), `workshop_link`/`workshop_name`, `description` (renders behind a Show/Hide toggle), plus `selected`/`preview` (currently unused by the template).

### Styling

CSS lives in `css/`. Edit `blog.css` directly: `blog.less` is an old, fully commented-out copy that nothing compiles. `trac.css` is Rouge syntax-highlighting colors. `markdown.css` only styles `.markdown-body`, which no template uses, so it has no effect. `bw.css` isn't linked anywhere.

Two column widths. Most pages (home, about, publications, notebook list) use `.wrap { max-width: 700px }` with 15px `.article` text, and 7px body margins below 700px. That's the original look, and the user wants it kept. Blog posts get `<body class='post'>` from `_layouts/default.html` (when `page.collection == "posts"`). For them, `body.post .wrap` is 860px wide, `body.post .article` text is 16px, and 16px body margins apply below 892px; change those two widths together. The "Long-form posts" block at the end of `blog.css` styles tables (bordered, scrolling sideways inside their box when too wide), code (shaded blocks; fences without a language wrap instead of scrolling, since they're mostly prompts), headings down to h6 (never smaller than body text), and display math (scrolls inside its box when too wide).

### Conventions

- Math in post bodies is rendered with **MathJax** (`_includes/mathjax.html`). If you write a new layout, include it for math support. `jekyll-katex` is still required: `_layouts/blog.html` wraps post titles and subtitles in its `{% katexmm %}` tag, so removing the gem breaks the build.
- `_config.yml` sets `url: "https://mh-amani.github.io"`, which `jekyll-sitemap` and `feed.xml` use for absolute links. `jekyll serve` replaces it with the local address automatically. For in-site links, use `relative_url`/`prepend: site.baseurl` like the existing templates do.
- Every page starts with `<!DOCTYPE html>` (standards mode) and includes `_includes/favicon.html`. Keep both in any new self-contained page or layout. The favicon is a lowercase "m" drawn as strokes in `favicon.svg`, with no font, so it renders the same everywhere. It's dark grey #333, or #e6e6e6 when the browser is in dark mode. `favicon.png` (32px, transparent) and `apple-touch-icon.png` (180px, on white) are renders of it. Re-render both if the SVG changes. The user rejected a photo of themselves and a bike.
