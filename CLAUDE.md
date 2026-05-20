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

## Architecture

This is a personal academic website (`masani` / Mohammad Hossein Amani) built with **Jekyll 4** and deployed to GitHub Pages via the workflow in `.github/workflows/jekyll.yml` on every push to `main`. The deploy uses native Jekyll (not GitHub Pages' restricted gem set), so non-whitelisted plugins like `jekyll-scholar` work.

### Page model

Top-level `.html` files (`index.html`, `about.html`, `publications.html`, `blog/blog.html`) are entry points with front-matter `permalink`s. Some use `layout: default` (wraps content in the title/byline/bibliography frame from `_layouts/default.html`); others are self-contained HTML that just `{% include nav.html %}`. The nav (`_includes/nav.html`) is the canonical site map: home, about, notebook, publications.

### Blog posts (`_posts/`) and the keywords plugin

Posts use the `default` layout and render math via MathJax (loaded by `_includes/mathjax.html`, configured for `$...$` and `$$...$$` delimiters). Each post declares comma-separated `keywords:` in front matter. The `blog/blog.html` page is a 6-line stub that just sets `layout: blog`; the real listing template is `_layouts/blog.html`. After every build, `_plugins/generate_keywords.rb` (a Jekyll `:site, :post_write` hook) walks all posts, collects unique keywords, and writes one stub markdown file per keyword into `blog/keywords/`. Each stub uses `layout: blog` and a `keyword:` front-matter field; the layout reads `page.keyword` to filter the post list. **Do not hand-edit files under `blog/keywords/`** — they are regenerated on every build. The permalink scheme for posts is `/blog/:year/:month/:day/:title/` (see `_config.yml`).

Posts marked `unlisted: true` in their front matter are filtered out of the blog listing, the keyword filter pills, the `<meta keywords>` tag, and the keyword-stub generator. Combined with `sitemap: false`, this gives an unguessable-by-browsing post that still renders at its normal permalink. Filtering happens in three places: two loops in `_layouts/blog.html` and the `each` in `_plugins/generate_keywords.rb` — keep them in sync if you add another loop over `site.posts`. **Caveat:** `jekyll-feed` has no per-post exclude, so unlisted posts still appear in `/feed.xml`.

Citations inside posts use `jekyll-scholar` against `_bibliography/references.bib`; `_layouts/default.html` renders `{% bibliography --cited %}` at the bottom of each article.

### Publishing notes from the notebook

`scripts/publish-post.py` converts a markdown note from `~/repos/notebook/` (or anywhere) into a Jekyll post. It:

1. Derives a slug by stripping `LITREV-`/`NOTE-`/`IDEA-`/`DEV-`/`REVIEW-` from the filename and lowercasing.
2. Strips the first body H1 (the `default` layout already renders `{{ page.title }}` from front matter).
3. Finds every `(\.\./)+assets/foo.png` reference, resolves it relative to the source file, copies the image to `assets/images/<slug>/foo.png`, and rewrites the path to absolute.
4. Writes `_posts/YYYY-MM-DD-<slug>.md` with frontmatter (`title`, `subtitle`, `layout: default`, `date`, `keywords`, optionally `unlisted: true` + `sitemap: false`, plus `render_with_liquid: false`).

Title/subtitle/keywords are prompted interactively unless passed as flags. Example:

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

CSS lives in `css/`. `blog.less` is the source for `blog.css`; if you change styles, edit the `.less` and recompile, or edit the `.css` directly and keep them in sync. `trac.css` is syntax highlighting (Rouge) and `markdown.css` styles article body text.

### Conventions

- Math is rendered with **MathJax**, not KaTeX, despite `jekyll-katex` being in the `Gemfile`/`_config.yml`. The KaTeX plugin is currently dead weight — the `default` layout only loads MathJax. If you write a new layout, include `_includes/mathjax.html` for math support, or remove `jekyll-katex` from the Gemfile if cleaning up.
- `_config.yml` has `url: "http://127.0.0.1:4000"` hard-coded for local dev. The GitHub Actions build passes `--baseurl` explicitly, so don't rely on `site.url` for production links — use `relative_url`/`prepend: site.baseurl` like the existing templates do.
