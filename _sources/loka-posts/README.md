# Loka article imports

Full-page HTML snapshots retrieved on 2026-10-26 are kept here for provenance.
Jekyll excludes this underscore-prefixed directory from the published site.
`manifest.json` records the canonical URLs, snapshot SHA-256 hashes, destination
posts, and the original URLs and SHA-256 hashes of all 17 downloaded assets.
The posts are maintained as Markdown in `_posts/`.

## Extraction

Both published pages deliver the article text in server-rendered HTML. No
browser automation, OCR, PDF conversion, or JavaScript rendering is required.
For the technical blog, select `article.content`; for Loka's Webflow blog,
select `.blog-content.w-richtext`. Strip navigation and retain only the article.
Beautiful Soup and Markdownify were used through `uv run --with beautifulsoup4
--with markdownify python` to turn paragraphs, headings, lists, and links into
Markdown. Keep figures as HTML to preserve captions, photo layouts, and the
leaderboard crop. For example, to inspect the extracted AgentCore text:

```sh
uv run --with beautifulsoup4 --with markdownify python - <<'PY'
from pathlib import Path
from bs4 import BeautifulSoup
from markdownify import markdownify
html = Path('_sources/loka-posts/agentcore.html').read_text()
article = BeautifulSoup(html, 'html.parser').select_one('.blog-content.w-richtext')
assert article is not None
print(markdownify(str(article), heading_style='ATX'))
PY
```

The imported copies preserve the published titles, dates (TorchNeuron:
2026-10-13; AgentCore: 2026-10-21, using UTC dates so the CI timezone does not
shift them to the previous day), author credits, source links, and article
prose, with these presentation changes:

- Remove the verbatim duplicated opening paragraph in AgentCore section 1,
  empty zero-width paragraphs, and empty superscript markers.
- Number all 13 AgentCore sources and link each superscript to its source.
- Save all article/cover images locally, remove remote responsive-image URLs,
  and resolve technical-blog links against the original publication URL.
- Retain the original full-size photo and leaderboard image links.
- Add descriptions to AgentCore figures whose source alt text was empty.
- Remove the technical blog's generated heading-anchor widgets and allow
  ordinary figures to scale without distorting their aspect ratios.

## Validation

From the repository root:

```sh
JEKYLL_ENV=production bundle exec jekyll build --destination /tmp/personal-blog-preview
python3 -m http.server 8765 --bind 127.0.0.1 --directory /tmp/personal-blog-preview
```

Open `/celebrating-the-release-of-torchneuron.html` and
`/agentcore-memory-is-all-you-need.html`. The homepage, archive, and feed pick
up both posts automatically. The import was checked for source-paragraph
coverage, local image and link targets, citation anchors, asset hashes, and
exclusion of `_sources/` from the output. Desktop and mobile previews were
inspected in the browser.
