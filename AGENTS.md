# AGENTS.md

## Overview

Jekyll-based deep learning course with multi-language support (English/Vietnamese). Deploys to GitHub Pages.

## Quick Commands

```bash
# Local development
bundle install
bundle exec jekyll serve

# Docker alternative
docker-compose up  # serves on localhost:4000

# Build only
bundle exec jekyll build
```

## Project Structure

```
contents/
  en/chapter00-25/       # English content
  vi/chapter00-25/       # Vietnamese content (mirrors en/)
_plugins/                # Custom Jekyll plugins for multilingual support
_layouts/                # page.html, post.html, default.html
_site/                   # Generated output (gitignored)
```

## Content Authoring

### Post frontmatter (required)

```yaml
---
layout: post
title: "00 Introduction"
chapter: '00'
order: 1           # Controls sort order within chapter
owner: Author Name
lang: en           # or 'vi'
categories:
  - chapter00
---
```

### File naming convention

Posts live in `contents/{lang}/chapter{NN}/_posts/` with format:
```
YYYY-MM-DD-{chapter}_{order}_{Title_With_Underscores}.md
```

Example: `2021-01-01-00_01_Calculus.md`

### Adding a new section

1. Create the `.md` file in both `contents/en/chapter{NN}/_posts/` and `contents/vi/chapter{NN}/_posts/`
2. Ensure matching `chapter` and `order` frontmatter for language switching to work

## Multilingual System

- Custom plugins in `_plugins/multilang.rb` handle language switching
- Translations for UI strings live in `_config.yml` under `t.en` and `t.vi`
- Language switch finds matching content by `chapter` + `order` frontmatter

## CI/CD

- GitHub Actions workflow at `.github/workflows/jekyll.yml`
- Triggers on push/PR to `main`
- Ruby 3.2, deploys via GitHub Pages

## Cursor Rules Reference

Two rules in `.cursor/rules/` for content generation:
- `deep-learning-theory.mdc` - Lecture note style guide (math, intuition, code snippets, papers)
- `excersices.mdc` - Jupyter notebook exercise generation patterns
