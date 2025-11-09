# GitHub Pages Deployment Setup

This site uses **GitHub Actions** for deployment to enable custom Jekyll plugins.

## Local Development

```bash
bundle install
bundle exec jekyll serve
```

The site will be available at `http://127.0.0.1:4000/deep-learning-self-learning/`

## GitHub Pages Deployment

This repository is configured to use GitHub Actions for deployment instead of the default GitHub Pages Jekyll build process. This allows us to:

1. Use Jekyll 4.x (instead of GitHub Pages' restricted Jekyll 3.x)
2. Use custom plugins (like our multilang plugin)
3. Have full control over the build process

### Configuration Files

- `.github/workflows/jekyll.yml` - GitHub Actions workflow for building and deploying
- `.nojekyll` - Tells GitHub Pages to skip the default Jekyll build
- `_config.yml` - Configured with proper baseurl for GitHub Pages

### Repository Settings

Make sure your repository has the following settings:
1. Go to **Settings > Pages**
2. Set **Source** to "GitHub Actions"
3. The workflow will automatically deploy on pushes to the `main` branch

## Multilang Support

The site supports English and Vietnamese content with automatic language switching. Custom Liquid tags:

- `{% t key %}` - Translate text based on current language
- `{% language_switch %}` - Generate language switch button
- `{% multilang_post_url %}` - Generate multilingual post URLs

## URL Structure

- English: `/contents/en/chapter00/`
- Vietnamese: `/contents/vi/chapter00/`

All URLs include the repository baseurl for proper GitHub Pages deployment.
