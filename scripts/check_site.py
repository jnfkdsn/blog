"""Check generated local pages, fragment targets, and embedded assets.

Run after `npm run build`: python3 scripts/check_site.py
External URLs are deliberately not fetched. This checks publication paths,
including VitePress's base prefix, rather than only Markdown source filenames.
"""

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.refs = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "name" in attrs:
            self.ids.add(attrs["name"])
        if tag in {"a", "link"} and attrs.get("href"):
            self.refs.append(attrs["href"])
        if tag in {"img", "script", "source", "video", "audio"} and attrs.get("src"):
            self.refs.append(attrs["src"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs/.vitepress/dist")
    parser.add_argument("--base", default="/blog/")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    dist = args.dist.resolve()
    base = "/" + args.base.strip("/") + "/" if args.base.strip("/") else "/"
    pages = {p.resolve(): Page(p.read_text()) for p in dist.rglob("*.html")}
    if not pages:
        parser.error(f"No built HTML in {dist}; run npm run build first")
    failures = []
    links = set()
    for path, page in pages.items():
        current = base + path.relative_to(dist).as_posix()
        for raw in set(page.refs):
            url = urlsplit(raw)
            if url.scheme or url.netloc:
                continue
            target_url = urlsplit(urljoin(current, raw))
            if not target_url.path.startswith(base):
                failures.append([str(path.relative_to(dist)), raw, "outside site base"])
                continue
            rel = unquote(target_url.path[len(base):])
            target = dist / rel
            candidates = [target, target / "index.html"]
            if not target.suffix:
                candidates.append(target.with_suffix(".html"))
            resolved = next((p.resolve() for p in candidates if p.is_file()), None)
            if resolved is None or not resolved.is_relative_to(dist):
                failures.append([str(path.relative_to(dist)), raw, "missing local target"])
                continue
            fragment = unquote(target_url.fragment)
            if fragment and resolved in pages and fragment not in pages[resolved].ids:
                failures.append([str(path.relative_to(dist)), raw, "missing fragment"])
                continue
            links.add((str(resolved.relative_to(dist)), fragment))
    report = {"html_pages": len(pages), "unique_local_targets": len(links),
              "failures": failures, "external_urls_checked": False}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
