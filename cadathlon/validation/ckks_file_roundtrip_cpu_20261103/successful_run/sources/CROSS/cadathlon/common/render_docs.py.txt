"""Render handouts in the CROSS tutorial style, with embedded offline fonts."""
from base64 import b64encode
from html import escape
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import markdown
from weasyprint import HTML
from weasyprint.text.fonts import FontConfiguration

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "common"
PROBLEMS = {"non_agent_pset/p1_benchmark": "MNIST throughput",
            "agent_pset/p2_bert": "BERT / MRPC",
            "non_agent_pset/p3_cnn": "Encrypted CNNs",
            "agent_pset/p4_qwen": "Qwen3-0.6B"}


def font_styles():
    rules = []
    for family in ("Inter", "Outfit"):
        font = b64encode((COMMON / "fonts" / f"{family}.ttf").read_bytes()).decode()
        rules.append(f"@font-face {{ font-family: '{family}'; font-style: normal; "
                     f"font-weight: 100 900; font-display: swap; "
                     f"src: url(data:font/ttf;base64,{font}) format('truetype'); }}")
    return "\n".join(rules)


def icon(paths, class_name=""):
    return (f'<svg class="{class_name}" width="20" height="20" viewBox="0 0 24 24" '
            f'fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">{paths}</svg>')


def navigation(tokens, nested=True):
    entries = []
    for token in tokens:
        text = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", token["name"])
        entries.append(f'<li><a href="#{token["id"]}">{escape(text)}</a></li>')
        if nested:
            for child in token["children"]:
                text = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", child["name"])
                entries.append(f'<li><a class="toc-h3" href="#{child["id"]}">{escape(text)}</a></li>')
    return "\n".join(entries)


def render(problem, fonts):
    source = ROOT / problem / "doc" / "problem.md"
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc"],
                           extension_configs={"toc": {"toc_depth": "2-3"}})
    document = ET.fromstring("<div>" + md.convert(source.read_text()) + "</div>")
    nodes = list(document)
    heading, metadata = nodes[:2]
    title = "".join(heading.itertext())
    metadata.set("class", "subtitle")
    hero = ET.Element("section", {"class": "hero-section", "id": "overview"})
    # `problem` is now "<pset>/<pN_name>", so take the leaf before the number.
    number = int(Path(problem).name.split("_", 1)[0][1:])
    ET.SubElement(hero, "p", {"class": "eyebrow"}).text = f"CROSS / PROBLEM {number:02d}"
    hero.extend((heading, metadata))
    sections, current = [hero], hero
    for node in nodes[2:]:
        if node.tag == "h2":
            current = ET.Element("section", {"id": node.attrib.pop("id")})
            sections.append(current)
        current.append(node)
    content = "\n".join(ET.tostring(section, encoding="unicode", method="html") for section in sections)
    tracks = "\n".join(
        f'<li><a href="../../../{key}/doc/problem.html"'
        + (' class="active" aria-current="page"' if key == problem else '')
        + f'>{escape(label)}</a></li>' for key, label in PROBLEMS.items())
    menu = icon('<path d="M3 6h18M3 12h18M3 18h18"/>')
    close = icon('<path d="m6 6 12 12M6 18 18 6"/>')
    sun = icon('<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/>', "sun-icon")
    moon = icon('<path d="M21 12.8A9 9 0 0 1 11.2 3 9 9 0 1 0 21 12.8Z"/>', "moon-icon")
    download = icon('<path d="M12 3v12m-5-5 5 5 5-5M5 16v5h14v-5"/>')
    style = fonts + "\n" + (COMMON / "tutorial.css").read_text()
    script = (COMMON / "tutorial.js").read_text()
    html = f'''<!doctype html>
<html lang="en" data-theme="dark">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{escape(title)} — CROSS CADathlon problem set.">
<title>{escape(title)} · CADathlon</title>
<!-- Inter and Outfit are embedded for offline use; SIL OFL notices are included below. -->
<style>{style}</style></head>
<body>
<a class="skip-link" href="#overview">Skip to problem</a>
<div class="app-layout">
  <aside class="sidebar" id="sidebar" aria-label="CADathlon navigation">
    <div class="sidebar-header"><a class="logo" href="#overview">CROSS CADATHLON</a>
      <button id="mobile-close-btn" class="icon-btn" aria-label="Close menu">{close}</button></div>
    <nav class="sidebar-nav">
      <div class="nav-section"><h2 class="nav-title">Problem set</h2><ul class="nav-links">{tracks}</ul></div>
      <div class="nav-section"><h2 class="nav-title">This problem</h2><ul class="nav-links">{navigation(md.toc_tokens, nested=False)}</ul></div>
      <div class="nav-section"><h2 class="nav-title">Resources</h2><ul class="nav-links">
        <li><a href="https://efficientppml.github.io/CROSS_Tutorial/">CROSS tutorial ↗</a></li>
        <li><a href="https://github.com/EfficientPPML/CROSS">CROSS library ↗</a></li>
        <li><a href="problem.pdf">PDF handout ↓</a></li>
      </ul></div>
    </nav>
  </aside>
  <div class="menu-backdrop" hidden></div>
  <main class="main-content">
    <header class="top-bar">
      <div class="header-group"><button id="mobile-menu-btn" class="icon-btn" aria-label="Toggle navigation" aria-controls="sidebar" aria-expanded="true">{menu}</button><span class="header-title">CADathlon problem set</span></div>
      <div class="top-actions"><a class="download-link" href="problem.pdf" aria-label="Download PDF">{download}<span>PDF</span></a>
        <button id="theme-toggle" class="icon-btn" aria-label="Switch to light theme">{sun}{moon}</button></div>
    </header>
    <div class="content-wrapper">
      <article class="doc-content" aria-label="Problem handout">{content}</article>
      <aside class="on-this-page" aria-label="On this page"><h2>On this page</h2><ul>{navigation(md.toc_tokens)}</ul></aside>
    </div>
    <footer class="site-footer">CROSS · CADathlon<span>Privacy-preserving inference on AI accelerators</span></footer>
  </main>
</div>
<script>{script}</script>
</body></html>'''
    licenses = "\n\n".join((COMMON / "fonts" / f"{family}-OFL.txt").read_text() for family in ("Inter", "Outfit"))
    html += "\n<!-- Embedded font license notices:\n" + licenses.replace("--", "—") + "\n-->\n"
    source.with_suffix(".html").write_text(html)
    HTML(string=html, base_url=str(source.parent)).write_pdf(
        source.with_suffix(".pdf"), font_config=FontConfiguration())
    print(source.with_suffix(".pdf"))


if __name__ == "__main__":
    embedded_fonts = font_styles()
    for name in PROBLEMS:
        render(name, embedded_fonts)
