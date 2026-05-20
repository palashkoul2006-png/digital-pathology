"""
export_diagrams.py
==================
Reads architecture_diagram.md and workflow_diagram.md,
extracts every Mermaid code block, renders each one via
the mermaid.ink public API (no local install needed),
and assembles a fully-formatted Word document.

Requirements:
    pip install python-docx requests

Output:
    Digital_Pathology_Diagrams.docx  (in the project folder)
"""

import re
import base64
import io
import os
import sys
import requests
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

# ── Config ────────────────────────────────────────────────────────────────────
ARTIFACT_DIR = r"C:\Users\palas\.gemini\antigravity\brain\31a5359d-3a03-48d2-be3d-68f21376b24d"
MD_FILES = [
    (os.path.join(ARTIFACT_DIR, "architecture_diagram.md"), "Architecture Diagram"),
    (os.path.join(ARTIFACT_DIR, "workflow_diagram.md"),     "Workflow Diagrams"),
]

# Save directly into the project folder (same folder as this script)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(PROJECT_DIR, "Digital_Pathology_Diagrams.docx")


# mermaid.ink renders a Mermaid diagram from a base64-encoded definition
MERMAID_INK_URL = "https://mermaid.ink/img/{code}?type=png&width=1200"

# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_mermaid(diagram_text: str) -> str:
    """Base64-encode a Mermaid diagram definition for mermaid.ink."""
    encoded = base64.urlsafe_b64encode(diagram_text.encode("utf-8")).decode("utf-8")
    return encoded


KROKI_URL = "https://kroki.io/mermaid/png"

def render_diagram(diagram_text: str, label: str) -> bytes | None:
    """
    Try mermaid.ink first; fall back to kroki.io on failure.
    Returns PNG bytes or None if both fail.
    """
    # ── Attempt 1: mermaid.ink ────────────────────────────────────────────────
    try:
        code = encode_mermaid(diagram_text)
        url  = MERMAID_INK_URL.format(code=code)
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("image"):
            print(f"  ✅ Rendered via mermaid.ink : {label}")
            return resp.content
        else:
            print(f"  ⚠️  mermaid.ink {resp.status_code} → trying kroki.io : {label}")
    except Exception as e:
        print(f"  ⚠️  mermaid.ink error ({e}) → trying kroki.io : {label}")

    # ── Attempt 2: kroki.io ───────────────────────────────────────────────────
    try:
        import zlib, base64 as _b64
        compressed = zlib.compress(diagram_text.encode("utf-8"), 9)
        encoded    = _b64.urlsafe_b64encode(compressed).decode("utf-8")
        kroki_url  = f"https://kroki.io/mermaid/png/{encoded}"
        resp2 = requests.get(kroki_url, timeout=40)
        if resp2.status_code == 200 and resp2.headers.get("Content-Type", "").startswith("image"):
            print(f"  ✅ Rendered via kroki.io   : {label}")
            return resp2.content
        else:
            print(f"  ❌ kroki.io also failed ({resp2.status_code}) for: {label}")
            print(f"     Diagram source:\n{diagram_text[:300]}...")
    except Exception as e2:
        print(f"  ❌ kroki.io error ({e2}) for: {label}")

    return None


def extract_sections(md_text: str):
    """
    Parse a markdown file into a list of sections.
    Each section is a dict:
      { 'heading': str, 'level': int, 'body': str,
        'diagrams': [(title, mermaid_code), ...] }
    """
    sections = []
    # Split on headings (## or ###)
    parts = re.split(r'(^#{1,4} .+$)', md_text, flags=re.MULTILINE)

    current_heading = "Introduction"
    current_level   = 1
    current_body    = ""

    for part in parts:
        heading_match = re.match(r'^(#{1,4}) (.+)$', part)
        if heading_match:
            if current_body.strip():
                sections.append(_make_section(current_heading, current_level, current_body))
            current_level   = len(heading_match.group(1))
            current_heading = heading_match.group(2).strip()
            current_body    = ""
        else:
            current_body += part

    if current_body.strip():
        sections.append(_make_section(current_heading, current_level, current_body))

    return sections


def _make_section(heading, level, body):
    """Extract mermaid blocks from a section body."""
    diagrams = []
    mermaid_pattern = re.compile(r'```mermaid\s*\n(.*?)```', re.DOTALL)
    for match in mermaid_pattern.finditer(body):
        diagrams.append((heading, match.group(1).strip()))

    # Remove mermaid blocks from body text
    clean_body = mermaid_pattern.sub('[Diagram rendered below]', body)
    return {
        'heading': heading,
        'level':   level,
        'body':    clean_body.strip(),
        'diagrams': diagrams,
    }


def add_heading(doc, text, level):
    """Add a styled heading to the Word document."""
    style_map = {1: 'Heading 1', 2: 'Heading 2', 3: 'Heading 3', 4: 'Heading 4'}
    style = style_map.get(level, 'Heading 3')
    h = doc.add_heading(text, level=level)
    h.style = doc.styles[style]
    return h


def add_body_text(doc, text):
    """Add non-empty plain-text paragraphs (skip code fences and placeholders)."""
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('```') or line == '[Diagram rendered below]':
            continue
        # Strip leading markdown symbols
        line = re.sub(r'^[>\-\*\|]+\s*', '', line)
        line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)   # bold
        line = re.sub(r'\*(.+?)\*',     r'\1', line)   # italic
        if line:
            p = doc.add_paragraph(line)
            p.paragraph_format.space_after = Pt(2)


def add_diagram_image(doc, png_bytes, caption):
    """Insert a PNG image into the document with a caption."""
    img_stream = io.BytesIO(png_bytes)
    doc.add_picture(img_stream, width=Inches(6.2))

    # Caption paragraph
    cap = doc.add_paragraph(f"Figure: {caption}")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.runs[0]
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    doc.add_paragraph()   # spacing


# ── Build the Word Document ────────────────────────────────────────────────────

def build_word_doc():
    print("\n🚀 Digital Pathology — Diagram Export to Word")
    print("=" * 50)

    doc = Document()

    # ── Page margins (narrow for bigger diagrams)
    from docx.oxml.ns import qn
    from docx.oxml   import OxmlElement
    section = doc.sections[0]
    section.page_width   = Inches(8.5)
    section.page_height  = Inches(11)
    section.left_margin  = Inches(0.9)
    section.right_margin = Inches(0.9)
    section.top_margin   = Inches(1.0)
    section.bottom_margin = Inches(1.0)

    # ── Cover title
    title = doc.add_heading("Digital Pathology AI", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph("Architecture & Workflow Diagrams")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].font.size = Pt(14)
    sub.runs[0].font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    doc.add_paragraph()
    doc.add_paragraph("Project: Skin Cancer Analysis using Vision Transformer (ViT)").paragraph_format.space_after = Pt(2)
    doc.add_paragraph("Model: fine-tuned google/vit-base-patch16-224 (HAM10000, 98.52% acc)").paragraph_format.space_after = Pt(2)
    doc.add_paragraph("Stack: Flask · PyTorch · Stable Diffusion · Gemini 2.5 Flash · MongoDB").paragraph_format.space_after = Pt(2)
    doc.add_page_break()

    diagram_count = 0
    failed_count  = 0

    for md_path, file_label in MD_FILES:
        if not os.path.exists(md_path):
            print(f"\n⚠️  File not found: {md_path}")
            continue

        print(f"\n📄 Processing: {file_label}")
        with open(md_path, encoding='utf-8') as f:
            md_text = f.read()

        sections = extract_sections(md_text)

        # File-level heading
        doc.add_heading(file_label, level=1)

        for sec in sections:
            if sec['heading'].lower() in ('', 'digital pathology ai — architecture diagram',
                                          'digital pathology ai — workflow diagrams'):
                continue

            add_heading(doc, sec['heading'], level=2)

            # Body text (descriptions, tables as plain text)
            if sec['body']:
                add_body_text(doc, sec['body'])

            # Render and insert each mermaid diagram
            for caption, mermaid_code in sec['diagrams']:
                print(f"\n  🔄 Rendering diagram: {caption}")
                png_bytes = render_diagram(mermaid_code, caption)
                if png_bytes:
                    add_diagram_image(doc, png_bytes, caption)
                    diagram_count += 1
                else:
                    p = doc.add_paragraph(f"⚠️ Could not render diagram: {caption}")
                    p.runs[0].font.color.rgb = RGBColor(0xCC, 0x00, 0x00)
                    failed_count += 1

        doc.add_page_break()

    # ── Save (via BytesIO to avoid OneDrive/Word file-lock PermissionError) ────
    import io as _io
    buf = _io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    with open(OUTPUT_FILE, 'wb') as out_f:
        out_f.write(buf.read())
    print(f"\n{'='*50}")
    print(f"✅ Word document saved to:\n   {OUTPUT_FILE}")
    print(f"   Diagrams rendered : {diagram_count}")
    if failed_count:
        print(f"   ⚠️  Failed renders  : {failed_count}")
    print("="*50)


if __name__ == '__main__':
    # Check dependencies
    missing = []
    try:
        import docx
    except ImportError:
        missing.append('python-docx')
    try:
        import requests
    except ImportError:
        missing.append('requests')

    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"   Run:  pip install {' '.join(missing)}")
        sys.exit(1)

    build_word_doc()
