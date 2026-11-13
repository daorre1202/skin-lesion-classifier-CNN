"""Figure 2 of the BIOIMAGING 2027 paper: one correct case per class.

Takes the qualitative panel written by notebooks/03_attention_audit.ipynb and
replaces its labels so they land at 7 pt once the panel is placed at 0.78 of
the column width. Images and contours are copied unchanged.

    python scripts/figures/make_fig2.py PANEL_FROM_NOTEBOOK_03.pdf fig_gradcam_seg_panel.pdf

Needs PyMuPDF and the Liberation Serif font (metric-compatible with Times).
"""
import sys

import pymupdf

FONT = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
DX, DY = 8.0, 8.0                    # extra left gutter and top band, in source points
PLACED_W = 0.78 * 212.598            # width of the panel on the page, pt
HEADERS = ["Original", "Grad-CAM", "Contours"]


def main(src, out):
    font = pymupdf.Font(fontfile=FONT)
    doc = pymupdf.open(src)
    pg = doc[0]
    w0, h0 = pg.rect.width, pg.rect.height
    size = 7.0 / (PLACED_W / (w0 + DX))  # source size that lands at 7 pt

    rows = []
    for b in pg.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                if abs(line["dir"][1]) > 0.5:
                    rows.append((s["text"], (s["bbox"][1] + s["bbox"][3]) / 2))
                pg.add_redact_annot(pymupdf.Rect(s["bbox"]) + (-1.5, -1.5, 1.5, 1.5))
    pg.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                        graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                        text=pymupdf.PDF_REDACT_TEXT_REMOVE)
    cols = sorted({(round(i["bbox"][0]), round(i["bbox"][2])) for i in pg.get_image_info()})
    top = min(i["bbox"][1] for i in pg.get_image_info())

    new = pymupdf.open()
    page = new.new_page(width=w0 + DX, height=h0 + DY)
    page.show_pdf_page(pymupdf.Rect(DX, DY, w0 + DX, h0 + DY), doc, 0)

    base = DY + top + font.descender * size - 1.5
    for (x0, x1), txt in zip(cols, HEADERS):
        w = font.text_length(txt, size)
        page.insert_text(pymupdf.Point(DX + (x0 + x1) / 2 - w / 2, base), txt,
                         fontname="lib", fontfile=FONT, fontsize=size)
    x = DX + cols[0][0] - 2.0 + font.descender * size
    for txt, cy in rows:
        w = font.text_length(txt, size)
        page.insert_text(pymupdf.Point(x, DY + cy + w / 2), txt, fontname="lib",
                         fontfile=FONT, fontsize=size, rotate=90)
    new.subset_fonts()
    new.save(out, garbage=4, deflate=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
