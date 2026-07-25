"""
Builds a synthetic PDF that reproduces the hard parts of an Arabic teacher's
guide spread:

  * two columns, RTL reading order (teacher notes right, pupil page left)
  * numbered activity chips that appear in BOTH columns and must be linked
  * a tinted panel containing a reading passage
  * a placed photograph
  * Arabic written as SHAPED PRESENTATION FORMS IN VISUAL ORDER -- which is
    what a large share of InDesign/ME exports actually embed, and the exact
    thing that makes naive extraction return mojibake

If the parser recovers logical order from this, it will recover it from the
real book.
"""
import fitz
import arabic_reshaper
from bidi.algorithm import get_display

FONT = "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"
BOLD = "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf"

MINT = (0.85, 0.93, 0.90)
CHIP = (0.13, 0.30, 0.55)
NAVY = (0.10, 0.20, 0.45)


def visual(text: str) -> str:
    """Shape + reverse: emulate a visual-order Arabic content stream."""
    return get_display(arabic_reshaper.reshape(text))


def build(path="fixture_page88.pdf"):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_font(fontname="ar", fontfile=FONT)
    page.insert_font(fontname="arb", fontfile=BOLD)

    def rtl(x_right, y, text, size=9, font="ar", color=(0, 0, 0)):
        s = visual(text)
        w = fitz.get_text_length(s, fontname=font, fontsize=size) if False else size * 0.5 * len(s)
        page.insert_text((x_right - w, y), s, fontname=font, fontsize=size, color=color)

    def chip(x, y, n, size=13):
        page.draw_rect(fitz.Rect(x, y, x + size, y + size), color=None, fill=CHIP)
        page.insert_text((x + 4, y + 10), str(n), fontname="helv", fontsize=8, color=(1, 1, 1))

    # ---- RIGHT column: teacher notes (logical column 0) -------------------
    chip(540, 90, 6)
    rtl(535, 112, "(10 دقيقة)", 8)
    rtl(535, 132, "الهدف: أن يتعرّف الطباق ودوره في النص.", 8.5, "arb")
    rtl(535, 148, "يتوزّع التلاميذ بشكل ثنائيّ، يتشاركان الأفكار", 8.5)
    rtl(535, 164, "للوصول إلى الإجابة ومن ثمّ يدوّنانها على", 8.5)
    rtl(535, 180, "بطاقات كرتونيّة، ويرفعانها عند الإجابة.", 8.5)
    rtl(535, 205, "إجابات", 9, "arb")
    rtl(535, 221, "1. الجنوب ≠ الشمال", 8.5)
    rtl(535, 237, "2. الأرض القاحلة ≠ مراع خصبة", 8.5)

    chip(540, 275, 7)
    rtl(535, 297, "(5 دقائق)", 8)
    rtl(535, 317, "الهدف: أن يتمكّن من القراءة الصحيحة", 8.5, "arb")
    rtl(535, 333, "وصولًا إلى الطلاقة التي تساعد على الفهم.", 8.5)
    rtl(535, 349, "وزّع جمل النص بين التلاميذ.", 8.5)

    # ---- LEFT column: pupil page facsimile (logical column 1) -------------
    page.draw_rect(fitz.Rect(60, 60, 340, 700), color=(0.85, 0.85, 0.85), width=0.5)

    chip(300, 90, 6)
    rtl(292, 100, "1. في العبارة الآتية طباق، أشِر إليه.", 8.5, "arb")
    rtl(292, 118, "«كائنٌ منقسمة إلى مملكتين اثنتين: واحدة في الجنوب، وأخرى في الشمال».", 7.5,
        "ar", (0.55, 0.25, 0.55))
    rtl(292, 140, "2. هل تجد طباقًا آخر في النص؟ سجّله في دفترك.", 8.5, "arb")

    chip(300, 175, 7)
    rtl(292, 185, "إقرأ النص، مع زملائك، قراءةً جهريةً مداورة، ثم أجب عن الأسئلة.", 8.5, "arb")

    # tinted passage panel
    page.draw_rect(fitz.Rect(75, 205, 330, 330), color=None, fill=MINT)
    rtl(318, 228, "موكبُ المومياوات", 12, "arb", NAVY)
    rtl(318, 252, "في الثالث من نيسان العام 2021 شهدت مصر حدثًا استثنائيًا أدهش العالم،", 7.5)
    rtl(318, 268, "تمّ نقل تفاصيله على الهواء مباشرة، وقد شارك في تقديمه نجوم الفن المصري.", 7.5)
    rtl(318, 284, "بدأ الحدث باختراق موكب ملكي فرعوني مؤلف من 22 مومياء شوارع القاهرة.", 7.5)
    rtl(318, 300, "انطلق موكب المومياوات الملكية من المتحف المصري القديم في وسط العاصمة.", 7.5)

    # placed photograph
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 400, 240))
    pix.set_rect(pix.irect, (28, 32, 60))
    page.insert_image(fitz.Rect(75, 345, 330, 500), pixmap=pix)

    page.insert_text((196, 690), "88", fontname="helv", fontsize=8)
    doc.save(path)
    doc.close()
    return path


if __name__ == "__main__":
    print(build())
