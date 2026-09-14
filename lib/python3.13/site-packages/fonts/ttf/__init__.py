from fonts import _load_fonts

font_files = _load_fonts("fonts_ttf")

for font in font_files:
    globals()[font] = font_files[font]
