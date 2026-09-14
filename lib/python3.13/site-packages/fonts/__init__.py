from importlib.metadata import entry_points


def _load_fonts(group):
    """Return a {name: path} dict for all fonts registered under an entry point group."""
    eps = entry_points()
    # Python 3.10+ exposes .select(); 3.9 returns a plain dict keyed by group.
    selected = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
    fonts = {}
    for entry_point in selected:
        fonts.update(entry_point.load())
    return fonts
