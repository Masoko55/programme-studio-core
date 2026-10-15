"""Deterministic cross-product samples; deliberately independent of production code."""

import random


def cases(count: int = 40) -> tuple:
    rng = random.Random(20261015)
    subjects = ("mineral landscape", "robotic orchard", "desert sound installation",
                "ceramic coastline", "floating observatory", "industrial garden")
    atmospheres = ("quietly luminous meditative", "nostalgic airy", "kinetic ceremonial",
                   "weathered intimate", "restorative rhythmic", "ethereal delicate")
    motifs = ("nested shell contours", "repeating rail arcs", "layered translucent discs",
              "asymmetric root patterns", "staggered ceramic forms", "flowing light bands")
    audiences = ("curious families", "adult visitors", "young learners", "design professionals")
    palettes = (("navy", "silver"), ("black", "white"), ("terracotta", "cream"),
                ("green", "gold"), ("purple", "blue"))
    result = []
    for index in range(count):
        primary, secondary = rng.choice(palettes)
        result.append((f"unfamiliar gathering {index}", rng.choice(subjects),
                       rng.choice(atmospheres), rng.choice(audiences), rng.choice(motifs),
                       primary, secondary, "harsh clutter"))
    return tuple(result)
