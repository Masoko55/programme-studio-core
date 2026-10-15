"""50 unseen briefs. HOLDOUT CASES MUST NOT BE USED AS EVENT-SPECIFIC IMPLEMENTATION TARGETS.

The two treatments per base brief test semantic transport, not renderer tuning.
"""

BASE = (
    ("marine research exhibition", "bioluminescent sea laboratory", "curious serene", "research visitors", "floating luminous coral clusters", "navy", "teal", "harsh"),
    ("railway heritage gathering", "vintage railway station", "nostalgic intimate", "heritage visitors", "receding tracks and iron arches", "brown", "cream", "futuristic"),
    ("lunar horticulture showcase", "moon garden habitat", "ethereal restorative", "garden visitors", "nested planting beds and transparent domes", "silver", "green", "chaotic"),
    ("ceramic craft market", "handmade pottery stalls", "warm tactile", "craft lovers", "repeated clay vessel silhouettes", "terracotta", "cream", "sterile"),
    ("desert astronomy evening", "star-filled desert horizon", "vast contemplative", "stargazers", "layered dunes and distributed constellations", "navy", "gold", "cluttered"),
    ("arctic architecture symposium", "icebound civic structures", "crisp thoughtful", "design professionals", "stepped ice geometry", "white", "blue", "tropical"),
    ("urban beekeeping festival", "city rooftop hives", "lively earthy", "families", "hexagonal hive rhythm", "gold", "brown", "ominous"),
    ("kinetic sculpture exhibition", "moving sculptural forms", "kinetic airy", "art visitors", "suspended arcs and counterweights", "silver", "orange", "static"),
    ("traditional weaving market", "woven textile displays", "intimate textured", "market visitors", "interlaced textile bands", "red", "cream", "plastic"),
    ("experimental jazz listening evening", "abstract sound stage", "immersive rhythmic", "music listeners", "layered syncopated waveforms", "purple", "gold", "silent"),
    ("children's storytelling day", "illustrated library landscape", "playful gentle", "children", "open books and flowing paper shapes", "blue", "yellow", "frightening"),
    ("formal civic ceremony", "civic hall architecture", "dignified calm", "adult guests", "symmetrical stone columns", "black", "white", "chaotic"),
    ("community remembrance", "quiet riverside memorial", "solemn peaceful", "families", "layered riverbank stones", "grey", "white", "neon party"),
    ("school science fair", "laboratory display forms", "curious bright", "students", "connected apparatus silhouettes", "blue", "orange", "ominous"),
    ("trail running celebration", "mountain trail switchbacks", "energetic fresh", "athletes", "receding trail contours", "green", "orange", "stagnant"),
    ("regional food harvest", "orchard produce displays", "abundant warm", "food lovers", "clustered fruit and woven baskets", "green", "red", "sterile"),
    ("migration heritage exhibition", "historic travel maps", "reflective weathered", "museum visitors", "layered route lines and archival textures", "brown", "blue", "futuristic"),
    ("coastal ecology seminar", "tidal research shore", "clean restorative", "scientists", "tidal bands and sea grass", "teal", "cream", "polluted"),
    ("robotics demonstration", "modular robotic workspace", "innovative precise", "engineers", "jointed mechanisms and wiring paths", "silver", "blue", "rustic"),
    ("forest restoration gathering", "regenerating woodland", "hopeful earthy", "volunteers", "layered saplings and canopy forms", "green", "brown", "barren"),
    ("abstract light installation", "prismatic chamber", "luminous meditative", "art visitors", "overlapping refracted planes", "purple", "silver", "dark"),
    ("fantasy cartography exhibit", "imagined mountain maps", "mysterious adventurous", "readers", "contour paths and island silhouettes", "blue", "gold", "mundane"),
    ("transport futures forum", "intermodal transit hub", "professional forward-looking", "planners", "converging transit lines", "navy", "silver", "antique"),
    ("artisan printmaking workshop", "inked printmaking studio", "tactile focused", "artists", "repeated ink impressions", "black", "cream", "digital-gloss"),
    ("neighborhood garden exchange", "shared courtyard garden", "welcoming peaceful", "neighbors", "interwoven garden paths", "green", "pink", "ominous"),
)

CASES = tuple(
    (event, theme, f"{mood} {treatment}", audience, motif, primary, secondary, incompatible)
    for event, theme, mood, audience, motif, primary, secondary, incompatible in BASE
    for treatment in ("layered tactile", "airy spacious")
)
