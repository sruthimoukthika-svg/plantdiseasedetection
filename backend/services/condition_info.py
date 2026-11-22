"""Reference information for each condition the trained model can predict.

Every entry corresponds to a label verified present in the datasets (see
training/reports/dataset_report.md). There are no entries for conditions
the model cannot predict, and no condition is described that does not
appear in the class mapping - a test asserts the two sets match exactly.

Descriptions are general agronomic background, not diagnosis. Each carries
the same caveat: an image-based prediction needs confirmation by someone
qualified, and often lab testing, because several of these conditions are
visually similar in early stages.
"""
from __future__ import annotations

CONFIRMATION_NOTE = (
    "Image-based predictions require confirmation. Several of these "
    "conditions look alike in early stages, and look-alike symptoms can "
    "also be caused by watering, heat, spray damage or soil problems."
)

INFO: dict[tuple[str, str], dict] = {
    # ---------------------------------------------------------------- tomato
    ("tomato", "bacterial_spot"): {
        "summary": "A bacterial disease causing small, dark, water-soaked spots on leaves and fruit.",
        "symptoms": ["Small dark brown to black spots, often with a yellow halo",
                     "Spots may merge, causing leaves to yellow and drop",
                     "Raised scabby lesions on fruit"],
        "spreads": "Splashing water, handling wet plants, infected seed and transplants.",
    },
    ("tomato", "early_blight"): {
        "summary": "A fungal disease that usually starts on older, lower leaves.",
        "symptoms": ["Brown spots with concentric rings, a target-like pattern",
                     "Yellowing tissue surrounding each spot",
                     "Progressive loss of lower leaves"],
        "spreads": "Soil and crop debris; favoured by warm, humid, wet weather.",
    },
    ("tomato", "late_blight"): {
        "summary": "An aggressive oomycete disease that can destroy a crop within days in cool, wet conditions.",
        "symptoms": ["Large grey-green to brown water-soaked patches",
                     "White fuzzy growth on leaf undersides in humid weather",
                     "Firm brown blotches on fruit"],
        "spreads": "Wind-borne spores over long distances; cool wet weather.",
    },
    ("tomato", "leaf_mold"): {
        "summary": "A fungal disease largely confined to greenhouses and other humid, enclosed growing areas.",
        "symptoms": ["Pale green to yellow patches on upper leaf surfaces",
                     "Olive-green to brown velvety mould beneath",
                     "Leaves wither but often stay attached"],
        "spreads": "High humidity and poor air movement.",
    },
    ("tomato", "septoria_leaf_spot"): {
        "summary": "A fungal disease producing many small, uniformly sized leaf spots.",
        "symptoms": ["Numerous small circular spots with dark borders and pale centres",
                     "Tiny dark fruiting bodies visible in spot centres",
                     "Starts low on the plant and moves upward"],
        "spreads": "Rain splash and crop debris.",
    },
    ("tomato", "target_spot"): {
        "summary": "A fungal disease affecting leaves, stems and fruit.",
        "symptoms": ["Small brown spots enlarging into target-like rings",
                     "Lesions on stems and petioles",
                     "Sunken circular lesions on fruit"],
        "spreads": "Warm humid conditions and infected debris.",
    },
    ("tomato", "yellow_leaf_curl_virus"): {
        "summary": "A virus transmitted by whitefly, causing severe stunting and yield loss.",
        "symptoms": ["Upward curling and cupping of leaves",
                     "Yellowing of leaf margins, reduced leaf size",
                     "Stunted growth, poor fruit set"],
        "spreads": "Whitefly vector. Not transmitted by touch or seed.",
    },
    ("tomato", "mosaic_virus"): {
        "summary": "A highly persistent virus producing mottled foliage.",
        "symptoms": ["Light and dark green mottling of leaves",
                     "Distorted, narrow or fern-like leaves",
                     "Internal browning of fruit in some cases"],
        "spreads": "Mechanical contact, tools, hands and infected seed. Very stable.",
    },
    ("tomato", "spotted_wilt_virus"): {
        "summary": "A thrips-transmitted virus with variable, often ring-shaped symptoms.",
        "symptoms": ["Bronzing or dark ring spots on leaves",
                     "Dark streaks on stems and petioles",
                     "Concentric rings or blotches on fruit"],
        "spreads": "Thrips vector.",
    },
    ("tomato", "spider_mites"): {
        "summary": "Pest damage from two-spotted spider mites, not an infectious disease.",
        "symptoms": ["Fine pale stippling or speckling on leaves",
                     "Bronzed or dusty appearance as damage builds",
                     "Fine webbing on leaf undersides"],
        "spreads": "Mites spread by wind, movement and contact; favoured by hot dry conditions.",
    },
    ("tomato", "leaf_miner"): {
        "summary": "Pest damage from leaf-mining larvae feeding inside the leaf, not an infectious disease.",
        "symptoms": ["Winding pale tunnels or mines within the leaf",
                     "Blotchy translucent patches",
                     "Premature leaf drop in heavy infestations"],
        "spreads": "Adult flies lay eggs in leaf tissue.",
    },
    ("tomato", "magnesium_deficiency"): {
        "summary": "A nutrient disorder, not a disease and not contagious.",
        "symptoms": ["Yellowing between leaf veins while veins stay green",
                     "Usually begins on older, lower leaves",
                     "Leaf margins may curl upward"],
        "spreads": "Not transmissible. Related to soil magnesium, pH and nutrient balance.",
    },
    ("tomato", "nitrogen_deficiency"): {
        "summary": "A nutrient disorder, not a disease and not contagious.",
        "symptoms": ["General pale green to yellow colour over the whole leaf",
                     "Oldest leaves affected first",
                     "Reduced growth and small leaves"],
        "spreads": "Not transmissible. Related to soil nitrogen availability.",
    },
    ("tomato", "potassium_deficiency"): {
        "summary": "A nutrient disorder, not a disease and not contagious.",
        "symptoms": ["Yellowing and scorching of leaf margins",
                     "Browning edges that progress inward",
                     "Uneven fruit ripening"],
        "spreads": "Not transmissible. Related to soil potassium availability.",
    },
    ("tomato", "healthy"): {
        "summary": "No condition detected in this image.",
        "symptoms": ["Uniform colour", "No lesions, mottling or pest damage visible"],
        "spreads": "",
    },
    # ---------------------------------------------------------------- potato
    ("potato", "early_blight"): {
        "summary": "A fungal disease of potato foliage, typically on older leaves first.",
        "symptoms": ["Dark brown spots with concentric rings",
                     "Yellow tissue around lesions",
                     "Lower leaves affected first"],
        "spreads": "Soil and crop debris; warm humid weather.",
    },
    ("potato", "late_blight"): {
        "summary": "A destructive oomycete disease, the cause of historic potato famines.",
        "symptoms": ["Dark water-soaked patches spreading rapidly",
                     "Pale fuzzy growth at lesion edges in humid weather",
                     "Firm reddish-brown rot in tubers"],
        "spreads": "Wind-borne spores; cool wet weather. Can destroy a crop in days.",
    },
    ("potato", "healthy"): {
        "summary": "No condition detected in this image.",
        "symptoms": ["Uniform colour", "No lesions or pest damage visible"],
        "spreads": "",
    },
    # ----------------------------------------------------------------- grape
    ("grape", "black_rot"): {
        "summary": "A fungal disease affecting leaves, shoots and fruit.",
        "symptoms": ["Small tan to brown leaf spots with dark borders",
                     "Tiny black fruiting bodies within spots",
                     "Berries shrivel into hard black mummies"],
        "spreads": "Overwintered mummies and canes; warm wet spring weather.",
    },
    ("grape", "esca"): {
        "summary": "A trunk disease complex, also called black measles, caused by wood-colonising fungi.",
        "symptoms": ["Interveinal yellowing or reddening in a tiger-stripe pattern",
                     "Dark spotting on berries",
                     "Sudden collapse of shoots in severe cases"],
        "spreads": "Infection through pruning wounds. Chronic, develops over years.",
    },
    ("grape", "leaf_blight"): {
        "summary": "A fungal leaf-spot disease, also known as Isariopsis leaf spot.",
        "symptoms": ["Irregular dark brown to black angular leaf lesions",
                     "Lesions may merge across the leaf",
                     "Premature leaf drop reducing fruit quality"],
        "spreads": "Warm humid weather and crop debris.",
    },
    ("grape", "healthy"): {
        "summary": "No condition detected in this image.",
        "symptoms": ["Uniform colour", "No lesions or pest damage visible"],
        "spreads": "",
    },
}


def get(crop: str, condition: str) -> dict:
    base = INFO.get((crop, condition))
    if base is None:
        return {"summary": "", "symptoms": [], "spreads": "",
                "confirmation_note": CONFIRMATION_NOTE}
    return {**base, "confirmation_note": CONFIRMATION_NOTE}
