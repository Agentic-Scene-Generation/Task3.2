from __future__ import annotations

SEATING = {
    "chair",
    "office_chair",
    "dining_chair",
    "stool",
    "armchair",
    "sofa",
    "loveseat",
    "bench",
}
WORK_SURFACES = {"desk", "table", "dining_table", "bar_table", "coffee_table"}
DIRECT_SEATING_WORK_SURFACES = {"desk", "table", "dining_table", "bar_table"}
OPTIONAL_SEATING_WORK_SURFACES = {
    "counter",
    "island",
    "console",
    "credenza",
    "sideboard",
    "buffet",
}
DIRECT_SEATING_SURFACE_MAX_GAP_M = 1.1
COFFEE_TABLE_SEATING_MAX_GAP_M = 0.85
LIVING_ROOM_COFFEE_TABLE_MAX_GAP_M = 1.35
OPTIONAL_SEATING_SURFACE_MAX_GAP_M = 0.45
OPTIONAL_SEATING_SURFACE_MAX_ANGLE_DEG = 120.0
STOOL_COUNTER_MAX_GAP_M = 0.8
MEDIA = {
    "display",
    "laptop",
    "monitor",
    "notebook_computer",
    "projection_screen",
    "screen",
    "tablet",
    "tablet_computer",
    "television",
    "tv",
}
BEDS = {"bed"}
NIGHTSTANDS = {"nightstand", "side_table"}
DINING_TABLES = {"dining_table", "table"}
SUPPORTS = {
    "cabinet",
    "counter",
    "desk",
    "drawer",
    "dresser",
    "table",
    "dining_table",
    "bar_table",
    "coffee_table",
    "island",
    "nightstand",
    "shelf",
    "wall_shelf",
    "bookshelf",
    "credenza",
    "sideboard",
    "console",
    "buffet",
    "media_console",
    "storage_furniture",
    "tv_stand",
    "wardrobe",
}
SUPPORTED_SMALL = {
    "alarm_clock",
    "bookend",
    "book",
    "bottle",
    "bowl",
    "carafe",
    "clock",
    "cup",
    "dish",
    "glass",
    "laptop",
    "keyboard",
    "magazine",
    "mug",
    "newspaper",
    "notebook_computer",
    "novel",
    "phone",
    "plant",
    "plate",
    "remote",
    "mouse",
    "smartphone",
    "soap",
    "soap_dispenser",
    "tablet",
    "tablet_computer",
    "pen_holder",
    "tray",
    "tumbler",
    "vase",
}
UPRIGHT_THIN_READING_MATERIALS = {"book", "magazine", "newspaper", "novel"}
LAMPS = {"desk_lamp", "table_lamp", "lamp"}
LAMP_SUBJECT_REJECT_HINTS = (
    "ceiling",
    "chandelier",
    "flush_mount",
    "floor_lamp",
    "pendant",
    "recessed_light",
    "track_light",
    "wall_light",
    "wall_sconce",
)
FLOOR_LAMP_TEXT_HINTS = ("floor_lamp", "standing_lamp", "standing lamp", "torchiere")
MOUNTED_LAMP_TEXT_HINTS = (
    "ceiling",
    "chandelier",
    "flush_mount",
    "pendant",
    "recessed_light",
    "track_light",
    "wall_light",
    "wall_sconce",
    "lightfixturewall",
)
SOFT_SUPPORT_TARGET_REJECT_HINTS = (
    "beanbag",
    "cushion",
    "floor_cushion",
    "meditation_pillow",
    "pillow",
    "seat_pad",
    "throw_pillow",
)
SECONDARY_SUPPORT_CATEGORIES = {"tray", "plate", "dish", "coaster"}
STACKABLE_SUPPORT_TEXT_HINTS = (
    "book",
    "bookend",
    "books",
    "hardcover",
    "magazine",
    "newspaper",
    "notebook",
    "novel",
    "paperback",
)
LIVING_ROOM_SEATING = {"armchair", "sofa", "loveseat"}
SEATING_SUBJECT_REJECT_HINTS = (
    "book",
    "notebook",
    "magazine",
)
SURFACE_TEXT_HINTS = ("desk", "table", "dining_table", "coffee_table", "bar_table")
SUPPORT_SURFACE_HINTS = (
    "desk",
    "table",
    "dining_table",
    "coffee_table",
    "bar_table",
    "nightstand",
    "shelf",
    "bookshelf",
    "credenza",
    "sideboard",
    "console",
    "buffet",
    "media_console",
    "tv_stand",
)
MEDIA_TEXT_HINTS = (
    "display",
    "laptop",
    "monitor",
    "notebook_computer",
    "projection_screen",
    "screen",
    "tablet_computer",
    "television",
    "tv",
)
WORK_SURFACE_TARGET_REJECT_HINTS = (
    "art",
    "artwork",
    "basket",
    "display_case",
    "displaycase",
    "painting",
    "picture",
    "poster",
)
MEDIA_TARGET_REJECT_HINTS = (
    "art",
    "artwork",
    "painting",
    "picture",
    "poster",
)
SIDE_SURFACE_HINTS = ("side_table", "sidetable", "end_table", "endtable", "nightstand")
SMALL_OBJECT_TEXT_HINTS = (
    "remote",
    "spoon",
    "tablespoon",
    "fork",
    "knife",
    "plate",
    "bowl",
    "coaster",
    "magazine",
    "book",
    "novel",
    "cup",
    "mug",
    "glass",
    "tumbler",
    "candle",
)
SUPPORT_SUBJECT_TEXT_HINTS = (
    "alarm_clock",
    "book",
    "bookend",
    "books",
    "bottle",
    "bowl",
    "carafe",
    "clock",
    "cup",
    "dish",
    "glass",
    "keyboard",
    "laptop",
    "magazine",
    "mouse",
    "mug",
    "newspaper",
    "novel",
    "phone",
    "plant",
    "plate",
    "remote",
    "smartphone",
    "soap",
    "soap_dispenser",
    "tablet",
    "tablet_computer",
    "pen_holder",
    "tray",
    "tumbler",
    "vase",
)
SUPPORT_SUBJECT_REJECT_HINTS = (
    "candle",
    "coaster",
    "fork",
    "knife",
    "light",
    "spoon",
    "eucalyptus",
)
PROPOSER_SUBJECT_MULTIPLIER = 2
PROPOSER_TARGET_MULTIPLIER = 3
PROPOSER_MAX_TARGETS_PER_RELATION = 4
PROPOSER_MAX_TASK_CHARS = 240
CORE_SUPPORTED_SMALL = {
    "keyboard",
    "laptop",
    "monitor",
    "mouse",
    "notebook_computer",
    "tablet",
    "tablet_computer",
}
DECORATIVE_SUPPORT_SMALL = {
    "alarm_clock",
    "book",
    "bookend",
    "bottle",
    "bowl",
    "carafe",
    "clock",
    "cup",
    "dish",
    "glass",
    "mug",
    "phone",
    "plate",
    "smartphone",
    "tray",
    "tumbler",
    "vase",
}
SUPPORT_TOP_SURFACE_AFFORDANCES = {
    "supportable",
    "support_surface",
    "support",
    "place_on_top",
}
SUPPORT_INTERNAL_AFFORDANCES = {"storage", "storage_shelf", "containable", "openable"}
SUPPORT_CATEGORY_GROUPS = {
    "work_surface",
    "storage",
    "storage_surface",
    "appliance_storage",
}
WORK_SURFACE_CATEGORY_GROUPS = {"work_surface", "storage_surface"}
WORK_SURFACE_REJECT_GROUPS = {"lighting", "seating", "small_object", "decor"}
WORK_SURFACE_PREFIX_FAMILIES = (
    "console",
    "credenza",
    "sideboard",
    "buffet",
    "nightstand",
    "side table",
    "end table",
    "counter",
    "island",
    "coffee table",
    "dining table",
    "bar table",
)
TOP_SURFACE_TERMS = (
    "support",
    "place",
    "place objects",
    "top panel",
    "top surface",
    "entire_top_surface",
    "top_edge",
)
SHELF_SURFACE_TERMS = (
    "lower shelf",
    "shelf",
    "shelves",
    "shelf tiers",
    "shelf surfaces",
    "open shelf",
    "open compartment",
    "compartment",
    "bay",
)
STORAGE_SURFACE_TERMS = (
    "drawer",
    "cabinet",
    "storage",
    "interior",
    "compartment",
    "bay",
)
