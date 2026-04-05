"""Canonical taxonomies and entity resolution for aviation safety domain.

Implements taxonomy-backed entity normalization following Agarwal et al.
(LREC 2022, arXiv:2205.15952). Resolves raw LLM-extracted entity strings
to canonical forms using exact match, fuzzy match (rapidfuzz), and
embedding similarity (sentence-transformers).

Taxonomies:
  - HFACS (Human Factors Analysis and Classification System)
  - ICAO standard phases of flight
  - Aircraft type designators (ICAO codes)
  - Weather condition categories
  - Airport ICAO codes (top 100 US + 50 international)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

DATA_DIR = Path(os.environ.get("AEROGRAPH_DATA_DIR", Path(__file__).parent.parent.parent / "data"))
PROCESSED_DIR = DATA_DIR / "processed"

# ---------------------------------------------------------------------------
# HFACS Taxonomy (Human Factors Analysis and Classification System)
# Wiegmann & Shappell (2003) — used by FAA, NTSB, Delta Safety
# ---------------------------------------------------------------------------

HFACS_TAXONOMY: dict[str, dict] = {
    # Level 1: Unsafe Acts > Errors
    "decision_error": {
        "level": 2, "parent": "errors",
        "l1": "unsafe_acts",
        "aliases": [
            "poor decision", "bad decision", "decision error",
            "poor judgment", "poor decision making", "wrong decision",
            "decision-making error", "improper decision",
        ],
    },
    "skill_based_error": {
        "level": 2, "parent": "errors",
        "l1": "unsafe_acts",
        "aliases": [
            "skill error", "skill-based error", "technique error",
            "stick and rudder error", "procedural error", "manual handling error",
            "inadvertent action", "control error",
        ],
    },
    "perceptual_error": {
        "level": 2, "parent": "errors",
        "l1": "unsafe_acts",
        "aliases": [
            "perceptual error", "spatial disorientation", "visual illusion",
            "misperception", "misjudged distance", "misjudged altitude",
            "depth perception error", "visual misperception",
        ],
    },
    # Level 1: Unsafe Acts > Violations
    "routine_violation": {
        "level": 2, "parent": "violations",
        "l1": "unsafe_acts",
        "aliases": [
            "routine violation", "habitual violation", "shortcut",
            "procedural deviation", "sop deviation", "checklist skip",
            "procedure_non_compliance", "non-compliance",
        ],
    },
    "exceptional_violation": {
        "level": 2, "parent": "violations",
        "l1": "unsafe_acts",
        "aliases": [
            "exceptional violation", "willful violation", "reckless behavior",
            "intentional deviation", "deliberate non-compliance",
        ],
    },
    # Level 1: Preconditions > Adverse Mental States
    "adverse_mental_state": {
        "level": 2, "parent": "preconditions_adverse_mental",
        "l1": "preconditions_for_unsafe_acts",
        "aliases": [
            "complacency", "distraction", "task saturation",
            "task_saturation", "channelized attention", "mental fatigue",
            "loss of situational awareness", "situational awareness",
            "lack of awareness", "inattention", "loss of awareness",
            "fixation", "cognitive overload", "tunnel vision",
            "stress", "overconfidence",
        ],
    },
    "adverse_physiological_state": {
        "level": 2, "parent": "preconditions_adverse_physiological",
        "l1": "preconditions_for_unsafe_acts",
        "aliases": [
            "fatigue", "crew fatigue", "pilot fatigue", "crew_fatigue",
            "sleep deprivation", "illness", "medical condition",
            "physical fatigue", "exhaustion", "drowsiness",
            "fatigue - loss of sleep", "hypoxia", "dehydration",
        ],
    },
    "physical_mental_limitation": {
        "level": 2, "parent": "preconditions_limitations",
        "l1": "preconditions_for_unsafe_acts",
        "aliases": [
            "physical limitation", "mental limitation",
            "inexperience", "lack of experience", "lack of training",
            "unfamiliar with aircraft", "insufficient training",
            "lack of proficiency", "inadequate training",
        ],
    },
    "crew_resource_management": {
        "level": 2, "parent": "preconditions_crm",
        "l1": "preconditions_for_unsafe_acts",
        "aliases": [
            "crm", "crew resource management", "poor crm",
            "communication breakdown", "communication_breakdown",
            "communication failure", "communication_failure",
            "miscommunication", "poor communication",
            "lack of coordination", "lack_of_coordination",
            "coordination failure", "crew coordination",
            "poor crew coordination", "lack_of_communication",
            "assertiveness", "lack of assertiveness",
        ],
    },
    "personal_readiness": {
        "level": 2, "parent": "preconditions_personal",
        "l1": "preconditions_for_unsafe_acts",
        "aliases": [
            "personal readiness", "inadequate rest",
            "self-medication", "alcohol", "violation of crew rest",
            "crew rest violation", "fitness for duty",
        ],
    },
    "physical_environment": {
        "level": 2, "parent": "preconditions_environment",
        "l1": "preconditions_for_unsafe_acts",
        "aliases": [
            "physical environment", "weather hazard",
            "altitude environment", "lighting condition",
            "terrain environment",
        ],
    },
    "technological_environment": {
        "level": 2, "parent": "preconditions_environment",
        "l1": "preconditions_for_unsafe_acts",
        "aliases": [
            "technological environment", "equipment design",
            "interface design", "automation design",
            "cockpit design", "display design",
        ],
    },
    # Level 1: Unsafe Supervision
    "inadequate_supervision": {
        "level": 2, "parent": "unsafe_supervision",
        "l1": "unsafe_supervision",
        "aliases": [
            "inadequate supervision", "poor supervision",
            "lack of oversight", "insufficient oversight",
            "failure to provide guidance", "lack of supervision",
        ],
    },
    "planned_inappropriate_operations": {
        "level": 2, "parent": "unsafe_supervision",
        "l1": "unsafe_supervision",
        "aliases": [
            "planned inappropriate operations", "poor planning",
            "inadequate risk assessment", "mission planning error",
            "scheduling error", "crew pairing error",
        ],
    },
    "failed_to_correct_problem": {
        "level": 2, "parent": "unsafe_supervision",
        "l1": "unsafe_supervision",
        "aliases": [
            "failed to correct", "failure to correct known problem",
            "known deficiency not corrected", "ignored known problem",
            "failed to address", "overlooked deficiency",
        ],
    },
    "supervisory_violation": {
        "level": 2, "parent": "unsafe_supervision",
        "l1": "unsafe_supervision",
        "aliases": [
            "supervisory violation", "supervisory misconduct",
            "authorized unnecessary hazard",
        ],
    },
    # Level 1: Organizational Influences
    "resource_management": {
        "level": 2, "parent": "organizational_influences",
        "l1": "organizational_influences",
        "aliases": [
            "resource management", "staffing issues", "understaffing",
            "insufficient resources", "equipment shortage",
            "budget constraints", "manpower shortage",
            "high traffic volume", "high_traffic_volume",
        ],
    },
    "organizational_climate": {
        "level": 2, "parent": "organizational_influences",
        "l1": "organizational_influences",
        "aliases": [
            "organizational climate", "safety culture",
            "corporate pressure", "management pressure",
            "organizational pressure", "company culture",
        ],
    },
    "organizational_process": {
        "level": 2, "parent": "organizational_influences",
        "l1": "organizational_influences",
        "aliases": [
            "organizational process", "procedural gap",
            "standard operating procedure", "sop inadequacy",
            "training program deficiency", "policy gap",
        ],
    },
}

# HFACS Level 1 categories for validation
HFACS_LEVEL1 = [
    "unsafe_acts",
    "preconditions_for_unsafe_acts",
    "unsafe_supervision",
    "organizational_influences",
]

# ---------------------------------------------------------------------------
# Phase of Flight Taxonomy (ICAO standard)
# ---------------------------------------------------------------------------

PHASE_OF_FLIGHT: dict[str, list[str]] = {
    "taxi": [
        "taxi", "taxiing", "taxi phase", "taxi_phase", "ground taxi",
        "taxi out", "taxi in", "taxi_out", "taxi_in", "pushback",
        "single_engine_taxi", "single engine taxi",
    ],
    "takeoff": [
        "takeoff", "take-off", "takeoff phase", "takeoff_phase",
        "takeoff roll", "takeoff_roll", "rotation", "v1", "departure roll",
        "rejected takeoff", "aborted takeoff",
    ],
    "initial_climb": [
        "initial climb", "initial_climb", "departure climb",
        "climb out", "climbout", "after takeoff",
    ],
    "climb": [
        "climb", "climb phase", "climb_phase", "climbing",
        "enroute climb", "enroute_climb",
    ],
    "cruise": [
        "cruise", "cruise_flight", "cruise flight", "cruise phase",
        "enroute", "en route", "level flight", "level_flight",
    ],
    "descent": [
        "descent", "descent_phase", "descent phase", "descending",
        "top of descent", "tod", "enroute descent",
    ],
    "initial_approach": [
        "initial approach", "initial_approach", "approach",
        "approach phase", "approach_phase", "arrival",
        "instrument approach", "ils approach", "ils_approach",
        "rnav approach", "rnav_approach", "vor approach",
    ],
    "final_approach": [
        "final approach", "final_approach", "final",
        "short final", "on final", "visual approach",
        "visual_approach", "base to final turn", "base to final",
    ],
    "landing": [
        "landing", "landing phase", "landing_phase",
        "touchdown", "landing roll", "landing_roll",
        "flare", "rollout",
    ],
    "go_around": [
        "go-around", "go around", "go_around", "missed approach",
        "missed_approach", "balked landing", "overshoot",
    ],
    "parking": [
        "parking", "parking/standing", "gate", "ramp",
        "gate arrival", "gate_arrival", "ground_operations",
        "ground operations", "parked",
    ],
    "preflight": [
        "preflight", "pre-flight", "preflight_duties",
        "preflight duties", "before start",
    ],
}

# ---------------------------------------------------------------------------
# Aircraft Type Normalization (ICAO type designators)
# Built from actual ASRS data — top 30+ types by frequency
# ---------------------------------------------------------------------------

AIRCRAFT_TYPES: dict[str, list[str]] = {
    "B737": [
        "b737", "boeing 737", "737", "b737-700", "b737-900",
        "b737ng", "b737 max", "737-700", "737-900",
        "boeing 737-700", "boeing 737-900",
    ],
    "B738": [
        "b738", "737-800", "boeing 737-800", "737-800w",
        "boeing 737-800w", "b737-800",
    ],
    "B757": [
        "b757", "boeing 757", "757", "b757-200", "b757-300",
        "757-200", "757-300", "boeing 757-200",
    ],
    "B767": [
        "b767", "boeing 767", "767", "b767-300", "b767-400",
        "767-300", "boeing 767-300",
    ],
    "B777": [
        "b777", "boeing 777", "777", "b777-200", "b777-300",
        "777-200", "777-300", "boeing 777-200",
    ],
    "B747": [
        "b747", "boeing 747", "747", "b747-8",
    ],
    "B744": [
        "b744", "747-400", "boeing 747-400", "b747-400", "747-400f",
    ],
    "B787": [
        "b787", "boeing 787", "787", "b787-8", "b787-9",
        "787-8", "787-9", "dreamliner", "boeing 787-9",
    ],
    "B38M": [
        "b38m", "737 max 8", "737max8", "boeing 737 max 8",
        "737-8 max", "737-8200",
    ],
    "A319": [
        "a319", "airbus a319", "a319-100", "airbus a319-100",
    ],
    "A320": [
        "a320", "airbus a320", "a320-200", "airbus a320-200",
        "a320neo", "a320-neo",
    ],
    "A321": [
        "a321", "airbus a321", "a321-200", "a321neo",
        "airbus a321-200", "a321-neo",
    ],
    "A330": [
        "a330", "airbus a330", "a330-200", "a330-300",
        "airbus a330-200", "airbus a330-300",
    ],
    "A340": [
        "a340", "airbus a340", "a340-300", "a340-600",
    ],
    "A350": [
        "a350", "airbus a350", "a350-900", "a350-1000",
    ],
    "A300": [
        "a300", "airbus a300", "a300-600", "a300f",
    ],
    "CRJ2": [
        "crj2", "crj-200", "crj200", "crj 200",
        "canadair regional jet 200", "bombardier crj-200",
    ],
    "CRJ7": [
        "crj7", "crj-700", "crj700", "crj 700",
        "canadair regional jet 700", "bombardier crj-700",
    ],
    "CRJ9": [
        "crj9", "crj-900", "crj900", "crj 900",
        "canadair regional jet 900", "bombardier crj-900",
    ],
    "E145": [
        "e145", "emb-145", "emb145", "embraer 145",
        "erj-145", "erj145", "embraer erj-145",
    ],
    "E170": [
        "e170", "emb-170", "emb170", "embraer 170",
        "erj-170", "embraer e170",
    ],
    "E75S": [
        "e75s", "e175", "emb-175", "emb175", "embraer 175",
        "erj-175", "embraer e175", "e-175",
    ],
    "E190": [
        "e190", "emb-190", "emb190", "embraer 190",
        "erj-190", "embraer e190",
    ],
    "MD80": [
        "md80", "md-80", "md80", "md-82", "md-83",
        "md82", "md83", "mcdonnell douglas md-80",
    ],
    "MD11": [
        "md11", "md-11", "mcdonnell douglas md-11",
    ],
    "DC10": [
        "dc10", "dc-10", "mcdonnell douglas dc-10",
    ],
    "C172": [
        "c172", "cessna 172", "c172p", "c172s",
        "cessna 172s", "cessna skyhawk", "skyhawk",
    ],
    "C182": [
        "c182", "cessna 182", "c182p", "c182t",
        "cessna 182t", "cessna skylane", "skylane",
    ],
    "C210": [
        "c210", "cessna 210", "centurion", "cessna centurion",
    ],
    "PA28": [
        "pa28", "pa-28", "piper cherokee", "cherokee",
        "piper warrior", "warrior", "piper archer", "archer",
    ],
    "PA32": [
        "pa32", "pa-32", "piper saratoga", "saratoga",
        "piper lance", "lance",
    ],
    "SR22": [
        "sr22", "cirrus sr22", "cirrus sr-22",
    ],
    "SR20": [
        "sr20", "cirrus sr20", "cirrus sr-20",
    ],
    "BE20": [
        "be20", "beech king air", "king air", "king air 200",
        "beechcraft king air", "be200",
    ],
    "Q400": [
        "q400", "dash 8", "dash-8", "dash 8-400", "dh8d",
        "bombardier q400", "de havilland dash 8",
    ],
    "ATR72": [
        "atr72", "atr 72", "atr-72",
    ],
    "SF34": [
        "sf34", "saab 340", "saab-340", "saab 340b",
    ],
}

# Anonymized aircraft patterns — keep these as-is, don't normalize
ANONYMIZED_AIRCRAFT = {
    "aircraft", "aircraft x", "aircraft_x", "aircraft y", "aircraft_y",
    "aircraft z", "aircraft_z", "air carrier aircraft", "reporting aircraft",
    "other aircraft", "other_aircraft", "light aircraft", "vfr aircraft",
    "helicopter", "drone", "cessna", "unmanned aerial drone",
}

# ---------------------------------------------------------------------------
# Weather Condition Taxonomy
# ---------------------------------------------------------------------------

WEATHER_CONDITIONS: dict[str, list[str]] = {
    "VMC": [
        "vmc", "vmc_conditions", "vmc conditions", "visual meteorological conditions",
        "visual conditions", "vfr_conditions", "vfr conditions",
        "clear weather", "clear skies", "clear",
    ],
    "IMC": [
        "imc", "imc_conditions", "imc conditions",
        "instrument meteorological conditions", "instrument conditions",
        "low_imc_midwest", "ifr conditions", "ifr_conditions",
    ],
    "thunderstorm": [
        "thunderstorm", "thunderstorms", "convective activity",
        "convective weather", "cumulonimbus", "cb", "tstorm",
        "lightning", "convective_activity",
    ],
    "icing": [
        "icing", "icing_conditions", "icing conditions",
        "ice accumulation", "airframe icing", "rime ice",
        "clear ice", "mixed icing", "ice",
    ],
    "turbulence_light": [
        "light turbulence", "light_turbulence", "light chop",
        "occasional light turbulence", "bumpy_conditions",
        "bumpy conditions", "light bumps",
    ],
    "turbulence_moderate": [
        "moderate turbulence", "moderate_turbulence",
        "moderate chop", "moderate_chop", "mod turb",
        "continuous moderate turbulence",
    ],
    "turbulence_severe": [
        "severe turbulence", "severe_turbulence",
        "extreme turbulence", "severe chop",
    ],
    "turbulence": [
        "turbulence",
    ],
    "wind_shear": [
        "wind shear", "windshear", "wind_shear",
        "microburst", "low level wind shear", "llws",
    ],
    "low_visibility": [
        "low visibility", "low_visibility", "reduced visibility",
        "poor visibility", "restricted visibility",
    ],
    "fog": [
        "fog", "dense fog", "ground fog", "radiation fog",
        "advection fog", "foggy",
    ],
    "snow_ice_runway": [
        "snow", "ice on runway", "snow on runway",
        "contaminated runway", "slippery runway",
        "icy runway", "braking action poor",
    ],
    "crosswind": [
        "crosswind", "crosswinds", "cross wind",
        "gusty crosswind", "gusty_winds", "gusty winds",
    ],
    "rain": [
        "rain", "heavy rain", "rainfall",
        "rain showers", "precipitation",
    ],
    "haze": [
        "haze", "hazy", "hazy_sky", "hazy sky",
        "smoke", "smoke and haze",
    ],
    "tailwind": [
        "tailwind", "tail wind", "tailwind component",
    ],
    "low_ceiling": [
        "low ceilings", "low ceiling", "low_ceilings",
        "overcast", "ceiling", "broken clouds",
    ],
    "clouds": [
        "clouds", "cloud", "cloud layer",
        "vfr_on_top", "vfr on top",
    ],
    "wake_turbulence": [
        "wake turbulence", "wake_turbulence", "wake vortex",
        "jet wash", "wake", "vortex encounter",
    ],
    "clear_air_turbulence": [
        "clear air turbulence", "cat", "upper level turbulence",
        "jet stream turbulence",
    ],
}

# ---------------------------------------------------------------------------
# Airport ICAO Code Normalization
# Top 100 US + 50 international airports
# ---------------------------------------------------------------------------

AIRPORT_CODES: dict[str, list[str]] = {
    # Top US airports
    "KATL": ["atl", "atlanta", "hartsfield", "hartsfield-jackson", "katl"],
    "KLAX": ["lax", "los angeles", "klax", "los angeles international"],
    "KORD": ["ord", "chicago", "o'hare", "ohare", "kord", "chicago o'hare"],
    "KDFW": ["dfw", "dallas", "dallas-fort worth", "dallas fort worth", "kdfw"],
    "KDEN": ["den", "denver", "kden", "denver international"],
    "KJFK": ["jfk", "kennedy", "john f kennedy", "kjfk", "new york jfk"],
    "KSFO": ["sfo", "san francisco", "ksfo", "san francisco international"],
    "KLAS": ["las", "las vegas", "mccarran", "klas", "harry reid"],
    "KPHX": ["phx", "phoenix", "kphx", "phoenix sky harbor"],
    "KMIA": ["mia", "miami", "kmia", "miami international"],
    "KIAH": ["iah", "houston", "george bush", "kiah", "houston intercontinental"],
    "KMSP": ["msp", "minneapolis", "kmsp", "minneapolis-st paul"],
    "KSEA": ["sea", "seattle", "ksea", "seattle-tacoma", "seatac"],
    "KDTW": ["dtw", "detroit", "kdtw", "detroit metropolitan"],
    "KEWR": ["ewr", "newark", "kewr", "newark liberty"],
    "KBOS": ["bos", "boston", "kbos", "boston logan", "logan"],
    "KLGA": ["lga", "laguardia", "klga", "la guardia"],
    "KMCO": ["mco", "orlando", "kmco", "orlando international"],
    "KCLT": ["clt", "charlotte", "kclt", "charlotte douglas"],
    "KPHL": ["phl", "philadelphia", "kphl"],
    "KBWI": ["bwi", "baltimore", "kbwi", "baltimore-washington"],
    "KSLC": ["slc", "salt lake city", "kslc", "salt lake"],
    "KSAN": ["san", "san diego", "ksan", "san diego international"],
    "KDCA": ["dca", "reagan", "kdca", "washington national", "ronald reagan"],
    "KIAD": ["iad", "dulles", "kiad", "washington dulles"],
    "KTPA": ["tpa", "tampa", "ktpa", "tampa international"],
    "KPDX": ["pdx", "portland", "kpdx"],
    "KFLL": ["fll", "fort lauderdale", "kfll"],
    "KSTL": ["stl", "st louis", "kstl", "lambert"],
    "KBNA": ["bna", "nashville", "kbna"],
    "KAUS": ["aus", "austin", "kaus"],
    "KMDW": ["mdw", "midway", "kmdw", "chicago midway"],
    "KRDU": ["rdu", "raleigh", "krdu", "raleigh-durham"],
    "KCLE": ["cle", "cleveland", "kcle"],
    "KPIT": ["pit", "pittsburgh", "kpit"],
    "KCVG": ["cvg", "cincinnati", "kcvg"],
    "KIND": ["ind", "indianapolis", "kind"],
    "KMKE": ["mke", "milwaukee", "kmke"],
    "KSMF": ["smf", "sacramento", "ksmf"],
    "KSJC": ["sjc", "san jose", "ksjc"],
    "KOAK": ["oak", "oakland", "koak"],
    "KHNL": ["hnl", "honolulu", "khnl"],
    "PANC": ["anc", "anchorage", "panc"],
    "KMEM": ["mem", "memphis", "kmem"],
    "KMCI": ["mci", "kansas city", "kmci"],
    "KABQ": ["abq", "albuquerque", "kabq"],
    "KELP": ["elp", "el paso", "kelp"],
    "KTUS": ["tus", "tucson", "ktus"],
    "KONT": ["ont", "ontario", "kont"],
    "KBUR": ["bur", "burbank", "kbur", "hollywood burbank"],
    # International
    "EGLL": ["lhr", "heathrow", "london heathrow", "egll"],
    "LFPG": ["cdg", "charles de gaulle", "paris cdg", "lfpg"],
    "EDDF": ["fra", "frankfurt", "eddf"],
    "EHAM": ["ams", "amsterdam", "schiphol", "eham"],
    "RJTT": ["hnd", "haneda", "tokyo haneda", "rjtt"],
    "RJAA": ["nrt", "narita", "tokyo narita", "rjaa"],
    "VHHH": ["hkg", "hong kong", "vhhh"],
    "WSSS": ["sin", "singapore", "changi", "wsss"],
    "OMDB": ["dxb", "dubai", "omdb"],
    "CYYZ": ["yyz", "toronto", "pearson", "cyyz", "toronto pearson"],
    "CYVR": ["yvr", "vancouver", "cyvr"],
    "CYUL": ["yul", "montreal", "cyul", "montreal trudeau"],
    "MMEX": ["mex", "mexico city", "mmex"],
    "SBGR": ["gru", "guarulhos", "sao paulo", "sbgr"],
    "LEMD": ["mad", "madrid", "lemd", "barajas"],
    "LEBL": ["bcn", "barcelona", "lebl"],
    "LIRF": ["fco", "rome", "fiumicino", "lirf"],
    "LTFM": ["ist", "istanbul", "ltfm"],
    "RKSI": ["icn", "incheon", "seoul", "rksi"],
    "ZBAA": ["pek", "beijing", "zbaa"],
    "ZSPD": ["pvg", "shanghai pudong", "zspd"],
    "VIDP": ["del", "delhi", "vidp", "indira gandhi"],
    "VABB": ["bom", "mumbai", "vabb"],
    "YSSY": ["syd", "sydney", "yssy"],
    "YMML": ["mel", "melbourne", "ymml"],
    "FAOR": ["jnb", "johannesburg", "faor"],
    "FACT": ["cpt", "cape town", "fact"],
    "DNMM": ["los", "lagos", "dnmm"],
    "HKJK": ["nbo", "nairobi", "hkjk"],
    # Additional international airports
    "RCTP": ["tpe", "taipei", "taoyuan", "rctp", "taipei taoyuan"],
    "RPLL": ["mnl", "manila", "rpll", "ninoy aquino"],
    "VTBS": ["bkk", "bangkok", "suvarnabhumi", "vtbs"],
    "WMKK": ["kul", "kuala lumpur", "wmkk", "klia"],
    "WIII": ["cgk", "jakarta", "wiii", "soekarno-hatta"],
    "OTHH": ["doh", "doha", "othh", "hamad"],
    "OEJN": ["jed", "jeddah", "oejn", "king abdulaziz"],
    "OERK": ["ruh", "riyadh", "oerk", "king khalid"],
    "HECA": ["cai", "cairo", "heca", "cairo international"],
    "GOBD": ["dss", "dakar", "gobd", "blaise diagne"],
    "SCEL": ["scl", "santiago", "scel", "arturo merino benitez"],
    "SKBO": ["bog", "bogota", "skbo", "el dorado"],
    "SPJC": ["lim", "lima", "spjc", "jorge chavez"],
    "MMMX": ["mex", "mexico city benito juarez", "mmmx"],
    "SABE": ["aep", "buenos aires aeroparque", "sabe"],
    "SAEZ": ["eze", "buenos aires ezeiza", "saez", "ministro pistarini"],
    "LOWW": ["vie", "vienna", "loww"],
    "LSZH": ["zrh", "zurich", "lszh"],
    "EKCH": ["cph", "copenhagen", "ekch", "kastrup"],
    "ENGM": ["osl", "oslo", "engm", "gardermoen"],
    "ESSA": ["arn", "stockholm", "essa", "arlanda"],
    "EFHK": ["hel", "helsinki", "efhk", "helsinki-vantaa"],
    "EPWA": ["waw", "warsaw", "epwa", "chopin"],
    "LKPR": ["prg", "prague", "lkpr", "vaclav havel"],
    "LPPT": ["lis", "lisbon", "lppt"],
    "EIDW": ["dub", "dublin", "eidw"],
    "EGKK": ["lgw", "gatwick", "london gatwick", "egkk"],
    "EGCC": ["man", "manchester", "egcc"],
    "LFPO": ["ory", "orly", "paris orly", "lfpo"],
    "EDDM": ["muc", "munich", "eddm", "franz josef strauss"],
    "EDDB": ["ber", "berlin", "eddb", "berlin brandenburg"],
    "LOWI": ["inn", "innsbruck", "lowi"],
    "ZGGG": ["can", "guangzhou", "zggg", "baiyun"],
    "ZUUU": ["ctu", "chengdu", "zuuu", "shuangliu"],
    "NZAA": ["akl", "auckland", "nzaa"],
    "YBBN": ["bne", "brisbane", "ybbn"],
    # --- Additional European airports ---
    "EGSS": ["stn", "stansted", "london stansted", "egss"],
    "LGAV": ["ath", "athens", "lgav", "eleftherios venizelos"],
    "LHBP": ["bud", "budapest", "lhbp", "budapest liszt ferenc"],
    "LIMC": ["mxp", "milan malpensa", "limc", "malpensa"],
    "EBBR": ["bru", "brussels", "ebbr", "brussels zaventem"],
    "LROP": ["otp", "bucharest", "lrop", "henri coanda"],
    "LYBE": ["beg", "belgrade", "lybe", "nikola tesla"],
    "LWSK": ["skp", "skopje", "lwsk"],
    "LDZA": ["zag", "zagreb", "ldza"],
    "LZIB": ["bts", "bratislava", "lzib"],
    "EVRA": ["rix", "riga", "evra"],
    "EYVI": ["vno", "vilnius", "eyvi"],
    "EETN": ["tll", "tallinn", "eetn"],
    "BIKF": ["kef", "reykjavik", "keflavik", "bikf"],
    "EGPH": ["edi", "edinburgh", "egph"],
    "LEAL": ["alc", "alicante", "leal"],
    "LGTS": ["skg", "thessaloniki", "lgts"],
    "LFML": ["mrs", "marseille", "lfml", "marseille provence"],
    # --- Additional Asian airports ---
    "VOMM": ["maa", "chennai", "vomm", "chennai international"],
    "VECC": ["ccu", "kolkata", "vecc", "netaji subhas chandra bose"],
    "VOBL": ["blr", "bangalore", "vobl", "kempegowda"],
    "OPKC": ["khi", "karachi", "opkc", "jinnah international"],
    "OPLA": ["lhe", "lahore", "opla", "allama iqbal"],
    "VDPP": ["pnh", "phnom penh", "vdpp"],
    "VLVT": ["vte", "vientiane", "vlvt", "wattay"],
    "VNKT": ["ktm", "kathmandu", "vnkt", "tribhuvan"],
    "VRMM": ["mle", "male", "vrmm", "velana"],
    "VCBI": ["cmb", "colombo", "vcbi", "bandaranaike"],
    # --- Additional Middle East / Africa airports ---
    "OBBI": ["bah", "bahrain", "obbi", "bahrain international"],
    "OKBK": ["kwi", "kuwait", "okbk", "kuwait international"],
    "OIIE": ["ika", "tehran", "oiie", "imam khomeini"],
    "LLBG": ["tlv", "tel aviv", "llbg", "ben gurion"],
    "GABS": ["abj", "abidjan", "gabs", "felix houphouet boigny"],
    # --- Additional Americas airports ---
    "SBBR": ["bsb", "brasilia", "sbbr", "presidente juscelino kubitschek"],
    "SEQM": ["uio", "quito", "seqm", "mariscal sucre"],
    "TNCM": ["sxm", "st maarten", "tncm", "princess juliana"],
    "SVMI": ["ccs", "caracas", "svmi", "simon bolivar"],
    "SUMU": ["mvd", "montevideo", "sumu", "carrasco"],
    "TJSJ": ["sju", "san juan", "tjsj", "luis munoz marin"],
}

# ---------------------------------------------------------------------------
# Aggregate all taxonomies by entity type
# ---------------------------------------------------------------------------

def _build_lookup(taxonomy: dict[str, list[str]]) -> dict[str, str]:
    """Build alias -> canonical lookup from a taxonomy dict."""
    lookup: dict[str, str] = {}
    for canonical, aliases in taxonomy.items():
        canonical_lower = canonical.lower()
        lookup[canonical_lower] = canonical
        for alias in aliases:
            lookup[alias.lower().strip()] = canonical
    return lookup

def _build_hfacs_lookup() -> dict[str, str]:
    """Build alias -> canonical lookup from the HFACS taxonomy."""
    lookup: dict[str, str] = {}
    for canonical, info in HFACS_TAXONOMY.items():
        lookup[canonical] = canonical
        for alias in info.get("aliases", []):
            lookup[alias.lower().strip()] = canonical
    return lookup

# Pre-built lookups for exact matching
_PHASE_LOOKUP = _build_lookup(PHASE_OF_FLIGHT)
_AIRCRAFT_LOOKUP = _build_lookup(AIRCRAFT_TYPES)
_WEATHER_LOOKUP = _build_lookup(WEATHER_CONDITIONS)
_AIRPORT_LOOKUP = _build_lookup(AIRPORT_CODES)
_HFACS_LOOKUP = _build_hfacs_lookup()

# Map entity types to their taxonomies
TAXONOMY_LOOKUPS: dict[str, dict[str, str]] = {
    "Phase": _PHASE_LOOKUP,
    "Aircraft": _AIRCRAFT_LOOKUP,
    "Weather": _WEATHER_LOOKUP,
    "ATC_Facility": _AIRPORT_LOOKUP,
    "Factor": _HFACS_LOOKUP,
}

# Canonical names for embedding-based matching (per type)
TAXONOMY_CANONICALS: dict[str, list[str]] = {
    "Phase": list(PHASE_OF_FLIGHT.keys()),
    "Aircraft": list(AIRCRAFT_TYPES.keys()),
    "Weather": list(WEATHER_CONDITIONS.keys()),
    "ATC_Facility": list(AIRPORT_CODES.keys()),
    "Factor": list(HFACS_TAXONOMY.keys()),
}


# ---------------------------------------------------------------------------
# Entity Resolution
# ---------------------------------------------------------------------------

def _normalize_raw(name: str) -> str:
    """Normalize a raw entity name for matching."""
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    name = name.rstrip(".,;:!?")
    # Collapse underscores to spaces for matching
    name = name.replace("_", " ")
    return name



def resolve_entity(
    raw_name: str,
    entity_type: str,
) -> tuple[str, float]:
    """Resolve a raw entity name to its canonical form via exact match.

    Returns (canonical_name, confidence).
    """
    lookup = TAXONOMY_LOOKUPS.get(entity_type)
    if lookup is None:
        return (raw_name, 0.0)

    normalized = _normalize_raw(raw_name)

    if entity_type == "Aircraft" and normalized in ANONYMIZED_AIRCRAFT:
        return (raw_name, 0.0)

    if normalized in lookup:
        return (lookup[normalized], 1.0)

    return (raw_name, 0.0)
