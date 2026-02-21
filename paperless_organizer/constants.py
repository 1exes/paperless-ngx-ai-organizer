"""Static data constants (hints, regex patterns, BLZ codes, guardrails)."""

from __future__ import annotations

# --- Erlaubte Dokumenttypen (20 Stueck) ---
ALLOWED_DOC_TYPES = [
    "Vertrag", "Rechnung", "Bescheinigung", "Information", "Kontoauszug",
    "Zeugnis", "Angebot", "Kuendigung", "Mahnung", "Versicherungspolice",
    "Steuerbescheid", "Arztbericht", "Gehaltsabrechnung", "Bestellung",
    "Korrespondenz", "Dokumentation", "Lizenz", "Formular", "Urkunde",
    "Bewerbung",
]

# --- Tag-Whitelist (nicht loeschen) ---
TAG_WHITELIST = {"kfz", "service", "termin", "aufhebungsvertrag", "ausbildung", "fachinformatiker"}

# --- Tag-Farbpalette (20 Farben) ---
TAG_COLORS = [
    "#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#2980b9", "#27ae60", "#c0392b",
    "#8e44ad", "#16a085", "#d35400", "#2c3e50", "#f1c40f",
    "#e91e63", "#00bcd4", "#ff5722", "#607d8b", "#795548",
]

# --- Korrespondenten-Dedupe (generisch, konservativ) ---
CORRESPONDENT_LEGAL_TOKENS = {
    "gmbh", "ag", "mbh", "kg", "kgaa", "ug", "eg", "e", "ev", "e.v", "ggmbh",
    "inc", "ltd", "llc", "corp", "co", "company", "corporation", "sarl", "sa",
}
CORRESPONDENT_STOPWORDS = {
    "der", "die", "das", "und", "im", "in", "am", "an", "auf", "mit", "von", "zu", "zum", "zur",
    "des", "dem", "den", "for", "of", "the", "via", "team", "support",
}

# --- Bekannte Marken / Hinweise ---
KNOWN_BRAND_HINTS = {
    "elevenlabs": "ElevenLabs ist ein KI-Voice/Audio SaaS Anbieter (Abo-Dienst, meist privat/IT/SaaS).",
    "github": "GitHub ist ein Entwickler- und SaaS-Dienst (oft IT/SaaS, nicht automatisch Arbeit).",
    "jetbrains": "JetBrains ist ein Software-Aboanbieter (IT/SaaS).",
    "openai": "OpenAI ist ein KI/SaaS Anbieter (IT/SaaS).",
    "google cloud": "Google Cloud ist ein Cloud/SaaS Anbieter (normalerweise privat/IT/Finanzen, nicht Arbeitgeber).",
    "google": "Google ist ein Technologieanbieter (Kontext pruefen, nicht automatisch Arbeitgeber).",
    "microsoft": "Microsoft ist ein Software/Cloud Anbieter (Azure, M365 etc.).",
}

EMPLOYER_HINTS: set[str] = set()  # Configure via learning_profile.json

# --- Korrespondenten-Aliase ---
CORRESPONDENT_ALIASES = {
    "google cloud": "Google Cloud",
}

# --- Rechtschreibkorrekturen ---
SPELLING_FIXES: dict[str, str] = {}

TITLE_SPELLING_FIXES: dict[str, str] = {}

# --- Vendor-Guardrails ---
VENDOR_GUARDRAILS = {
    "google cloud": {
        "correspondent": "Google Cloud",
        "path_preferences": ["Freizeit/IT", "Finanzen/Rechnungen", "Finanzen", "IT"],
    },
    "github": {
        "correspondent": "GitHub",
        "path_preferences": ["Freizeit/IT", "Finanzen/Rechnungen", "Finanzen", "IT"],
    },
    "elevenlabs": {
        "correspondent": "ElevenLabs",
        "path_preferences": ["Freizeit/IT", "Finanzen/Rechnungen", "Finanzen", "IT"],
    },
    "jetbrains": {
        "correspondent": "JetBrains",
        "path_preferences": ["Freizeit/IT", "Finanzen/Rechnungen", "Finanzen", "IT"],
    },
    "openai": {
        "correspondent": "OpenAI",
        "path_preferences": ["Freizeit/IT", "Finanzen/Rechnungen", "Finanzen", "IT"],
    },
}

# --- Fahrzeug / Gesundheit / Schule / Event / Transport Hinweise ---
PRIVATE_VEHICLE_HINTS: list[str] = []  # Configure via learning_profile.json
COMPANY_VEHICLE_HINTS: list[str] = []  # Configure via learning_profile.json
HEALTH_HINTS = [
    "allergie", "allergen", "arzt", "klinikum", "testzentrum", "sars-cov-2", "covid",
    "krankenkasse", "arbeitsunfaehigkeit", "arbeitsunfähigkeit", "krankschreibung", "anamnese",
]
SCHOOL_HINTS = [
    "berufliches schulzentrum", "oberschule", "abschlusszeugnis", "zeugnis", "pruefung",
    "prüfung", "fahrschule", "fahrerlaubnispruefung", "fahrerlaubnisprüfung", "anmeldebestaetigung",
    "anmeldebestätigung",
]
EVENT_TICKET_HINTS = [
    "event ticket", "openair", "konzert", "festival", "ticket.io", "eintrittskarte",
    "party", "birthday party", "club", "veranstaltung",
]
TRANSPORT_TICKET_HINTS = [
    "fahrkarte", "deutschlandticket", "deutsche bahn", "bahn", "verkehrsverbund", "omnibus", "bus",
    "zugticket", "bahnticket",
]

# --- German month names for date parsing ---
GERMAN_MONTHS = {
    "januar": 1, "februar": 2, "maerz": 3, "märz": 3, "april": 4,
    "mai": 5, "juni": 6, "juli": 7, "august": 8, "september": 9,
    "oktober": 10, "november": 11, "dezember": 12,
    "jan": 1, "feb": 2, "mär": 3, "mar": 3, "apr": 4, "mai": 5,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "okt": 10, "nov": 11, "dez": 12,
}

# --- Language detection markers ---
GERMAN_MARKERS = {
    "und", "oder", "der", "die", "das", "ein", "eine", "ist", "sind", "wird", "wurde",
    "bei", "mit", "nach", "fuer", "ueber", "unter", "zwischen", "durch", "gegen",
    "rechnung", "vertrag", "sehr", "geehrte", "freundlichen", "gruessen", "bestaetigung",
    "versicherung", "kontoauszug", "bescheinigung", "mitteilung", "kuendigung",
    "antrag", "mietvertrag", "steuerbescheid",
}
ENGLISH_MARKERS = {
    "the", "and", "for", "with", "this", "that", "from", "your", "have", "been",
    "invoice", "receipt", "agreement", "confirmation", "payment", "subscription",
    "account", "statement", "service", "please", "thank", "dear", "regards",
}

# --- BLZ (Bankleitzahlen) ---
KNOWN_BLZ = {
    "10010010": "Postbank",
    "10020500": "Bank fuer Sozialwirtschaft",
    "10050000": "Landesbank Berlin",
    "10070000": "Deutsche Bank Berlin",
    "10070024": "Deutsche Bank",
    "10070848": "Commerzbank",
    "10090000": "Berliner Volksbank",
    "12030000": "Deutsche Kreditbank (DKB)",
    "20010020": "Postbank Hamburg",
    "20050550": "Haspa",
    "20070000": "Deutsche Bank Hamburg",
    "20070024": "Deutsche Bank",
    "25050000": "Nord/LB",
    "26050001": "Sparkasse Osnabrueck",
    "30010111": "SEB",
    "30020900": "Targobank",
    "30060601": "Deutsche Apotheker- und Aerztebank",
    "37010050": "Postbank Koeln",
    "43060967": "Volksbank Paderborn",
    "50010517": "ING",
    "50020200": "BHF-BANK",
    "50040000": "Commerzbank Frankfurt",
    "50050201": "Frankfurter Sparkasse",
    "50060400": "Evangelische Bank",
    "50070010": "Deutsche Bank Frankfurt",
    "50070024": "Deutsche Bank",
    "50080000": "Commerzbank",
    "50090500": "Sparda-Bank Hessen",
    "50310400": "Baader Bank",
    "51230800": "Volksbank Mittelhessen",
    "60020290": "UniCredit Bank - HypoVereinsbank",
    "60050101": "BW-Bank",
    "60070070": "Deutsche Bank Stuttgart",
    "68452290": "Sparkasse Markgraeflerland",
    "70010080": "Postbank Muenchen",
    "70020270": "UniCredit Bank - HypoVereinsbank",
    "70050000": "Bayerische Landesbank",
    "70070010": "Deutsche Bank Muenchen",
    "70070024": "Deutsche Bank",
    "70090100": "Volksbank Raiffeisenbank",
    "76010085": "Postbank Nuernberg",
    "82050000": "Sparkasse Gera-Greiz",
    "83050000": "Sparkasse Mittelthueringen",
    "85050100": "Sparkasse Chemnitz",
    "85050300": "Ostsaechsische Sparkasse Dresden",
    "86050200": "Sparkasse Leipzig",
    "10050006": "Landesbank Berlin - Berliner Sparkasse",
    "30050110": "Stadtsparkasse Duesseldorf",
    "76050101": "Sparkasse Nuernberg",
    "44050199": "Sparkasse Dortmund",
}

# --- Monatsnamen (deutsch, fuer Statistik-Ausgabe) ---
MONTH_NAMES_DE = [
    "", "Januar", "Februar", "Maerz", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Dezember",
]
