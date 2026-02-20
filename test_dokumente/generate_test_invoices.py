"""
Generiert realistische Test-PDFs (Rechnungen, Bescheinigungen, etc.)
fuer Paperless-NGX Organizer-Tests.

Layout orientiert sich an echten deutschen Geschaeftsbriefen (DIN 5008):
- Farbiger Firmenkopf mit Branding
- Absenderzeile klein ueber Empfaenger
- Infoblock rechts (Datum, Kundennr, etc.)
- Professionelle Tabellen mit Rahmen
- Fusszeile mit Bankverbindung / Impressum

Ausfuehren:
    pip install fpdf2
    python generate_test_invoices.py
"""

from __future__ import annotations

import os
from fpdf import FPDF

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

EMPFAENGER = ("Max Mustermann", "Musterstrasse 42", "01099 Dresden")


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _safe(text: str) -> str:
    """Latin-1-safe fuer fpdf Standard-Fonts."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _text_for_bg(r: int, g: int, b: int) -> tuple[int, int, int]:
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (255, 255, 255) if lum < 160 else (0, 0, 0)


# ---------------------------------------------------------------------------
# PDF-Basisklasse mit professionellem Layout
# ---------------------------------------------------------------------------

class BusinessPDF(FPDF):
    """Professionelle Briefvorlage mit Firmenbranding."""

    def __init__(self, doc: dict):
        super().__init__()
        self.doc = doc
        brand = doc.get("brand_color", "#2c3e50")
        self.brand_rgb = _hex_to_rgb(brand)
        self.brand_text = _text_for_bg(*self.brand_rgb)
        accent = doc.get("accent_color", "#ecf0f1")
        self.accent_rgb = _hex_to_rgb(accent)
        self.set_auto_page_break(auto=True, margin=28)

    # --- Header: farbiger Balken + Firmenname + Adresse rechts ---
    def header(self):
        doc = self.doc
        r, g, b = self.brand_rgb
        tr, tg, tb = self.brand_text

        # Farbbalken oben
        self.set_fill_color(r, g, b)
        self.rect(0, 0, 210, 22, "F")

        # Firmenname im Balken
        self.set_xy(12, 4)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(tr, tg, tb)
        self.cell(120, 7, _safe(doc["firma"]), new_x="LMARGIN", new_y="NEXT")

        # Untertitel / Slogan
        if doc.get("slogan"):
            self.set_xy(12, 12)
            self.set_font("Helvetica", "", 8)
            self.cell(120, 4, _safe(doc["slogan"]), new_x="LMARGIN", new_y="NEXT")

        # Firmenadresse rechts oben
        self.set_font("Helvetica", "", 7)
        self.set_text_color(tr, tg, tb)
        y_start = 5
        for line in doc["adresse"].split("\n"):
            self.set_xy(140, y_start)
            self.cell(60, 3.5, _safe(line), align="R", new_x="LMARGIN", new_y="NEXT")
            y_start += 3.5

        self.set_text_color(0, 0, 0)
        self.set_y(28)

    # --- Footer: Bankdaten / Impressum ---
    def footer(self):
        doc = self.doc
        r, g, b = self.brand_rgb

        self.set_y(-24)
        # Trennlinie
        self.set_draw_color(r, g, b)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)

        self.set_font("Helvetica", "", 6.5)
        self.set_text_color(100, 100, 100)

        footer_lines = doc.get("footer", [])
        if footer_lines:
            # Bis zu 3 Spalten im Footer
            col_w = 63
            y_base = self.get_y()
            for i, col_text in enumerate(footer_lines[:3]):
                self.set_xy(10 + i * col_w, y_base)
                for line in col_text.split("\n"):
                    self.cell(col_w, 3, _safe(line), new_x="LMARGIN", new_y="NEXT")
                    self.set_x(10 + i * col_w)
        else:
            self.cell(0, 3, _safe(f"{doc['firma']} - Seite {self.page_no()}"), align="C")

        self.set_text_color(0, 0, 0)

    # --- Absenderzeile + Empfaenger (DIN 5008) ---
    def add_address_block(self):
        doc = self.doc
        r, g, b = self.brand_rgb

        # Absenderzeile klein, unterstrichen
        self.set_font("Helvetica", "U", 6.5)
        self.set_text_color(r, g, b)
        sender_line = " - ".join(doc["adresse"].split("\n")[:3])
        self.cell(90, 3.5, _safe(sender_line), new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

        # Empfaenger
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "", 10.5)
        for line in EMPFAENGER:
            self.cell(90, 5.5, _safe(line), new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    # --- Infoblock rechts (Datum, Kundennr., Referenz) ---
    def add_info_block(self):
        doc = self.doc
        info = doc.get("info_block", {})
        if not info:
            return

        r, g, b = self.brand_rgb
        x_label = 130
        x_value = 162
        y_start = 30

        self.set_font("Helvetica", "B", 7.5)
        self.set_text_color(r, g, b)

        for i, (label, value) in enumerate(info.items()):
            y = y_start + i * 5.5
            self.set_xy(x_label, y)
            self.cell(30, 4, _safe(label), new_x="LMARGIN")
            self.set_xy(x_value, y)
            self.set_font("Helvetica", "", 7.5)
            self.set_text_color(40, 40, 40)
            self.cell(40, 4, _safe(str(value)), new_x="LMARGIN", new_y="NEXT")
            self.set_font("Helvetica", "B", 7.5)
            self.set_text_color(r, g, b)

        self.set_text_color(0, 0, 0)

    # --- Betreffzeile ---
    def add_subject(self, text: str):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, _safe(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    # --- Fliesstext ---
    def add_paragraph(self, text: str, bold: bool = False, size: float = 9.5):
        self.set_font("Helvetica", "B" if bold else "", size)
        self.multi_cell(0, 5, _safe(text))
        self.ln(1.5)

    # --- Tabelle mit Rahmen ---
    def add_table(self, headers: list[str], rows: list[list[str]],
                  col_widths: list[float] | None = None,
                  total_row: list[str] | None = None,
                  subtotal_rows: list[list[str]] | None = None):
        r, g, b = self.brand_rgb
        ar, ag, ab = self.accent_rgb

        if col_widths is None:
            w = 190 / len(headers)
            col_widths = [w] * len(headers)

        # Header
        self.set_fill_color(r, g, b)
        tr, tg, tb = self.brand_text
        self.set_text_color(tr, tg, tb)
        self.set_font("Helvetica", "B", 8)
        for i, h in enumerate(headers):
            align = "R" if i == len(headers) - 1 else "L"
            self.cell(col_widths[i], 7, _safe(h), border=1, fill=True, align=align)
        self.ln()

        # Zeilen
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "", 8)
        for row_idx, row in enumerate(rows):
            if row_idx % 2 == 1:
                self.set_fill_color(ar, ag, ab)
                fill = True
            else:
                fill = False
            for i, cell in enumerate(row):
                align = "R" if i == len(row) - 1 else "L"
                self.cell(col_widths[i], 6, _safe(cell), border="LR", fill=fill, align=align)
            self.ln()

        # Abschluss-Linie
        self.set_draw_color(r, g, b)
        self.set_line_width(0.3)
        x = self.get_x()
        self.line(x, self.get_y(), x + sum(col_widths), self.get_y())
        self.ln(1)

        # Zwischensummen
        if subtotal_rows:
            self.set_font("Helvetica", "", 8)
            for sr in subtotal_rows:
                label_w = sum(col_widths[:-1])
                self.cell(label_w, 5.5, _safe(sr[0]), align="R")
                self.cell(col_widths[-1], 5.5, _safe(sr[1]), align="R")
                self.ln()

        # Summenzeile
        if total_row:
            self.ln(1)
            self.set_fill_color(r, g, b)
            self.set_text_color(tr, tg, tb)
            self.set_font("Helvetica", "B", 9)
            label_w = sum(col_widths[:-1])
            self.cell(label_w, 7.5, _safe(total_row[0]), border=1, fill=True, align="R")
            self.cell(col_widths[-1], 7.5, _safe(total_row[1]), border=1, fill=True, align="R")
            self.set_text_color(0, 0, 0)
            self.ln()

    # --- Unterschriftsblock ---
    def add_signature(self, city_date: str, name: str, title: str = ""):
        self.ln(12)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, _safe(city_date), new_x="LMARGIN", new_y="NEXT")
        self.ln(10)
        self.set_draw_color(60, 60, 60)
        self.line(10, self.get_y(), 70, self.get_y())
        self.ln(2)
        self.set_font("Helvetica", "B", 9)
        self.cell(0, 5, _safe(name), new_x="LMARGIN", new_y="NEXT")
        if title:
            self.set_font("Helvetica", "", 8)
            self.cell(0, 4, _safe(title), new_x="LMARGIN", new_y="NEXT")

    # --- Highlight-Box ---
    def add_highlight_box(self, text: str, style: str = "info"):
        colors = {
            "info": ("#eaf2f8", "#2980b9"),
            "success": ("#eafaf1", "#27ae60"),
            "warning": ("#fef9e7", "#f39c12"),
            "danger": ("#fdedec", "#e74c3c"),
        }
        bg_hex, border_hex = colors.get(style, colors["info"])
        bg = _hex_to_rgb(bg_hex)
        border = _hex_to_rgb(border_hex)

        self.set_fill_color(*bg)
        self.set_draw_color(*border)
        self.set_line_width(0.5)

        x, y = self.get_x(), self.get_y()
        self.rect(x, y, 190, 14, "DF")
        self.set_xy(x + 4, y + 3)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*border)
        self.cell(0, 5, _safe(text))
        self.set_text_color(0, 0, 0)
        self.set_xy(x, y + 17)


# ---------------------------------------------------------------------------
# Dokumentdefinitionen
# ---------------------------------------------------------------------------

DOCUMENTS = [
    # 1 - Telekom Rechnung
    {
        "filename": "rechnung_telekom_2025.pdf",
        "firma": "Deutsche Telekom AG",
        "slogan": "Erleben, was verbindet.",
        "adresse": "Telekom Deutschland GmbH\nLandgrabenweg 151\n53227 Bonn",
        "brand_color": "#e20074",
        "accent_color": "#fce4ec",
        "info_block": {
            "Rechnungsdatum:": "28.02.2025",
            "Kundennummer:": "4920-8837-2211",
            "Rechnungsnr.:": "RE-2025-TMO-448291",
            "Rufnummer:": "0170 1234567",
            "Tarif:": "MagentaMobil L",
        },
        "footer": [
            "Deutsche Telekom AG\nFriedrich-Ebert-Allee 140\n53113 Bonn",
            "USt-IdNr.: DE 123456789\nAmtsgericht Bonn HRB 5919\nSitz der Gesellschaft: Bonn",
            "Bankverbindung:\nDeutsche Bank AG\nIBAN: DE85 3707 0024 0182 0000 00",
        ],
    },
    # 2 - Amazon Rechnung
    {
        "filename": "rechnung_amazon_januar.pdf",
        "firma": "Amazon EU S.a.r.l.",
        "slogan": "",
        "adresse": "Amazon EU S.a.r.l.\n38 avenue John F. Kennedy\nL-1855 Luxemburg",
        "brand_color": "#232f3e",
        "accent_color": "#f5f5f5",
        "info_block": {
            "Rechnungsdatum:": "14.01.2025",
            "Bestellnummer:": "302-4418823-9912654",
            "Rechnungsnr.:": "INV-2025-AMZ-302441",
            "Zahlungsart:": "Visa **** 4821",
        },
        "footer": [
            "Amazon EU S.a.r.l.\n38 avenue John F. Kennedy\nL-1855 Luxemburg",
            "R.C.S. Luxembourg: B-101818\nTVA: LU 20260743",
            "Kundenservice:\nwww.amazon.de/kontakt\nTel: 0800 363 8469",
        ],
    },
    # 3 - DREWAG Stadtwerke
    {
        "filename": "rechnung_stadtwerke_dresden.pdf",
        "firma": "SachsenEnergie AG",
        "slogan": "Energie fuer Dresden und Ostsachsen",
        "adresse": "SachsenEnergie AG\nFriedrich-List-Platz 2\n01069 Dresden",
        "brand_color": "#00703c",
        "accent_color": "#e8f5e9",
        "info_block": {
            "Rechnungsdatum:": "10.02.2025",
            "Vertragsnr.:": "SDD-2019-74523",
            "Abrechnungszeit:": "01.01. - 31.12.2024",
            "Kundennummer:": "700.841.291",
            "Zaehler-Nr.:": "1EMH-0047829",
        },
        "footer": [
            "SachsenEnergie AG\nFriedrich-List-Platz 2\n01069 Dresden",
            "AG Dresden HRB 2389\nVorsitzender des AR: D. Hilbert\nUSt-IdNr.: DE140857441",
            "Bankverbindung:\nOstsaechsische Sparkasse Dresden\nIBAN: DE06 8505 0300 3120 0800 06",
        ],
    },
    # 4 - JetBrains
    {
        "filename": "rechnung_jetbrains_2025.pdf",
        "firma": "JetBrains s.r.o.",
        "slogan": "The Drive to Develop",
        "adresse": "JetBrains s.r.o.\nNa Hrebenech II 1718/10\n140 00 Prague 4\nCzech Republic",
        "brand_color": "#6b57ff",
        "accent_color": "#ede7f6",
        "info_block": {
            "Invoice Date:": "01.02.2025",
            "Invoice No.:": "JB-INV-2025-22847",
            "License ID:": "JB-LIC-2025-98413",
            "Customer:": "Max Mustermann",
        },
        "footer": [
            "JetBrains s.r.o.\nNa Hrebenech II 1718/10\n140 00 Prague 4, Czech Republic",
            "IC: 26502275\nDIC: CZ26502275\nRegistered: Municipal Court Prague",
            "Payment:\nCredit Card ending 4821\nPaid: 01.02.2025",
        ],
    },
    # 5 - Hetzner
    {
        "filename": "rechnung_hetzner_hosting.pdf",
        "firma": "Hetzner Online GmbH",
        "slogan": "Cloud & Dedicated Hosting",
        "adresse": "Hetzner Online GmbH\nIndustriestrasse 25\n91710 Gunzenhausen",
        "brand_color": "#d50c2d",
        "accent_color": "#fce4ec",
        "info_block": {
            "Rechnungsdatum:": "01.02.2025",
            "Kundennummer:": "HZN-884721",
            "Rechnungsnr.:": "HZN-2025-01-884721",
            "Zeitraum:": "01.01. - 31.01.2025",
        },
        "footer": [
            "Hetzner Online GmbH\nIndustriestrasse 25\n91710 Gunzenhausen",
            "AG Ansbach HRB 6089\nGF: Martin Hetzner\nUSt-IdNr.: DE 812871812",
            "Bankverbindung:\nSparkasse Gunzenhausen\nIBAN: DE43 7655 0000 0008 0444 80",
        ],
    },
    # 6 - HUK-COBURG Kfz
    {
        "filename": "kfz_versicherung_huk.pdf",
        "firma": "HUK-COBURG Versicherungsgruppe",
        "slogan": "Das Plus fuer Ihre Versicherung",
        "adresse": "HUK-COBURG\nBahnhofsplatz\n96450 Coburg",
        "brand_color": "#003d6a",
        "accent_color": "#e3f2fd",
        "info_block": {
            "Datum:": "15.12.2024",
            "Versicherungsnr.:": "KH-7234-9182-01",
            "Fahrzeug:": "VW Polo 1.0 TSI",
            "Kennzeichen:": "DD-MM 123",
            "SF-Klasse:": "SF 8",
            "Zeitraum:": "01.01. - 31.12.2025",
        },
        "footer": [
            "HUK-COBURG\nHaftpflicht-Unterstützungs-Kasse\nkraftf. Beamter Deutschlands a.G.",
            "Sitz: Coburg\nAG Coburg VR 289\nVorstand: Klaus-Juergen Heitmann (Sprecher)",
            "Bankverbindung:\nHypoVereinsbank Coburg\nIBAN: DE21 7832 0076 0012 3456 78",
        ],
    },
    # 7 - WBS Gehaltsabrechnung
    {
        "filename": "gehaltsabrechnung_wbs_januar.pdf",
        "firma": "WBS TRAINING AG",
        "slogan": "Bildung. Digital. Mit Herz.",
        "adresse": "WBS TRAINING AG\nLorenzweg 5\n12099 Berlin",
        "brand_color": "#e85d00",
        "accent_color": "#fff3e0",
        "info_block": {
            "Abrechnungsmonat:": "Januar 2025",
            "Personalnummer:": "WBS-2025-0842",
            "Steuerklasse:": "I",
            "Eintrittsdatum:": "01.08.2025",
            "Krankenkasse:": "AOK PLUS",
        },
        "footer": [
            "WBS TRAINING AG\nLorenzweg 5\n12099 Berlin",
            "AG Charlottenburg HRB 165474 B\nVorstand: Heinrich Hueppe\nUSt-IdNr.: DE 275839078",
            "Diese Abrechnung ist maschinell\nerstellt und ohne Unterschrift\ngueltig.",
        ],
    },
    # 8 - Augenarzt
    {
        "filename": "befund_augenarzt_2025.pdf",
        "firma": "Dr. med. Katharina Lehmann",
        "slogan": "Fachpraxis fuer Augenheilkunde",
        "adresse": "Dr. med. K. Lehmann\nPrager Strasse 12\n01069 Dresden",
        "brand_color": "#0277bd",
        "accent_color": "#e1f5fe",
        "info_block": {
            "Untersuchungsdatum:": "22.01.2025",
            "Patient:": "Max Mustermann",
            "Geb.:": "01.01.1985",
            "Versicherung:": "AOK PLUS",
            "Vers.-Nr.:": "A123456789",
        },
        "footer": [
            "Praxis Dr. med. Katharina Lehmann\nFacharztin f. Augenheilkunde\nPrager Strasse 12, 01069 Dresden",
            "Tel.: 0351 / 482 73 00\nFax: 0351 / 482 73 01\npraxis@dr-lehmann-augen.de",
            "Sprechzeiten:\nMo-Fr 8:00-12:00\nMo,Di,Do 14:00-17:00",
        ],
    },
    # 9 - Scalable Capital
    {
        "filename": "kontoauszug_scalable_capital.pdf",
        "firma": "Scalable Capital GmbH",
        "slogan": "Vermoegensverwaltung",
        "adresse": "Scalable Capital GmbH\nSeitzstrasse 8e\n80538 Muenchen",
        "brand_color": "#1a237e",
        "accent_color": "#e8eaf6",
        "info_block": {
            "Stichtag:": "31.01.2025",
            "Depotnummer:": "SC-DE-2847193",
            "Depotinhaber:": "Max Mustermann",
            "Depotbank:": "Baader Bank AG",
        },
        "footer": [
            "Scalable Capital GmbH\nSeitzstrasse 8e\n80538 Muenchen",
            "AG Muenchen HRB 217778\nGF: Erik Podzuweit, Florian Prucker\nBaFin-Reg.: 155506",
            "Dieses Dokument dient zur\nInformation und ist kein\nSteuerdokument.",
        ],
    },
    # 10 - DRK
    {
        "filename": "mitgliedsbescheinigung_drk.pdf",
        "firma": "DRK Kreisverband Dresden e.V.",
        "slogan": "Aus Liebe zum Menschen",
        "adresse": "DRK KV Dresden e.V.\nKlingerstrasse 20\n01139 Dresden",
        "brand_color": "#cc0000",
        "accent_color": "#ffebee",
        "info_block": {
            "Datum:": "20.01.2025",
            "Mitgliedsnr.:": "DRK-DD-2018-4291",
            "Mitglied seit:": "01.04.2018",
            "Einsatzgruppe:": "Wasserwacht Nord",
        },
        "footer": [
            "DRK Kreisverband Dresden e.V.\nKlingerstrasse 20\n01139 Dresden",
            "VR 1784, AG Dresden\nVorsitzender: Dr. Frank Lippmann\nGemeinnuetziger Verein",
            "Spendenkonto:\nOstsaechs. Sparkasse Dresden\nIBAN: DE18 8505 0300 3120 0001 55",
        ],
    },
    # 11 - Vonovia Mietanpassung
    {
        "filename": "mietvertrag_aenderung_2025.pdf",
        "firma": "Vonovia SE",
        "slogan": "Zuhause ist, was wir daraus machen.",
        "adresse": "Vonovia SE\nUniversitaetsstrasse 133\n44803 Bochum",
        "brand_color": "#00457c",
        "accent_color": "#e3f2fd",
        "info_block": {
            "Datum:": "05.02.2025",
            "Mietvertragsnr.:": "VON-DD-2021-08834",
            "Objekt:": "Musterstr. 42, Dresden",
            "Wohnflaeche:": "62,5 m2",
            "Mieterin/Mieter:": "Max Mustermann",
        },
        "footer": [
            "Vonovia SE\nUniversitaetsstrasse 133\n44803 Bochum",
            "AG Bochum HRB 16879\nVorstand: Rolf Buch (CEO)\nUSt-IdNr.: DE 309189498",
            "Kundenservice:\nTel.: 0234 / 314 0\nvonovia.de/mieterservice",
        ],
    },
    # 12 - AOK PLUS
    {
        "filename": "aok_plus_bonusheft_2024.pdf",
        "firma": "AOK PLUS",
        "slogan": "Die Gesundheitskasse fuer Sachsen und Thueringen",
        "adresse": "AOK PLUS\nSternplatz 7\n01067 Dresden",
        "brand_color": "#00884a",
        "accent_color": "#e8f5e9",
        "info_block": {
            "Datum:": "28.01.2025",
            "Versichertennr.:": "A123456789",
            "Bonusjahr:": "2024",
            "Versicherter:": "Max Mustermann",
        },
        "footer": [
            "AOK PLUS\nDie Gesundheitskasse fuer\nSachsen und Thueringen",
            "Koerperschaft des oeffentlichen\nRechts\nHausanschrift: Sternplatz 7, Dresden",
            "Servicetelefon:\n0800 10 59 000 (kostenfrei)\naokplus.de",
        ],
    },
    # 13 - Steuerbescheid
    {
        "filename": "steuerbescheid_2023.pdf",
        "firma": "Finanzamt Dresden-Nord",
        "slogan": "Freistaat Sachsen - Saechsische Finanzverwaltung",
        "adresse": "Finanzamt Dresden-Nord\nRadeberger Strasse 22\n01099 Dresden",
        "brand_color": "#1b5e20",
        "accent_color": "#e8f5e9",
        "info_block": {
            "Datum:": "15.01.2025",
            "Steuernummer:": "203/123/45678",
            "Steuerpflichtiger:": "Max Mustermann",
            "Veranlagungsjahr:": "2023",
            "Sachbearbeiter/in:": "Fr. Bergmann",
            "Tel. Durchwahl:": "0351 / 8855-7412",
        },
        "footer": [
            "Finanzamt Dresden-Nord\nRadeberger Strasse 22\n01099 Dresden",
            "Oeffnungszeiten:\nDi 8-12, 13-18 Uhr\nDo 8-12, 13-16 Uhr",
            "Bankverbindung:\nFreistaat Sachsen\nIBAN: DE28 8600 0000 0086 0015 22",
        ],
    },
    # 14 - Fahrschule
    {
        "filename": "rechnung_fahrschule_update.pdf",
        "firma": "Fahrschule Schmidt GmbH",
        "slogan": "Sicher ans Ziel seit 1998",
        "adresse": "Fahrschule Schmidt GmbH\nKoenigsbruecker Str. 78\n01099 Dresden",
        "brand_color": "#ff6f00",
        "accent_color": "#fff8e1",
        "info_block": {
            "Rechnungsdatum:": "12.02.2025",
            "Rechnungsnr.:": "FS-2025-0294",
            "Fahrschueler:": "Max Mustermann",
            "Klasse:": "A2 (Aufstieg A1)",
        },
        "footer": [
            "Fahrschule Schmidt GmbH\nKoenigsbruecker Str. 78\n01099 Dresden",
            "AG Dresden HRB 38291\nGF: Klaus Schmidt\nUSt-IdNr.: DE 274918273",
            "Bankverbindung:\nOstsaechs. Sparkasse Dresden\nIBAN: DE12 8505 0300 0221 1234 56",
        ],
    },
    # 15 - Eventim / Rammstein
    {
        "filename": "ticket_rammstein_2025.pdf",
        "firma": "CTS EVENTIM AG & Co. KGaA",
        "slogan": "eventim.de - Dein Ticketshop",
        "adresse": "CTS EVENTIM AG & Co. KGaA\nContrescarpe 75A\n28195 Bremen",
        "brand_color": "#0050a0",
        "accent_color": "#e3f2fd",
        "info_block": {
            "Buchungsdatum:": "18.01.2025",
            "Buchungsnr.:": "EVT-2025-RMM-884291",
            "Zahlungsart:": "PayPal",
            "Event-Datum:": "12.07.2025",
        },
        "footer": [
            "CTS EVENTIM AG & Co. KGaA\nContrescarpe 75A\n28195 Bremen",
            "AG Bremen HRA 20499\nGF: Klaus-Peter Schulenberg\nUSt-IdNr.: DE 811 992 459",
            "Kundenservice:\n01806 / 570 070\neventim.de/help",
        ],
    },
]


# ---------------------------------------------------------------------------
# Generatoren pro Dokumenttyp
# ---------------------------------------------------------------------------

def _gen_telekom(pdf: BusinessPDF, doc: dict):
    pdf.add_address_block()
    pdf.add_info_block()
    pdf.add_subject("Ihre Mobilfunkrechnung - Februar 2025")
    pdf.add_paragraph(
        "Sehr geehrter Herr Mustermann,\n\n"
        "anbei erhalten Sie Ihre Mobilfunkrechnung fuer den Abrechnungszeitraum "
        "01.02.2025 bis 28.02.2025."
    )
    pdf.add_table(
        headers=["Position", "Beschreibung", "Betrag"],
        col_widths=[25, 120, 45],
        rows=[
            ["1", "Monatlicher Grundpreis MagentaMobil L", "39,95 EUR"],
            ["2", "Zusatzoptionen (StreamOn Music)", "0,00 EUR"],
            ["3", "Verbindungsentgelte Inland", "2,18 EUR"],
            ["4", "Verbindungsentgelte Ausland", "1,29 EUR"],
        ],
        subtotal_rows=[
            ["Nettobetrag:", "36,49 EUR"],
            ["USt. 19%:", "6,93 EUR"],
        ],
        total_row=["Gesamtbetrag (brutto):", "43,42 EUR"],
    )
    pdf.ln(4)
    pdf.add_highlight_box("Abbuchung am 15.03.2025 von IBAN DE89 3704 0044 0532 0130 00", "info")
    pdf.add_paragraph(
        "Haben Sie Fragen zu Ihrer Rechnung? Rufen Sie uns an unter 0800 330 1000 (kostenfrei) "
        "oder besuchen Sie unser Kundencenter unter telekom.de/kundencenter.\n\n"
        "Mit freundlichen Gruessen\nIhre Telekom"
    )


def _gen_amazon(pdf: BusinessPDF, doc: dict):
    pdf.add_address_block()
    pdf.add_info_block()
    pdf.add_subject("Rechnung zu Ihrer Bestellung 302-4418823-9912654")
    pdf.add_paragraph("Lieferadresse: Max Mustermann, Musterstrasse 42, 01099 Dresden")
    pdf.add_table(
        headers=["Pos.", "Artikel", "Menge", "Einzelpreis", "Gesamt"],
        col_widths=[15, 95, 18, 30, 32],
        rows=[
            ["1", "Anker USB-C Hub 7-in-1 Aluminium", "1", "34,99 EUR", "34,99 EUR"],
            ["2", "UGREEN Cat 7 Netzwerkkabel 3m", "2", "4,49 EUR", "8,98 EUR"],
        ],
        subtotal_rows=[
            ["Zwischensumme (netto):", "36,95 EUR"],
            ["Versandkosten:", "0,00 EUR"],
            ["USt. 19%:", "7,02 EUR"],
        ],
        total_row=["Rechnungsbetrag:", "51,97 EUR"],
    )
    pdf.ln(3)
    pdf.add_highlight_box("Zahlung erhalten am 14.01.2025 via Visa **** 4821", "success")
    pdf.add_paragraph(
        "Vielen Dank fuer Ihren Einkauf bei Amazon.de.\n\n"
        "Retouren koennen Sie innerhalb von 30 Tagen ueber Ihr Amazon-Kundenkonto einleiten."
    )


def _gen_stadtwerke(pdf: BusinessPDF, doc: dict):
    pdf.add_address_block()
    pdf.add_info_block()
    pdf.add_subject("Jahresabrechnung Strom 2024")
    pdf.add_paragraph(
        "Sehr geehrter Herr Mustermann,\n\n"
        "hiermit erhalten Sie Ihre Jahresabrechnung fuer Strom fuer den Zeitraum "
        "01.01.2024 bis 31.12.2024."
    )
    pdf.add_table(
        headers=["Zaehlerstand", "Datum", "kWh"],
        col_widths=[80, 55, 55],
        rows=[
            ["Zaehlerstand alt", "01.01.2024", "48.291 kWh"],
            ["Zaehlerstand neu", "31.12.2024", "50.847 kWh"],
            ["Verbrauch", "", "2.556 kWh"],
        ],
    )
    pdf.ln(2)
    pdf.add_table(
        headers=["Position", "Berechnung", "Betrag"],
        col_widths=[60, 80, 50],
        rows=[
            ["Arbeitspreis", "2.556 kWh x 0,3142 EUR/kWh", "803,09 EUR"],
            ["Grundpreis", "12 Monate x 12,50 EUR", "150,00 EUR"],
        ],
        subtotal_rows=[
            ["Gesamtbetrag (brutto):", "953,09 EUR"],
            ["Geleistete Abschlaege:", "-900,00 EUR"],
        ],
        total_row=["Nachzahlung:", "53,09 EUR"],
    )
    pdf.ln(3)
    pdf.add_highlight_box("Nachzahlung 53,09 EUR wird am 28.02.2025 per SEPA eingezogen.", "warning")
    pdf.add_paragraph(
        "Ihr neuer monatlicher Abschlag ab 01.03.2025: 82,00 EUR\n\n"
        "Mit freundlichen Gruessen\nIhr SachsenEnergie-Team"
    )


def _gen_jetbrains(pdf: BusinessPDF, doc: dict):
    pdf.add_address_block()
    pdf.add_info_block()
    pdf.add_subject("Invoice - IntelliJ IDEA Ultimate Subscription")
    pdf.add_paragraph("Bill to: Max Mustermann, Dresden, Germany")
    pdf.add_table(
        headers=["Item", "Period", "Qty", "Amount"],
        col_widths=[70, 55, 20, 45],
        rows=[
            ["IntelliJ IDEA Ultimate (Year 3)", "02/2025 - 01/2026", "1", "149,00 EUR"],
            ["Continuity Discount (-20%)", "", "", "-29,80 EUR"],
        ],
        subtotal_rows=[
            ["Subtotal:", "119,20 EUR"],
            ["VAT 19% (reverse charge):", "0,00 EUR"],
        ],
        total_row=["Total:", "119,20 EUR"],
    )
    pdf.ln(3)
    pdf.add_highlight_box("Payment received 01.02.2025 - Credit Card ending 4821", "success")
    pdf.add_paragraph(
        "Thank you for your continued subscription to JetBrains products!\n\n"
        "Your license key has been automatically activated. You can manage your "
        "subscriptions at account.jetbrains.com."
    )


def _gen_hetzner(pdf: BusinessPDF, doc: dict):
    pdf.add_address_block()
    pdf.add_info_block()
    pdf.add_subject("Rechnung - Cloud Services Januar 2025")
    pdf.add_table(
        headers=["Pos.", "Service", "Details", "Betrag"],
        col_widths=[15, 65, 70, 40],
        rows=[
            ["1", "Cloud Server CX21", "2 vCPU, 4 GB RAM, 40 GB SSD, 31 Tage", "5,83 EUR"],
            ["2", "Snapshot", "20 GB Snapshot-Speicher", "0,24 EUR"],
            ["3", "Automated Backup", "Taegliches Backup (31 Tage)", "1,17 EUR"],
        ],
        subtotal_rows=[
            ["Netto:", "7,24 EUR"],
            ["USt. 19%:", "1,38 EUR"],
        ],
        total_row=["Gesamtbetrag:", "8,62 EUR"],
    )
    pdf.ln(3)
    pdf.add_highlight_box("Abbuchung per SEPA-Lastschrift von Ihrem Konto.", "info")
    pdf.add_paragraph(
        "Ihre Server-Uebersicht finden Sie im Hetzner Cloud Console unter\n"
        "console.hetzner.cloud. Bei Fragen wenden Sie sich an support@hetzner.com."
    )


def _gen_huk(pdf: BusinessPDF, doc: dict):
    pdf.add_address_block()
    pdf.add_info_block()
    pdf.add_subject("Beitragsrechnung Kfz-Versicherung 2025")
    pdf.add_paragraph(
        "Sehr geehrter Herr Mustermann,\n\n"
        "nachfolgend erhalten Sie Ihre Beitragsrechnung fuer die Kfz-Versicherung "
        "Ihres Fahrzeugs VW Polo 1.0 TSI (Kennzeichen: DD-MM 123)."
    )
    pdf.add_table(
        headers=["Versicherungsart", "Selbstbeteiligung", "Jahresbeitrag"],
        col_widths=[80, 55, 55],
        rows=[
            ["Kfz-Haftpflicht", "-", "287,40 EUR"],
            ["Teilkasko", "150,00 EUR", "124,80 EUR"],
            ["Schutzbrief", "-", "18,90 EUR"],
        ],
        total_row=["Jahresbeitrag gesamt:", "431,10 EUR"],
    )
    pdf.ln(3)
    pdf.add_paragraph(
        "Schadenfreiheitsklasse: SF 8\n"
        "Zahlungsweise: halbjaehrlich (215,55 EUR je Rate)\n"
        "Naechste Abbuchung: 01.07.2025"
    )
    pdf.add_highlight_box("Erste Rate 215,55 EUR abgebucht am 01.01.2025", "success")
    pdf.add_paragraph("Mit freundlichen Gruessen\nIhre HUK-COBURG")


def _gen_wbs(pdf: BusinessPDF, doc: dict):
    pdf.add_info_block()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, _safe("GEHALTSABRECHNUNG"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.add_paragraph(
        "Mitarbeiter: Max Mustermann\n"
        "Personalnummer: WBS-2025-0842\n"
        "Steuerklasse: I / 0,5 Kinderfreibetrag\n"
        "Konfession: ev\n"
        "Krankenkasse: AOK PLUS (15,5% + 1,3% Zusatzbeitrag)"
    )
    pdf.add_table(
        headers=["Lohnart", "Bezeichnung", "Betrag"],
        col_widths=[30, 115, 45],
        rows=[
            ["1000", "Grundgehalt", "3.800,00 EUR"],
            ["1020", "Vermoegenswirksame Leistungen (AG-Anteil)", "40,00 EUR"],
        ],
        total_row=["Gesamtbrutto:", "3.840,00 EUR"],
    )
    pdf.ln(2)
    pdf.add_table(
        headers=["Abzug", "Bezeichnung", "Betrag"],
        col_widths=[30, 115, 45],
        rows=[
            ["", "Lohnsteuer", "-542,16 EUR"],
            ["", "Solidaritaetszuschlag", "0,00 EUR"],
            ["", "Kirchensteuer", "-48,74 EUR"],
            ["", "KV-Beitrag (AOK PLUS, AN-Anteil)", "-302,88 EUR"],
            ["", "RV-Beitrag (AN-Anteil)", "-357,12 EUR"],
            ["", "AV-Beitrag (AN-Anteil)", "-49,92 EUR"],
            ["", "PV-Beitrag (AN-Anteil)", "-65,28 EUR"],
        ],
        total_row=["Nettoverdienst:", "2.473,90 EUR"],
    )
    pdf.ln(3)
    pdf.add_highlight_box("Auszahlung 2.473,90 EUR am 31.01.2025 auf IBAN DE89 ...0130 00", "success")


def _gen_augenarzt(pdf: BusinessPDF, doc: dict):
    pdf.add_address_block()
    pdf.add_info_block()
    pdf.add_subject("Befundbericht - Augenarztliche Routineuntersuchung")
    pdf.add_paragraph(
        "Patient: Max Mustermann, geb. 01.01.1985\n"
        "Untersuchungsdatum: 22.01.2025\n"
        "Anlass: Routineuntersuchung / Vorsorge"
    )
    pdf.add_table(
        headers=["Untersuchung", "Rechtes Auge", "Linkes Auge", "Bewertung"],
        col_widths=[50, 40, 40, 60],
        rows=[
            ["Visus (ohne Korrektur)", "1,0", "0,8", "Normbereich"],
            ["Augeninnendruck", "14 mmHg", "15 mmHg", "Normbereich (10-21)"],
            ["Spaltlampe", "reizfrei", "reizfrei", "unauffaellig"],
            ["CDR (Papille)", "0,3", "0,3", "Normbereich"],
            ["Makula", "intakt", "intakt", "unauffaellig"],
        ],
    )
    pdf.ln(3)
    pdf.add_highlight_box("Beurteilung: Altersentsprechender Normalbefund", "success")
    pdf.add_paragraph(
        "Leichte Myopie am linken Auge, aktuell keine Korrektur notwendig.\n\n"
        "Empfehlung: Kontrolluntersuchung in 12 Monaten."
    )
    pdf.add_signature("Dresden, 22.01.2025", "Dr. med. Katharina Lehmann", "Facharztin fuer Augenheilkunde")


def _gen_scalable(pdf: BusinessPDF, doc: dict):
    pdf.add_address_block()
    pdf.add_info_block()
    pdf.add_subject("Depotauszug per 31.01.2025")
    pdf.add_paragraph("Depotinhaber: Max Mustermann | Depotnummer: SC-DE-2847193 | Depotbank: Baader Bank AG")
    pdf.add_table(
        headers=["ISIN", "Fondsname", "Anteile", "Kurs EUR", "Wert EUR"],
        col_widths=[32, 62, 24, 32, 40],
        rows=[
            ["IE00B4L5Y983", "iShares Core MSCI World UCITS ETF", "42,381", "90,78", "3.847,12"],
            ["IE00B3RBWM25", "Vanguard FTSE All-World UCITS ETF", "18,200", "115,52", "2.102,44"],
            ["DE0005933931", "iShares Core DAX UCITS ETF", "5,000", "178,47", "892,35"],
        ],
        total_row=["Gesamtwert Depot:", "6.841,91 EUR"],
    )
    pdf.ln(3)
    pdf.add_highlight_box("Wertentwicklung seit Eroeffnung: +12,4% | Sparplan: 250 EUR/Monat", "info")
    pdf.add_paragraph(
        "Naechste Sparplan-Ausfuehrung: 01.02.2025\n\n"
        "Hinweis: Dieses Dokument dient ausschliesslich zu Informationszwecken. "
        "Es stellt keine steuerliche Bescheinigung dar. Ihre Jahressteuerbescheinigung "
        "erhalten Sie gesondert."
    )


def _gen_drk(pdf: BusinessPDF, doc: dict):
    pdf.add_address_block()
    pdf.add_info_block()
    pdf.set_font("Helvetica", "B", 16)
    r, g, b = pdf.brand_rgb
    pdf.set_text_color(r, g, b)
    pdf.cell(0, 10, _safe("MITGLIEDSBESCHEINIGUNG"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)
    pdf.add_paragraph(
        "Hiermit bescheinigen wir, dass\n\n"
        "    Max Mustermann, geb. 01.01.1985\n\n"
        "seit dem 01.04.2018 aktives Mitglied des DRK Kreisverband Dresden e.V. ist."
    )
    pdf.add_table(
        headers=["Angabe", "Details"],
        col_widths=[70, 120],
        rows=[
            ["Mitgliedsnummer", "DRK-DD-2018-4291"],
            ["Einsatzgruppe", "Wasserwacht Dresden-Nord"],
            ["Qualifikation 1", "Rettungsschwimmer Silber (DRSA)"],
            ["Qualifikation 2", "Erste-Hilfe-Ausbilder (BG-zertifiziert)"],
            ["Jahresbeitrag 2025", "48,00 EUR (eingezogen am 15.01.2025)"],
        ],
    )
    pdf.ln(4)
    pdf.add_paragraph(
        "Diese Bescheinigung wird zur Vorlage bei Behoerden, "
        "Arbeitgebern und Versicherungen ausgestellt."
    )
    pdf.add_signature("Dresden, 20.01.2025", "Thomas Mueller", "Geschaeftsfuehrer DRK KV Dresden e.V.")


def _gen_vonovia(pdf: BusinessPDF, doc: dict):
    pdf.add_address_block()
    pdf.add_info_block()
    pdf.add_subject("Mietanpassung zum 01.04.2025")
    pdf.add_paragraph(
        "Sehr geehrter Herr Mustermann,\n\n"
        "hiermit teilen wir Ihnen die Anpassung Ihrer monatlichen Miete "
        "gemaess Paragraph 558 BGB mit."
    )
    pdf.add_table(
        headers=["Position", "Bisher", "Neu ab 01.04.2025"],
        col_widths=[80, 55, 55],
        rows=[
            ["Kaltmiete", "485,00 EUR", "512,00 EUR"],
            ["Betriebskosten-VZ", "120,00 EUR", "120,00 EUR"],
            ["Heizkosten-VZ", "60,00 EUR", "60,00 EUR"],
        ],
        total_row=["Gesamtmiete (warm):", "692,00 EUR"],
    )
    pdf.ln(3)
    pdf.add_highlight_box("Erhoehung Kaltmiete: +27,00 EUR (+5,6%) | Neue Gesamtmiete: 692,00 EUR", "warning")
    pdf.add_paragraph(
        "Rechtsgrundlage: Paragraph 558 BGB (ortsueblicher Mietspiegel)\n"
        "Der Dresdner Mietspiegel 2024 weist fuer vergleichbare Wohnungen "
        "(Baujahr, Lage, Ausstattung) eine Spanne von 7,20 - 9,80 EUR/m2 aus. "
        "Die neue Kaltmiete von 8,19 EUR/m2 liegt innerhalb dieser Spanne.\n\n"
        "Bitte ueberweisen Sie den neuen Betrag ab dem 01.04.2025.\n\n"
        "Mit freundlichen Gruessen\nVonovia Kundenservice"
    )


def _gen_aok(pdf: BusinessPDF, doc: dict):
    pdf.add_address_block()
    pdf.add_info_block()
    pdf.add_subject("Bonusprogramm - Jahresabrechnung 2024")
    pdf.add_paragraph(
        "Sehr geehrter Herr Mustermann,\n\n"
        "herzlichen Glueckwunsch! Hier ist Ihre Bonuspunkte-Abrechnung fuer das Jahr 2024."
    )
    pdf.add_table(
        headers=["Gesundheitsmassnahme", "Datum", "Punkte"],
        col_widths=[90, 50, 50],
        rows=[
            ["Zahnaerztliche Vorsorge", "15.03.2024", "200"],
            ["Hautkrebsscreening", "22.05.2024", "150"],
            ["Gesundheits-Check-up (ab 35)", "10.07.2024", "300"],
            ["Sportverein (DRK Wasserwacht)", "ganzjaehrig", "250"],
            ["Blutspende (2 Spenden)", "04/2024, 10/2024", "200"],
        ],
        total_row=["Gesamtpunkte 2024:", "1.100"],
    )
    pdf.ln(3)
    pdf.add_highlight_box("Ihre Praemie: 65,00 EUR - Ueberweisung bis 28.02.2025", "success")
    pdf.add_paragraph(
        "Tipp: Ab 1.500 Punkten erhalten Sie im naechsten Jahr die Gold-Praemie "
        "von 120,00 EUR!\n\n"
        "Bleiben Sie gesund!\nIhre AOK PLUS"
    )


def _gen_steuerbescheid(pdf: BusinessPDF, doc: dict):
    pdf.add_address_block()
    pdf.add_info_block()
    r, g, b = pdf.brand_rgb
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(r, g, b)
    pdf.cell(0, 10, _safe("EINKOMMENSTEUERBESCHEID 2023"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)
    pdf.add_paragraph(
        "Auf Grundlage Ihrer Einkommensteuererklaerung wird die Einkommensteuer "
        "fuer das Kalenderjahr 2023 wie folgt festgesetzt:"
    )
    pdf.add_table(
        headers=["Position", "Betrag"],
        col_widths=[130, 60],
        rows=[
            ["Einnahmen aus nichtselbstaendiger Arbeit", "45.600,00 EUR"],
            ["Werbungskosten (Entfernungspauschale, Arbeitsmittel)", "-1.230,00 EUR"],
            ["Sonderausgaben (Vorsorge, Spenden)", "-2.847,00 EUR"],
            ["Vorsorgeaufwendungen", "-4.512,00 EUR"],
        ],
        total_row=["Zu versteuerndes Einkommen:", "37.011,00 EUR"],
    )
    pdf.ln(2)
    pdf.add_table(
        headers=["Steuerberechnung", "Betrag"],
        col_widths=[130, 60],
        rows=[
            ["Festgesetzte Einkommensteuer", "6.842,00 EUR"],
            ["Solidaritaetszuschlag", "0,00 EUR"],
            ["Bereits einbehaltene Lohnsteuer", "-7.124,00 EUR"],
        ],
        total_row=["Erstattungsbetrag:", "282,00 EUR"],
    )
    pdf.ln(3)
    pdf.add_highlight_box("Erstattung 282,00 EUR auf IBAN DE89 3704 0044 0532 0130 00", "success")
    pdf.add_paragraph(
        "Rechtsbehelfsbelehrung:\n"
        "Gegen diesen Bescheid kann innerhalb eines Monats nach Bekanntgabe "
        "schriftlich oder elektronisch Einspruch beim Finanzamt Dresden-Nord eingelegt werden."
    )


def _gen_fahrschule(pdf: BusinessPDF, doc: dict):
    pdf.add_address_block()
    pdf.add_info_block()
    pdf.add_subject("Rechnung - Aufstieg Klasse A1 auf A2")
    pdf.add_paragraph("Fahrschueler: Max Mustermann\nFuehrerscheinklasse: A2 (Aufstieg von A1)")
    pdf.add_table(
        headers=["Pos.", "Leistung", "Anzahl", "Einzelpreis", "Gesamt"],
        col_widths=[15, 80, 20, 35, 40],
        rows=[
            ["1", "Grundbetrag Aufstieg A1 -> A2", "1", "280,00 EUR", "280,00 EUR"],
            ["2", "Uebungsfahrt (je 45 min)", "4", "55,00 EUR", "220,00 EUR"],
            ["3", "Sonderfahrt Autobahn (je 45 min)", "3", "65,00 EUR", "195,00 EUR"],
            ["4", "Pruefungsvorbereitung (Theorie)", "1", "60,00 EUR", "60,00 EUR"],
            ["5", "Praktische Pruefung (TUeV-Gebuehr)", "1", "146,56 EUR", "146,56 EUR"],
        ],
        subtotal_rows=[
            ["Gesamtbetrag:", "901,56 EUR"],
            ["Bereits gezahlte Anzahlung:", "-300,00 EUR"],
        ],
        total_row=["Restbetrag:", "601,56 EUR"],
    )
    pdf.ln(3)
    pdf.add_highlight_box("Bitte ueberweisen Sie 601,56 EUR bis zum 28.02.2025", "warning")
    pdf.add_paragraph(
        "Bankverbindung: Ostsaechsische Sparkasse Dresden\n"
        "IBAN: DE12 8505 0300 0221 1234 56\n"
        "Verwendungszweck: FS-2025-0294 / Mustermann\n\n"
        "Vielen Dank und allzeit gute Fahrt!\nFahrschule Schmidt"
    )


def _gen_eventim(pdf: BusinessPDF, doc: dict):
    pdf.add_address_block()
    pdf.add_info_block()
    pdf.add_subject("Buchungsbestaetigung - Rammstein Europa Stadion Tour 2025")
    pdf.add_highlight_box("Event: Rammstein | 12.07.2025 | Rudolf-Harbig-Stadion, Dresden", "info")
    pdf.ln(1)
    pdf.add_paragraph(
        "Einlass: 17:00 Uhr | Beginn: 20:00 Uhr\n"
        "Veranstalter: Rammstein GbR c/o Pilgrim Management GmbH"
    )
    pdf.add_table(
        headers=["Pos.", "Beschreibung", "Anzahl", "Einzelpreis", "Gesamt"],
        col_widths=[15, 80, 20, 35, 40],
        rows=[
            ["1", "Stehplatz Innenraum (General Admission)", "1", "109,00 EUR", "109,00 EUR"],
            ["2", "Servicegebuehr & Vorverkauf", "1", "12,90 EUR", "12,90 EUR"],
        ],
        total_row=["Gesamtbetrag:", "121,90 EUR"],
    )
    pdf.ln(3)
    pdf.add_highlight_box("Bezahlt am 18.01.2025 via PayPal (max.mustermann@example.de)", "success")
    pdf.add_paragraph(
        "Ihr E-Ticket wurde an Ihre E-Mail-Adresse gesendet. Bitte bringen Sie "
        "einen gueltigen Lichtbildausweis zur Veranstaltung mit.\n\n"
        "Hinweise:\n"
        "- Das Ticket ist personalisiert und nicht uebertragbar.\n"
        "- Einlass nur mit gueltigem Ticket + Ausweis.\n"
        "- Weitere Infos: eventim.de/help"
    )


# Generator-Mapping
GENERATORS = {
    "rechnung_telekom_2025.pdf": _gen_telekom,
    "rechnung_amazon_januar.pdf": _gen_amazon,
    "rechnung_stadtwerke_dresden.pdf": _gen_stadtwerke,
    "rechnung_jetbrains_2025.pdf": _gen_jetbrains,
    "rechnung_hetzner_hosting.pdf": _gen_hetzner,
    "kfz_versicherung_huk.pdf": _gen_huk,
    "gehaltsabrechnung_wbs_januar.pdf": _gen_wbs,
    "befund_augenarzt_2025.pdf": _gen_augenarzt,
    "kontoauszug_scalable_capital.pdf": _gen_scalable,
    "mitgliedsbescheinigung_drk.pdf": _gen_drk,
    "mietvertrag_aenderung_2025.pdf": _gen_vonovia,
    "aok_plus_bonusheft_2024.pdf": _gen_aok,
    "steuerbescheid_2023.pdf": _gen_steuerbescheid,
    "rechnung_fahrschule_update.pdf": _gen_fahrschule,
    "ticket_rammstein_2025.pdf": _gen_eventim,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_pdf(doc: dict, output_dir: str) -> str:
    pdf = BusinessPDF(doc)
    pdf.add_page()

    gen_fn = GENERATORS.get(doc["filename"])
    if gen_fn:
        gen_fn(pdf, doc)

    filepath = os.path.join(output_dir, doc["filename"])
    pdf.output(filepath)
    return filepath


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generiere {len(DOCUMENTS)} professionelle Test-Dokumente in: {OUTPUT_DIR}\n")

    for doc in DOCUMENTS:
        path = generate_pdf(doc, OUTPUT_DIR)
        brand = doc.get("brand_color", "")
        print(f"  [OK] {doc['filename']:45s}  {brand}  {doc['firma']}")

    print(f"\nFertig! {len(DOCUMENTS)} PDFs erstellt in:\n  {OUTPUT_DIR}")
    print("\nDiese Dateien kannst du jetzt in Paperless-NGX hochladen.")


if __name__ == "__main__":
    main()
