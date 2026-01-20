import os
import re
import json
import hashlib
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dotenv import load_dotenv

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from PIL import Image

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env") 

logger = logging.getLogger("GridVision_Enrichment")

HALLUCINATION_EXACT_NAMES = {"unknown", "component", "n/a", "not visible", ""}
HALLUCINATION_SUBSTRINGS = ("not visible", "n/a")

RE_HYPHEN_BREAK = re.compile(r'(\w+)-\s*\n\s*(\w+)') 
RE_WHITESPACE = re.compile(r'\s+')
RE_INNER_WS = re.compile(r"[ \t]+")

BASE_SCORE = 0.5

TECH_TERM_BOOST_PER_MATCH = 0.1
TECH_TERM_MAX_BOOST = 0.4

LOW_WORD_COUNT_THRESHOLD = 10
LOW_WORD_PENALTY = 0.2

LOW_OCR_CONF_THRESHOLD = 0.8
LOW_OCR_PENALTY = 0.3

ILLEGIBLE_PENALTY = 0.4
ILLEGIBLE_PATTERNS = ("not legible", "unreadable", "blur", "unknown")

TARGET_TYPES = ("text", "ocr_text", "table_md", "image_caption")

RE_TECH_TERMS = [
    r'\b\d{3}-\d{4}\b',                  # Standard PN (123-4567)
    r'\b[A-Z]{1,4}-\d+\b',               # Pin IDs (PIN-4, WH-12)
    r'\bTB\s*-?\s*\d+\b',                # Terminal Blocks (TB-1, TB 1)
    r'\b\d{2,}[A-Z]\d{1,}\b',            # Wire Codes (130A16 - common in schematics)
    r'\b[JKP]\d{1,3}\b',                 # Connectors (J1, P2, K16)
    r'\b(?=[A-Z0-9]{5,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b'
]

class TextNormalizer:
    @staticmethod
    def normalize(text: str) -> str:
        if not text: 
            return ""
        text = text.replace("\u00ad", "")
        text = RE_HYPHEN_BREAK.sub(r'\1\2', text)
        text = RE_WHITESPACE.sub(' ', text).strip()
        
        return text

    @staticmethod
    def normalize_keep_paragraphs(text: str) -> str:
        if not text:
            return ""
        text = text.replace("\u00ad", "")
        text = RE_HYPHEN_BREAK.sub(r"\1\2", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [RE_INNER_WS.sub(" ", ln).strip() for ln in text.split("\n")]
        out = "\n".join(lines)
        out = re.sub(r"\n{3,}", "\n\n", out).strip()
        return out


    @staticmethod
    def compute_hash(content: str) -> str:
        norm = TextNormalizer.normalize(content)
        return hashlib.sha256(norm.encode('utf-8')).hexdigest()


class QualityScorer:
    @staticmethod
    def evaluate(content: str, element_type: str, ocr_conf: float = 1.0) -> Tuple[float, List[str]]:
        score = BASE_SCORE
        reasons: List[str] = []

        matches = 0
        for pattern in RE_TECH_TERMS:
            matches += len(re.findall(pattern, content))

        if matches > 0:
            boost = min(
                matches * TECH_TERM_BOOST_PER_MATCH,
                TECH_TERM_MAX_BOOST,
            )
            score += boost
            reasons.append(f"tech_terms_found (+{boost:.1f})")

        word_count = len(content.split())
        if word_count < LOW_WORD_COUNT_THRESHOLD and element_type in TARGET_TYPES:
            score -= LOW_WORD_PENALTY
            reasons.append(f"low_word_count ({word_count})")

        if element_type == "ocr_text" and ocr_conf < LOW_OCR_CONF_THRESHOLD:
            score -= LOW_OCR_PENALTY
            reasons.append("low_ocr_conf")

        if any(w in content.lower() for w in ILLEGIBLE_PATTERNS):
            score -= ILLEGIBLE_PENALTY
            reasons.append("illegible_content")

        final_score = max(0.0, min(1.0, score))
        return round(final_score, 2), reasons


class ImageEnricher:

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash", max_retries: int = 3):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.max_retries = max_retries
        
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY missing. Enrichment disabled.")
            self.model = None
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)

    def _clean_json_response(self, text: str) -> str:
        s = text.strip()

        if s.startswith("```"):
            s = re.sub(r"^\s*```(?:json)?\s*\n?", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\n?\s*```\s*$", "", s)
            s = s.strip()
        try:
            json.loads(s)
            return s
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        start_candidates = [i for i in (s.find("{"), s.find("[")) if i != -1]
        if not start_candidates:
            return s

        start = min(start_candidates)
        tail = s[start:]

        try:
            _, end = decoder.raw_decode(tail)
            return tail[:end].strip()
        except json.JSONDecodeError:
            return tail.strip()



    def _render_caption_template(self, data: Dict) -> str:
        comp_name = data.get("component_name", "Unknown Component")
        type_str = data.get("visual_type", "Diagram")
        parts = ", ".join(data.get("part_numbers", []))
        conns = ", ".join(data.get("connections", []))
        
        caption = f"{type_str} for {comp_name}."
        if parts: caption += f" Part Numbers: {parts}."
        if conns: caption += f" Connects to: {conns}."
        if "technical_description" in data:
            caption += f" Details: {data['technical_description']}"
            
        return caption

    def _try_parse_json(self, s: str) -> Dict:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            s2 = s.strip()
            s2 = re.sub(r",\s*([}\]])", r"\1", s2)
            s2 = s2.replace("“", '"').replace("”", '"').replace("’", "'")
            return json.loads(s2)


    def _is_useful_data(self, data: Dict) -> bool:
        name = str(data.get("component_name", "")).lower().strip()
        parts = data.get("part_numbers") or []
        conns = data.get("connections") or []

        is_placeholder = (name in HALLUCINATION_EXACT_NAMES) or any(s in name for s in HALLUCINATION_SUBSTRINGS)
        if is_placeholder and not parts and not conns:
            return False
        return True

    def enrich_image(self, image_path: Path) -> Tuple[Optional[str], Optional[Dict]]:
        if not self.model or not image_path.exists():
            return None, None

        raw_text = self._call_model_with_retry(image_path)
        if raw_text is None:
            return None, None

        data = self._parse_and_validate_enrichment(raw_text, image_path)
        if not data:
            return None, None

        return self._render_caption_template(data), data

    def _build_image_prompt(self) -> str:
        return """
        You are extracting structured information from a technical manual image.
        Return ONLY a valid JSON object.
        - No markdown, no code fences, no commentary, no extra text.
        - The response MUST start with { and end with }.
        - Use double quotes for all keys and string values.
        - If a field is unknown, use "" for strings and [] for lists.
        Schema (return exactly these keys):
        {
        "component_name": "",
        "visual_type": "Schematic|Photo|Chart|Table|Diagram|Other",
        "part_numbers": [],
        "connections": [],
        "technical_description": ""
        }
        Rules:
        - "component_name": a specific component name if visible (e.g., "Fuel Pump", "K1 Relay"). If not visible, "".
        - "visual_type": pick ONE value from the allowed list.
        - "part_numbers": only part numbers explicitly visible in the image (strings).
        - "connections": only pin/wire/connector IDs explicitly visible (strings).
        - "technical_description": max 2 sentences, factual, based only on visible content. If not enough info, "".
        """.strip()

    def _call_model_with_retry(self, image_path: Path) -> Optional[str]:
        prompt = self._build_image_prompt()

        for attempt in range(self.max_retries):
            try:
                with Image.open(image_path) as img:
                    response = self.model.generate_content([prompt, img])
                return (response.text or "").strip()

            except google_exceptions.ResourceExhausted:
                wait = self._rate_limit_backoff_seconds(attempt)
                logger.warning(f"Rate limit. Waiting {wait}s...")
                time.sleep(wait)

            except Exception as e:
                logger.error(f"Enrichment failed for {image_path.name}: {e}")
                break

        return None

    def _rate_limit_backoff_seconds(self, attempt: int) -> int:
        return (2**attempt) * 5

    def _parse_and_validate_enrichment(self, raw_text: str, image_path: Path) -> Optional[Dict]:
        try:
            cleaned = self._clean_json_response(raw_text)
            data = self._try_parse_json(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"JSON Decode Error on {image_path.name}. Skipping.")
            return None

        if not data or not self._is_useful_data(data):
            logger.info(f"Skipping low-value caption for {image_path.name}")
            return None

        return data

if __name__ == "__main__":
    test_str = "The connec-\r\n tion to 130A16 wire is loose."
    print(f"Norm: '{TextNormalizer.normalize(test_str)}'")
    
    test_content = "Check wire 130A16 at J1 connector."
    s, r = QualityScorer.evaluate(test_content, "text")
    print(f"Score: {s}, Reasons: {r}")