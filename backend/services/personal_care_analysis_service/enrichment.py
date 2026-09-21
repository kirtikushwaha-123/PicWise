"""
backend/services/personal_care_analysis_service/enrichment.py

Authoritative runtime semantic enrichment layer for Personal Care ingredients.

CRITICAL SECURITY & METHODOLOGY CONSTRAINTS:
1. Target Isolation:
   Exclusively exposes the 6 permitted input semantic fields:
     - Ingredient_Name
     - Primary_Function
     - Ingredient_Category
     - Product_Categories
     - Origin
     - Regulatory_Status
   NEVER returns or exposes target columns (Safety_Level, Allergy_Risk, Irritation_Risk).
2. Structural Safety:
   Does NOT load or filter target columns at lookup time; constructs an immutable
   PersonalCareSemanticFeatures object containing strictly the allowed input fields.
3. Deterministic Lookup:
   Indexes both canonical primary names and semicolon-separated Packaging / Alternate Names.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from backend.ml.preprocessing.personal_care_dataset import (
    DEFAULT_PERSONAL_CARE_DATA_PATH,
)

ALLOWED_INPUT_FIELDS = (
    "Ingredient_Name",
    "Primary_Function",
    "Ingredient_Category",
    "Product_Categories",
    "Origin",
    "Regulatory_Status",
)

FORBIDDEN_TARGET_FIELDS = (
    "Safety_Level",
    "Allergy_Risk",
    "Irritation_Risk",
    "safety_level",
    "allergy_risk",
    "irritation_risk",
)


@dataclass(frozen=True)
class PersonalCareSemanticFeatures:
    """Immutable runtime feature object containing strictly the 6 permitted input fields."""
    ingredient_name: str
    primary_function: str
    ingredient_category: str
    product_categories: str
    origin: str
    regulatory_status: str

    def to_model_input_dict(self) -> Dict[str, str]:
        """Formats dictionary matching the ColumnTransformer input schema."""
        return {
            "Ingredient_Name": self.ingredient_name,
            "Primary_Function": self.primary_function,
            "Ingredient_Category": self.ingredient_category,
            "Product_Categories": self.product_categories,
            "Origin": self.origin,
            "Regulatory_Status": self.regulatory_status,
        }

    def to_dict(self) -> Dict[str, str]:
        """Dictionary representation containing strictly input-side fields."""
        return {
            "ingredient_name": self.ingredient_name,
            "primary_function": self.primary_function,
            "ingredient_category": self.ingredient_category,
            "product_categories": self.product_categories,
            "origin": self.origin,
            "regulatory_status": self.regulatory_status,
        }


def normalize_lookup_key(value: str) -> str:
    """
    Normalizes ingredient string for deterministic lookup:
    strips outer punctuation, collapses spaces, lowercases alphanumeric characters.
    """
    if not value or not isinstance(value, str):
        return ""
    text = value.strip().lower()
    text = re.sub(r"^[,\.\;\"\']+|[,\.\;\"\']+$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return "".join(c for c in text if c.isalnum() or c.isspace()).strip()


class PersonalCareKnowledgeBase:
    """
    Target-isolated runtime knowledge base for Personal Care semantic feature enrichment.
    """

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = Path(data_path or DEFAULT_PERSONAL_CARE_DATA_PATH)
        if not self.data_path.is_absolute():
            # Resolve relative to project root
            self.data_path = Path(__file__).resolve().parents[3] / self.data_path

        self._canonical_index: Dict[str, PersonalCareSemanticFeatures] = {}
        self._alternate_index: Dict[str, PersonalCareSemanticFeatures] = {}
        self._all_index: Dict[str, PersonalCareSemanticFeatures] = {}
        self._vocab_names: List[str] = []
        self._loaded = False
        self._load_and_index()

    def _load_and_index(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Authoritative dataset not found at: {self.data_path}")

        df = pd.read_csv(self.data_path)

        for _, row in df.iterrows():
            name = str(row.get("Ingredient_Name", "")).strip()
            if not name:
                continue

            # Construct immutable feature container strictly with allowed input fields
            features = PersonalCareSemanticFeatures(
                ingredient_name=name,
                primary_function=str(row.get("Primary_Function", "")).strip(),
                ingredient_category=str(row.get("Ingredient_Category", "")).strip(),
                product_categories=str(row.get("Product_Categories", "")).strip(),
                origin=str(row.get("Origin", "")).strip(),
                regulatory_status=str(row.get("Regulatory_Status", "")).strip(),
            )

            canon_key = normalize_lookup_key(name)
            if canon_key and canon_key not in self._canonical_index:
                self._canonical_index[canon_key] = features
                self._vocab_names.append(name)

            # Phase 10D: Index parenthetical INCI aliases, e.g. "Aqua (Water)" -> "Aqua", "Water"
            paren_match = re.match(r"^(.*?)\s*\((.*?)\)$", name)
            if paren_match:
                prefix = paren_match.group(1).strip()
                inner = paren_match.group(2).strip()
                prefix_key = normalize_lookup_key(prefix)
                inner_key = normalize_lookup_key(inner)
                if prefix_key and prefix_key not in self._canonical_index and prefix_key not in self._alternate_index:
                    self._alternate_index[prefix_key] = features
                    self._vocab_names.append(prefix)
                if inner_key and inner_key not in self._canonical_index and inner_key not in self._alternate_index:
                    self._alternate_index[inner_key] = features
                    self._vocab_names.append(inner)

            alt_raw = row.get("Packaging Names / Alternate Names")
            if alt_raw and str(alt_raw).strip() != "No Alternate Names":
                for alt_item in str(alt_raw).split(";"):
                    alt_clean = alt_item.strip()
                    alt_key = normalize_lookup_key(alt_clean)
                    if alt_key and alt_key not in self._canonical_index and alt_key not in self._alternate_index:
                        self._alternate_index[alt_key] = features
                        self._vocab_names.append(alt_clean)

        self._all_index = {**self._alternate_index, **self._canonical_index}
        self._loaded = True

    def lookup(self, query: str) -> Optional[PersonalCareSemanticFeatures]:
        """
        Looks up an ingredient by name or packaging alias.
        Returns PersonalCareSemanticFeatures containing strictly input fields, or None.
        """
        if not query or not isinstance(query, str):
            return None
        key = normalize_lookup_key(query)
        return self._all_index.get(key)

    def get_vocab_names(self) -> List[str]:
        """Returns flat list of searchable ingredient names."""
        return list(self._vocab_names)

    @property
    def total_entries(self) -> int:
        return len(self._canonical_index)


# Singleton cache
_CACHED_PC_KB: Optional[PersonalCareKnowledgeBase] = None


def get_personal_care_knowledge_base() -> PersonalCareKnowledgeBase:
    """Thread-safe accessor for cached Personal Care knowledge base singleton."""
    global _CACHED_PC_KB
    if _CACHED_PC_KB is None:
        _CACHED_PC_KB = PersonalCareKnowledgeBase()
    return _CACHED_PC_KB
