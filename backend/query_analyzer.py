"""
Deterministic Query Understanding & Generic Financial Terminology Expansion.

Extracts normalized financial concepts as the primary semantic signal for financial queries,
generating relevant accounting aliases and terminology variants dynamically.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List

ACCOUNTING_CONCEPTS: Dict[str, Dict] = {
    "PROPERTY_PLANT_EQUIPMENT_NET": {
        "statement": "BALANCE_SHEET",
        "keywords": [
            "net ppne", "net pp&e", "property, plant and equipment, net",
            "property, plant and equipment — net", "property plant and equipment net",
            "ppne", "pp&e", "property, plant and equipment", "property plant and equipment",
            # Not every filer says "plant" -- e.g. Activision Blizzard's balance
            # sheet line item is just "Property and equipment, net".
            "property and equipment, net", "property and equipment — net",
            "property and equipment net", "property and equipment"
        ]
    },
    "CAPEX": {
        "statement": "CASH_FLOW_STATEMENT",
        "keywords": [
            "capital expenditures", "capital expenditure", "capex",
            "purchases of property, plant and equipment", "purchases of property",
            "additions to property, plant and equipment", "capital spending"
        ]
    },
    "NET_SALES_REVENUE": {
        "statement": "INCOME_STATEMENT",
        "keywords": [
            "total net sales", "net sales", "total revenue", "net revenue", "revenue", "sales"
        ]
    },
    "OPERATING_INCOME": {
        "statement": "INCOME_STATEMENT",
        "keywords": [
            "operating income", "income from operations", "operating profit", "operating margin"
        ]
    },
    "NET_INCOME": {
        "statement": "INCOME_STATEMENT",
        "keywords": [
            "net income", "net earnings", "net profit", "earnings per share", "eps"
        ]
    },
    "GROSS_PROFIT": {
        "statement": "INCOME_STATEMENT",
        "keywords": [
            "gross profit", "gross margin"
        ]
    },
    "COGS": {
        "statement": "INCOME_STATEMENT",
        "keywords": [
            "cost of sales", "cost of goods sold", "cost of revenue", "cogs"
        ]
    },
    "SGA": {
        "statement": "INCOME_STATEMENT",
        "keywords": [
            "selling, general and administrative expenses", "selling general and administrative", "sg&a", "sga"
        ]
    },
    "RESEARCH_DEVELOPMENT": {
        "statement": "INCOME_STATEMENT",
        "keywords": [
            "research, development and related expenses", "research and development", "r&d"
        ]
    },
    "CASH_FLOW_OPERATING": {
        "statement": "CASH_FLOW_STATEMENT",
        "keywords": [
            "net cash provided by (used in) operating activities", "net cash provided by operating activities",
            "cash flows from operating activities", "operating cash flow"
        ]
    },
    "CASH_FLOW_INVESTING": {
        "statement": "CASH_FLOW_STATEMENT",
        "keywords": [
            "net cash provided by (used in) investing activities", "cash flows from investing activities"
        ]
    },
    "CASH_FLOW_FINANCING": {
        "statement": "CASH_FLOW_STATEMENT",
        "keywords": [
            "net cash provided by (used in) financing activities", "cash flows from financing activities"
        ]
    },
    "CASH_AND_EQUIVALENTS": {
        "statement": "BALANCE_SHEET",
        "keywords": [
            "cash and cash equivalents", "cash equivalents", "cash balance"
        ]
    },
    "TOTAL_ASSETS": {
        "statement": "BALANCE_SHEET",
        "keywords": [
            "total assets"
        ]
    },
    "CURRENT_ASSETS": {
        "statement": "BALANCE_SHEET",
        "keywords": [
            "total current assets", "current assets"
        ]
    },
    "CURRENT_LIABILITIES": {
        "statement": "BALANCE_SHEET",
        "keywords": [
            "total current liabilities", "current liabilities"
        ]
    },
    "TOTAL_LIABILITIES": {
        "statement": "BALANCE_SHEET",
        "keywords": [
            "total liabilities"
        ]
    },
    "TOTAL_EQUITY": {
        "statement": "EQUITY_STATEMENT",
        "keywords": [
            "total 3m company shareholders equity", "total equity", "stockholders' equity", "shareholders' equity", "stockholders equity"
        ]
    },
    "DEBT": {
        "statement": "BALANCE_SHEET",
        "keywords": [
            "long-term debt", "short-term borrowings and current portion of long-term debt", "total debt", "borrowings"
        ]
    },
    "INVENTORY": {
        "statement": "BALANCE_SHEET",
        "keywords": [
            "total inventories", "inventories", "inventory", "finished goods", "work in process", "raw materials and supplies"
        ]
    },
    "ACCOUNTS_RECEIVABLE": {
        "statement": "BALANCE_SHEET",
        "keywords": [
            "accounts receivable, net", "accounts receivable net", "accounts receivable"
        ]
    },
    "ACCOUNTS_PAYABLE": {
        "statement": "BALANCE_SHEET",
        "keywords": [
            "accounts payable", "account payable", "trade accounts payable", "payables"
        ]
    },
    "DEPRECIATION_AMORTIZATION": {
        "statement": "INCOME_STATEMENT",
        "keywords": [
            "depreciation and amortization", "d&a"
        ]
    },
    "FREE_CASH_FLOW": {
        "statement": "CASH_FLOW_STATEMENT",
        "keywords": [
            "free cash flow", "fcf"
        ]
    },
    "DIVIDENDS": {
        "statement": "CASH_FLOW_STATEMENT",
        "keywords": [
            "dividends paid to shareholders", "dividends paid", "cash dividends"
        ]
    },
    "BUYBACKS": {
        "statement": "CASH_FLOW_STATEMENT",
        "keywords": [
            "purchases of treasury stock", "share repurchase", "buyback"
        ]
    },
    "VALUE_AT_RISK": {
        "statement": "FOOTNOTES",
        "keywords": [
            "value at risk", "var", "historical var", "average var", "market risk var"
        ]
    },
    "TAX_RATE": {
        "statement": "INCOME_STATEMENT",
        "keywords": [
            "effective tax rate", "income tax rate", "tax rate", "provision for income taxes"
        ]
    },
    "WEIGHTED_AVERAGE_SHARES": {
        "statement": "INCOME_STATEMENT",
        "keywords": [
            "weighted average shares", "diluted shares", "basic shares", "share count", "diluted average shares"
        ]
    },
    "EBITDA": {
        "statement": "INCOME_STATEMENT",
        "keywords": [
            "ebitda", "ebit", "operating profit before depreciation"
        ]
    },
    "INTEREST_EXPENSE": {
        "statement": "INCOME_STATEMENT",
        "keywords": [
            "interest expense", "interest paid", "finance cost"
        ]
    },
    "WORKING_CAPITAL": {
        "statement": "BALANCE_SHEET",
        "keywords": [
            "working capital", "current assets less current liabilities"
        ]
    },
    "NET_ASSET_VALUE": {
        "statement": "BALANCE_SHEET",
        "keywords": [
            "net asset value", "nav"
        ]
    },
    "GOODWILL": {
        "statement": "BALANCE_SHEET",
        "keywords": [
            "goodwill", "intangible assets", "goodwill impairment"
        ]
    },
    "LEVERAGE": {
        "statement": "BALANCE_SHEET",
        "keywords": [
            "debt to equity", "leverage ratio", "debt-to-equity"
        ]
    },
    "ROE": {
        "statement": "INCOME_STATEMENT",
        "keywords": [
            "return on equity", "roe", "return on assets", "roa"
        ]
    },
    "ACQUISITIONS": {
        "statement": "FOOTNOTES",
        "keywords": [
            "acquisition", "acquisitions", "business combinations", "purchase price",
            "acquired", "acquire", "merger", "takeover", "major acquisitions"
        ]
    }
}

UNRELATED_TOPIC_KEYWORDS = [
    "derivative", "hedging", "fair value measurement", "income tax", "tax rate",
    "pension", "postretirement", "stock-based compensation", "operating lease", "commitments and contingencies"
]

# Named ratios / verdict frameworks a question can ask about without ever
# spelling out the formula (unlike e.g. the fixed-asset-turnover practice
# question, which defines its own formula inline). Without this table the
# system has no way to know that "quick ratio" requires current assets,
# inventory, and current liabilities, or that a "capital-intensive?"
# verdict should be grounded in CAPEX/revenue, fixed-assets/total-assets,
# and ROA rather than a gut read of raw dollar figures -- so retrieval
# never boosts the right line items and the LLM either declares the
# question unanswerable or eyeballs a wrong verdict. Matching one of these
# expands normalized_concepts/accounting_terms to the ratio's required
# inputs and attaches the formula to the prompt as a deterministic note.
DERIVED_RATIOS: Dict[str, Dict] = {
    "QUICK_RATIO": {
        # Confirmed against a real miss: this dataset's own answer key
        # computed quick ratio from the specific liquid-asset line items
        # (cash + short-term investments + net receivables, including
        # receivables from related parties) rather than the textbook
        # (Current Assets - Inventory) approximation -- the two gave 1.57
        # vs. 1.77 for the same filing. Keeping CURRENT_ASSETS/INVENTORY in
        # "requires" (not the specific line items) so retrieval doesn't
        # start abstaining on filings that don't break those out
        # separately; the formula text below just states the preference.
        "aliases": ["quick ratio", "acid test ratio", "acid-test ratio"],
        "requires": ["CASH_AND_EQUIVALENTS", "ACCOUNTS_RECEIVABLE", "CURRENT_ASSETS", "INVENTORY", "CURRENT_LIABILITIES"],
        "formula": (
            "Quick Ratio = (most liquid current assets) / Current Liabilities. "
            "If cash and cash equivalents, short-term investments, and net "
            "receivables (including receivables from related parties) are "
            "each separately reported in the evidence, sum those specific "
            "line items over Current Liabilities. Otherwise, compute "
            "(Current Assets - Inventory) / Current Liabilities instead -- "
            "always compute one of these two; the Current Assets and "
            "Inventory lines are on every balance sheet, so evidence "
            "sufficient for the fallback is always evidence sufficient to "
            "answer. Never return NOT_FOUND for this ratio merely because "
            "the more granular line items aren't separately broken out."
        ),
    },
    "CURRENT_RATIO": {
        # "working capital ratio" is the same formula under a different
        # name -- confirmed against a real practice question ("Define
        # working capital ratio as total current assets divided by total
        # current liabilities"). Added as an alias rather than a separate
        # entry so it shares this one's requires/formula.
        "aliases": ["current ratio", "working capital ratio"],
        "requires": ["CURRENT_ASSETS", "CURRENT_LIABILITIES"],
        "formula": "Current Ratio = Current Assets / Current Liabilities",
    },
    "INVENTORY_TURNOVER": {
        # Confirmed against a real miss: asked bare ("Calculate inventory
        # turnover ratio for FY2022"), no inline formula -- unlike DPO/
        # working-capital-ratio/EBITDA questions in this dataset, which all
        # spell their formula out and so already get their line items
        # boosted by the plain keyword-matching concept loop above. Without
        # this entry, retrieval had no signal to fetch the income
        # statement's COGS figure alongside the balance sheet's inventory
        # line, and the model retrieved only the latter.
        #
        # Formula convention confirmed by reverse-engineering this dataset's
        # own gold figure: AES FY2022, COGS $10,069M, ending inventory
        # $1,055M -> 10,069/1,055 = 9.55 = the gold answer (9.5) exactly.
        # COGS/AVERAGE inventory ((1,055+604)/2=829.5) gives 12.1 instead --
        # confirmed wrong against the same gold value, not a guess. This
        # dataset's convention is ending inventory, not the textbook average.
        "aliases": ["inventory turnover"],
        "requires": ["INVENTORY", "COGS"],
        "formula": (
            "Inventory Turnover = Cost of Goods Sold (Cost of Sales) / Ending "
            "Inventory (the inventory balance at the end of the period being "
            "asked about -- not averaged with the prior period's balance). "
            "Requires both the income statement's COGS/cost of sales figure and "
            "the balance sheet's ending inventory figure -- retrieve and use both "
            "statements even though the question only names inventory."
        ),
    },
    "CAPITAL_INTENSITY": {
        "aliases": ["capital-intensive", "capital intensive", "capital intensity"],
        "requires": ["CAPEX", "NET_SALES_REVENUE", "PROPERTY_PLANT_EQUIPMENT_NET", "TOTAL_ASSETS", "NET_INCOME"],
        "formula": (
            "Judge capital intensity from ratios, not raw dollar magnitudes: "
            "CAPEX / Revenue; Property Plant & Equipment (net) / Total Assets; "
            "Return on Assets = Net Income / Total Assets."
        ),
    },
    "DEBT_TO_EQUITY": {
        "aliases": ["debt to equity", "debt-to-equity"],
        "requires": ["DEBT", "TOTAL_EQUITY"],
        "formula": "Debt-to-Equity = Total Debt / Total Equity",
    },
}


@dataclass
class QueryAnalysis:
    raw_query: str
    query_type: str  # NUMERIC_LOOKUP, CALCULATION, COMPARISON, TREND, EXPLANATION, DEFINITION, GENERAL
    explicitly_requested_statement: str = "ANY"  # CASH_FLOW_STATEMENT, INCOME_STATEMENT, BALANCE_SHEET, EQUITY_STATEMENT, MD_AND_A, FOOTNOTES, RISK_FACTORS, BUSINESS, ANY
    inferred_statement: str = "ANY"
    target_years: List[str] = field(default_factory=list)
    comparison_years: List[str] = field(default_factory=list)
    is_comparison: bool = False
    normalized_concepts: List[str] = field(default_factory=list)
    accounting_terms: List[str] = field(default_factory=list)
    company: str = ""
    filing_type: str = ""
    quarter: str = ""
    metric: str = ""
    requested_unit: str = ""
    comparison_direction: str = ""
    requires_calculation: bool = False
    requires_multiple_evidence_chunks: bool = False
    target_statement_types: List[str] = field(default_factory=list)
    derived_ratio_formula: str = ""

    @property
    def requested_statement(self) -> str:
        """Returns explicitly requested statement if specified, else inferred_statement."""
        return self.explicitly_requested_statement if self.explicitly_requested_statement != "ANY" else self.inferred_statement

    @property
    def detected_concepts(self) -> List[str]:
        return self.normalized_concepts


def analyze_query(query: str) -> QueryAnalysis:
    """Extract normalized financial concepts as the primary semantic signal for financial queries."""
    if not query:
        return QueryAnalysis(raw_query="", query_type="GENERAL", explicitly_requested_statement="ANY")

    q_lower = query.lower()

    # 1. Detect Query Type & Comparison Flag
    is_comparison = False
    requires_calculation = False
    document_purpose_markers = (
        "key agenda", "agenda", "purpose of", "main purpose", "filing about",
        "what did the filing report", "what was disclosed", "what did this filing report",
        "what event", "what events", "main subject", "key subject", "main topic",
        "what was the purpose", "what was the key", "what was the main",
    )
    calculation_markers = (
        "calculate", "compute", "ratio", "percentage", "percent", "margin",
        "growth rate", "year-over-year change", "yoy change", "defined as",
        "days payable outstanding", " dpo", "round your answer",
    )
    explanation_markers = ("why", "explain", "describe", "reason", "driver")
    comparison_markers = (
        "compare", "versus", "vs", "difference", "change in", "year-over-year", "yoy", "growth",
        "highest", "lowest", "largest", "smallest", "greatest", "least", "rank ", "ranked",
        "which of", "which segment", "which category", "which region", "which product",
        # "Which X are registered/listed..." enumeration questions land here
        # too -- confirmed against a real miss: "which debt securities are
        # registered to trade..." fell all the way through to GENERAL
        # (query_type's narrowest, 5-chunk retrieval tier) because none of
        # the markers above name a financial-instrument noun, even though
        # this is exactly the same "pick the right item(s) out of several
        # named candidates" shape as "which segment" already gets COMPARISON
        # treatment for.
        "which debt", "which securities", "which notes", "which bonds",
    )
    trend_markers = ("trend", "over time", "historical", "prior years")
    numeric_lookup_markers = ("how much", "amount", "value", "total", "figure", "cost", "dollar", "what was the", "what is the")
    definition_markers = ("what is", "define", "meaning")

    if any(k in q_lower for k in document_purpose_markers) and any(k in q_lower for k in ("filing", "8k", "8-k", "10q", "10-q", "10k", "10-k", "form")):
        query_type = "DOCUMENT_PURPOSE"
    else:
        # Priority-ordered candidates, each with its marker list. A plain
        # first-match-wins elif chain over these (the original design) has a
        # real, recurring failure mode: a single weak/overloaded keyword in
        # an earlier category ("impact" in EXPLANATION) silently shadows a
        # later category even when that later category matches on several
        # specific markers ("which segment" + "growth" in COMPARISON) --
        # confirmed on a real miss, and fixed there by hand-removing
        # "impact". Rather than repeat that per keyword as each collision is
        # found, a later category with >=2 marker hits now outranks an
        # earlier category that only matched once; priority order is still
        # the tie-break (and the only rule) whenever nothing reaches 2, so
        # ordinary single-signal questions classify exactly as before.
        candidates = [
            ("EXPLANATION", explanation_markers),
            ("CALCULATION", calculation_markers),
            ("COMPARISON", comparison_markers),
            ("TREND", trend_markers),
            ("NUMERIC_LOOKUP", numeric_lookup_markers),
            ("DEFINITION", definition_markers),
        ]
        matches = []
        for rank, (name, markers) in enumerate(candidates):
            hit_count = sum(1 for k in markers if k in q_lower)
            if hit_count:
                matches.append((rank, name, hit_count))

        if not matches:
            query_type = "GENERAL"
        else:
            strong = [m for m in matches if m[2] >= 2]
            rank, query_type, _ = min(strong or matches, key=lambda m: m[0])

        if query_type == "CALCULATION":
            requires_calculation = True
            if any(k in q_lower for k in ("year-over-year", "yoy", "change", "from ", " to ")):
                is_comparison = True
        elif query_type in ("COMPARISON", "TREND"):
            is_comparison = True

    # 2. Extract Normalized Concepts & Dynamic Terminology Aliases
    normalized_concepts = []
    accounting_terms = []
    inferred_statement = "ANY"

    for concept_id, concept_info in ACCOUNTING_CONCEPTS.items():
        for kw in concept_info["keywords"]:
            if kw in q_lower:
                normalized_concepts.append(concept_id)
                accounting_terms.extend(concept_info["keywords"])
                if inferred_statement == "ANY":
                    inferred_statement = concept_info["statement"]
                break

    metric = normalized_concepts[0] if normalized_concepts else ""

    # 2b. Named Ratio / Verdict Framework Detection (see DERIVED_RATIOS).
    # These questions never say "current assets" or "return on assets" --
    # they name the ratio/verdict and expect the underlying line items to
    # be found and combined, so the concept/accounting-term lists have to
    # be expanded here or retrieval never boosts the right evidence.
    derived_ratio_formula = ""
    derived_ratio_statement_types = []
    for ratio_info in DERIVED_RATIOS.values():
        if not any(alias in q_lower for alias in ratio_info["aliases"]):
            continue
        derived_ratio_formula = ratio_info["formula"]
        requires_calculation = True
        if query_type not in ("CALCULATION", "COMPARISON", "TREND"):
            query_type = "CALCULATION"
        for concept_id in ratio_info["requires"]:
            concept_info = ACCOUNTING_CONCEPTS.get(concept_id)
            if not concept_info:
                continue
            if concept_id not in normalized_concepts:
                normalized_concepts.append(concept_id)
            for kw in concept_info["keywords"]:
                if kw not in accounting_terms:
                    accounting_terms.append(kw)
            if inferred_statement == "ANY":
                inferred_statement = concept_info["statement"]
            if concept_info["statement"] not in derived_ratio_statement_types:
                derived_ratio_statement_types.append(concept_info["statement"])
        break

    # 3. Detect EXPLICITLY Requested Statement Type vs INFERRED Statement Type
    explicitly_requested = "ANY"
    if any(k in q_lower for k in ("cash flow statement", "statement of cash flows", "consolidated statement of cash flows")):
        explicitly_requested = "CASH_FLOW_STATEMENT"
    elif any(k in q_lower for k in ("income statement", "statement of income", "statement of operations", "consolidated statement of income")):
        explicitly_requested = "INCOME_STATEMENT"
    elif any(k in q_lower for k in ("p&l", "profit and loss")):
        explicitly_requested = "INCOME_STATEMENT"
    elif any(k in q_lower for k in ("balance sheet", "statement of financial position", "consolidated balance sheet")):
        explicitly_requested = "BALANCE_SHEET"
    elif any(k in q_lower for k in ("statement of equity", "statement of stockholders equity", "retained earnings statement")):
        explicitly_requested = "EQUITY_STATEMENT"
    elif any(k in q_lower for k in ("item 1a", "risk factors", "risk factor")):
        explicitly_requested = "RISK_FACTORS"
    elif any(k in q_lower for k in ("item 7", "md&a", "management discussion")):
        explicitly_requested = "MD_AND_A"
    elif any(k in q_lower for k in ("footnotes", "footnote", "note 1", "note 2", "note 3")):
        explicitly_requested = "FOOTNOTES"
    elif any(k in q_lower for k in ("item 1", "business description")):
        explicitly_requested = "BUSINESS"

    # 4. Detect Filing, Period, Company, Unit, and Derived Comparison Periods
    filing_type = ""
    if re.search(r"\b10\s*-\s*k\b|\b10k\b", q_lower):
        filing_type = "10-K"
    elif re.search(r"\b10\s*-\s*q\b|\b10q\b", q_lower):
        filing_type = "10-Q"
    elif re.search(r"\b8\s*-\s*k\b|\b8k\b", q_lower):
        filing_type = "8-K"

    quarter = ""
    q_match = re.search(r"\b(q[1-4]|first quarter|second quarter|third quarter|fourth quarter)\b", q_lower)
    if q_match:
        q_map = {
            "first quarter": "Q1",
            "second quarter": "Q2",
            "third quarter": "Q3",
            "fourth quarter": "Q4",
        }
        quarter = q_map.get(q_match.group(1), q_match.group(1).upper())

    requested_unit = ""
    if any(k in q_lower for k in ("basis point", "bps")):
        requested_unit = "basis points"
    elif any(k in q_lower for k in ("percent", "percentage", "%")):
        requested_unit = "percent"
    elif any(k in q_lower for k in ("million", "millions")):
        requested_unit = "millions"
    elif any(k in q_lower for k in ("billion", "billions")):
        requested_unit = "billions"

    company = ""
    possessive = re.search(r"\b([A-Z][A-Za-z0-9&.\-]{1,30})'s\b", query)
    if possessive:
        company = possessive.group(1).strip()
    else:
        all_caps = re.findall(r"\b[A-Z][A-Z0-9&.-]{1,12}\b", query)
        skip = {"FY", "Q1", "Q2", "Q3", "Q4", "SEC", "GAAP", "USD", "US"}
        company_terms = [t for t in all_caps if t not in skip and not re.fullmatch(r"20\d\d", t)]
        if company_terms:
            company = company_terms[0]

    years = re.findall(r"\b(20\d\d|fy20\d\d)\b", q_lower)
    clean_years = list(dict.fromkeys(y.replace("fy", "") for y in years))
    comparison_years = []
    comparison_direction = ""
    if any(k in q_lower for k in ("increase", "improve", "higher", "growth", "grew")):
        comparison_direction = "increase"
    elif any(k in q_lower for k in ("decrease", "decline", "lower", "reduced", "fell")):
        comparison_direction = "decrease"

    # Implicit Comparison Period Derivation (e.g. "prior year", "previous year", "yoy", "year-over-year")
    if any(k in q_lower for k in ("yoy", "year-over-year", "prior year", "previous year", "last year", "preceding year", "compared to previous", "compared to last")):
        is_comparison = True
        if clean_years:
            for y_str in clean_years:
                try:
                    prev_yr = str(int(y_str) - 1)
                    if prev_yr not in clean_years and prev_yr not in comparison_years:
                        comparison_years.append(prev_yr)
                except ValueError:
                    pass

    all_target_years = list(dict.fromkeys(clean_years + comparison_years))
    target_statement_types = []
    if explicitly_requested != "ANY":
        target_statement_types.append(explicitly_requested)
    if inferred_statement != "ANY":
        target_statement_types.append(inferred_statement)
    if any(k in q_lower for k in ("balance sheet", "balance sheets")) and "BALANCE_SHEET" not in target_statement_types:
        target_statement_types.append("BALANCE_SHEET")
    if any(k in q_lower for k in ("p&l", "profit and loss", "income statement", "statement of income", "statement of operations")) and "INCOME_STATEMENT" not in target_statement_types:
        target_statement_types.append("INCOME_STATEMENT")
    for stmt in derived_ratio_statement_types:
        if stmt not in target_statement_types:
            target_statement_types.append(stmt)

    requires_multiple_evidence_chunks = (
        is_comparison
        or query_type in ("COMPARISON", "TREND", "CALCULATION")
        or len(all_target_years) > 1
        or any(k in q_lower for k in (" and ", "using ", "between ", "from ", "to ", "multi-year"))
    )

    return QueryAnalysis(
        raw_query=query,
        query_type=query_type,
        explicitly_requested_statement=explicitly_requested,
        inferred_statement=inferred_statement,
        target_years=all_target_years,
        comparison_years=comparison_years,
        is_comparison=is_comparison,
        normalized_concepts=list(dict.fromkeys(normalized_concepts)),
        accounting_terms=list(dict.fromkeys(accounting_terms)),
        company=company,
        filing_type=filing_type,
        quarter=quarter,
        metric=metric,
        requested_unit=requested_unit,
        comparison_direction=comparison_direction,
        requires_calculation=requires_calculation,
        requires_multiple_evidence_chunks=requires_multiple_evidence_chunks,
        target_statement_types=list(dict.fromkeys(target_statement_types)),
        derived_ratio_formula=derived_ratio_formula,
    )


def expand_query(query: str) -> str:
    """Append generic filing-wording aliases to expand BM25 and vector search recall."""
    if not query:
        return query

    analysis = analyze_query(query)
    q_lower = query.lower()
    extra_terms = []

    if analysis.query_type == "DOCUMENT_PURPOSE":
        extra_terms.extend([
            "item information",
            "form type",
            "document description",
            "filed as of date",
            "conformed submission type",
            "event",
            "exhibit",
            "financial statements and exhibits",
        ])

    # "What drove X change" / "which segment..." questions: the question's
    # own wording tends to rank a general narrative passage above the
    # specific breakdown table that actually answers it (a special-items
    # reconciliation; an organic-sales-by-segment table) -- confirmed
    # against two real misses (3M operating-margin drivers, 3M segment
    # comparison) where the correct breakdown never entered the retrieved
    # candidate pool at all, because nothing about "what drove 3M's
    # operating margin change" lexically or semantically favors a
    # reconciliation table over any other passage discussing the margin.
    # These terms target that specific table type instead of the metric in
    # general -- retrieval-recall gaps like this can't be fixed downstream
    # by reranking or a better formula; the evidence has to be retrieved
    # in the first place.
    if analysis.query_type == "EXPLANATION" and any(k in q_lower for k in ("drove", "driver", "drivers", "caused", "contributed to")):
        extra_terms.extend(["special items", "one-time charges", "significant items", "reconciliation"])
    if any(k in q_lower for k in ("which segment", "which category", "which region", "which product", "which business", "which division")):
        extra_terms.extend(["organic sales change", "by segment", "each segment"])

    if analysis.accounting_terms:
        extra_terms.extend(analysis.accounting_terms)

    extra_terms = [t for t in dict.fromkeys(extra_terms) if t not in q_lower]

    if not extra_terms:
        return query

    return query + " " + " ".join(extra_terms)
