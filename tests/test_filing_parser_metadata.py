import tempfile
import unittest
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from filing_parser import parse_filing_to_window_chunks


def parse_tables(html: str):
    with tempfile.NamedTemporaryFile("w", suffix=".htm", encoding="utf-8", delete=False) as f:
        f.write(html)
        path = f.name
    try:
        return [c for c in parse_filing_to_window_chunks(path) if c.get("chunk_type") == "table"]
    finally:
        Path(path).unlink(missing_ok=True)


class FilingParserMetadataTests(unittest.TestCase):
    def test_balance_sheet_statement_title_and_caption(self):
        tables = parse_tables("""
        <html><body>
          <p>Item 8 - Financial Statements and Supplementary Data</p>
          <p>3M Company and Subsidiaries</p>
          <p>Consolidated Balance Sheet</p>
          <p>At December 31</p>
          <table>
            <tr><th>Assets</th><th>2018</th><th>2017</th></tr>
            <tr><td>Current assets</td><td>13,709</td><td>14,277</td></tr>
            <tr><td>Property, plant and equipment - net</td><td>8,738</td><td>8,866</td></tr>
            <tr><td>Total assets</td><td>36,500</td><td>37,987</td></tr>
            <tr><td>Current liabilities</td><td>7,244</td><td>7,687</td></tr>
            <tr><td>Total liabilities</td><td>26,652</td><td>26,365</td></tr>
          </table>
        </body></html>
        """)
        self.assertEqual(tables[0]["statement_title"], "Consolidated Balance Sheet")
        self.assertEqual(tables[0]["table_title"], "At December 31")
        self.assertEqual(tables[0]["statement_type"], "BALANCE_SHEET")

    def test_income_statement_operations(self):
        tables = parse_tables("""
        <html><body>
          <p>Item 8 - Financial Statements and Supplementary Data</p>
          <p>Consolidated Statements of Operations</p>
          <p>For the Years Ended December 31</p>
          <table>
            <tr><th>Years ended December 31</th><th>2022</th><th>2021</th></tr>
            <tr><td>Net sales</td><td>10,000</td><td>9,000</td></tr>
            <tr><td>Cost of sales</td><td>6,000</td><td>5,500</td></tr>
            <tr><td>Operating income</td><td>2,000</td><td>1,700</td></tr>
            <tr><td>Net income</td><td>1,500</td><td>1,200</td></tr>
            <tr><td>Earnings per share</td><td>1.20</td><td>1.00</td></tr>
          </table>
        </body></html>
        """)
        self.assertEqual(tables[0]["statement_type"], "INCOME_STATEMENT")

    def test_cash_flow_statement(self):
        tables = parse_tables("""
        <html><body>
          <p>Item 8 - Financial Statements and Supplementary Data</p>
          <p>Consolidated Statements of Cash Flows</p>
          <p>For the Years Ended December 31</p>
          <table>
            <tr><th>Years ended December 31</th><th>2022</th><th>2021</th></tr>
            <tr><td>Cash flows from operating activities</td><td>800</td><td>700</td></tr>
            <tr><td>Net cash provided by operating activities</td><td>800</td><td>700</td></tr>
            <tr><td>Cash flows from investing activities</td><td>(200)</td><td>(150)</td></tr>
            <tr><td>Cash flows from financing activities</td><td>(300)</td><td>(250)</td></tr>
          </table>
        </body></html>
        """)
        self.assertEqual(tables[0]["statement_type"], "CASH_FLOW_STATEMENT")

    def test_equity_statement(self):
        tables = parse_tables("""
        <html><body>
          <p>Item 8 - Financial Statements and Supplementary Data</p>
          <p>Consolidated Statements of Changes in Equity</p>
          <table>
            <tr><th>Shareholders' equity</th><th>2022</th></tr>
            <tr><td>Common stock</td><td>10</td></tr>
            <tr><td>Additional paid-in capital</td><td>100</td></tr>
            <tr><td>Retained earnings</td><td>900</td></tr>
            <tr><td>Treasury stock</td><td>(50)</td></tr>
            <tr><td>Accumulated other comprehensive income</td><td>5</td></tr>
          </table>
        </body></html>
        """)
        self.assertEqual(tables[0]["statement_type"], "EQUITY_STATEMENT")

    def test_10q_item_1_financial_statement_not_business(self):
        tables = parse_tables("""
        <html><body>
          <p>Item 1 - Financial Statements</p>
          <p>(Unaudited)</p>
          <p>Consolidated Balance Sheets</p>
          <p>At June 30</p>
          <table>
            <tr><th>Assets</th><th>June 30, 2023</th><th>December 31, 2022</th></tr>
            <tr><td>Current assets</td><td>100</td><td>90</td></tr>
            <tr><td>Total assets</td><td>500</td><td>450</td></tr>
            <tr><td>Current liabilities</td><td>80</td><td>75</td></tr>
            <tr><td>Total liabilities</td><td>300</td><td>280</td></tr>
          </table>
        </body></html>
        """)
        self.assertEqual(tables[0]["statement_title"], "Consolidated Balance Sheets")
        self.assertEqual(tables[0]["table_title"], "At June 30")
        self.assertEqual(tables[0]["statement_type"], "BALANCE_SHEET")
        self.assertNotEqual(tables[0]["statement_type"], "BUSINESS")


if __name__ == "__main__":
    unittest.main()
