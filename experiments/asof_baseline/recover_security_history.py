"""Download verified successor quote histories and build an explicit custody scenario.

Run with --acknowledge AGRO FIXP only to assume the documented local-custody
distribution/exchange was completed. Baseline quote files are never overwritten.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from ml_stocks_analysis.security_continuity import (
    VERIFIED_TRANSITIONS, VERIFIED_ALIASES, SAME_TICKER_TRANSITIONS, bridge_held_security_quotes,
    transition_records, validate_alias_history,
)


def collect_identity(ticker: str, expected_isin: str, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    url = f"https://iss.moex.com/iss/securities/{ticker}.json?iss.meta=off"
    with urlopen(url, timeout=45) as response:
        data = response.read()
    payload = json.loads(data)
    description = payload.get("description")
    if not description:
        raise RuntimeError(f"Missing security identity for {ticker}")
    fields = {row["name"]: row["value"]
              for row in (dict(zip(description["columns"], values)) for values in description["data"])}
    if fields.get("ISIN") != expected_isin or fields.get("SECID") != ticker:
        raise RuntimeError(f"Security identity differs from verified ledger for {ticker}: {fields.get('ISIN')}")
    file = output / f"{ticker}_identity.json"
    file.write_bytes(data)
    return fields["ISIN"], {"url": url, "file": file.name, "sha256": hashlib.sha256(data).hexdigest(), "verified_isin": fields["ISIN"]}


def collect_history(ticker: str, since: str, till: str, output: Path):
    base = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
    output.mkdir(parents=True, exist_ok=True)
    rows, columns, pages, expected = [], None, [], None
    offset = 0
    while True:
        url = base + "?" + urlencode({"from": since, "till": till, "iss.meta": "off", "start": offset})
        with urlopen(url, timeout=45) as response:
            data = response.read()
        payload = json.loads(data)
        history = payload.get("history")
        cursor = payload.get("history.cursor")
        if not history or not cursor or not cursor.get("data"):
            raise RuntimeError(f"Missing ISS history/cursor: {url}")
        page = history["data"]
        header = history["columns"]
        c = dict(zip(cursor["columns"], cursor["data"][0]))
        if expected is None:
            expected = int(c["TOTAL"])
            columns = header
        if int(c["TOTAL"]) != expected or header != columns or int(c["INDEX"]) != offset:
            raise RuntimeError("Inconsistent ISS pagination")
        file = output / f"{ticker}_{offset:05d}.json"
        file.write_bytes(data)
        pages.append({"url": url, "file": file.name, "sha256": hashlib.sha256(data).hexdigest(), "rows": len(page)})
        rows.extend(page)
        if len(rows) == expected:
            break
        if not page or len(rows) > expected:
            raise RuntimeError("Incomplete or overfull ISS pagination")
        offset += len(page)
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty or frame.duplicated("TRADEDATE").any():
        raise RuntimeError(f"Missing or duplicate successor history: {ticker}")
    frame["TRADEDATE"] = pd.to_datetime(frame.TRADEDATE)
    if not frame.SECID.eq(ticker).all() or not frame.BOARDID.eq("TQBR").all():
        raise RuntimeError("History payload contains another instrument or board")
    if not frame.TRADEDATE.between(pd.Timestamp(since), pd.Timestamp(till)).all():
        raise RuntimeError("History payload outside requested dates")
    frame["ticker"] = ticker
    frame = frame.sort_values("TRADEDATE")
    observed = frame.CLOSE.dropna()
    if observed.pct_change(fill_method=None).abs().gt(.7).any():
        raise RuntimeError(f"Unverified successor unit jump: {ticker}")
    return frame, pages


def recover_aliases(output: Path):
    """Save original-code raw history for a later controlled panel rebuild."""
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for alias in VERIFIED_ALIASES:
        actual_isin, identity = collect_identity(alias.predecessor, alias.isin, output)
        frame, pages = collect_history(alias.predecessor, alias.history_start,
                                       alias.last_predecessor_session, output / "iss_pages")
        frame["ISIN"] = actual_isin
        frame = validate_alias_history(frame, alias)
        observed = frame.loc[frame.CLOSE.notna(), "TRADEDATE"]
        if observed.min() != pd.Timestamp(alias.history_start) or observed.max() != pd.Timestamp(alias.last_predecessor_session):
            raise RuntimeError(f"Incomplete verified alias boundary: {alias.predecessor}")
        file = output / f"{alias.predecessor}_same_isin.csv"
        frame.to_csv(file, index=False)
        records.append(dict(**asdict(alias), rows=len(frame), identity=identity, pages=pages,
                            file=file.name, sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        print(alias.predecessor, len(frame), "same-ISIN historical alias rows", flush=True)
    manifest = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "aliases": records,
                "price_basis": "raw unadjusted MOEX per-share prices; apply normal split ledger once on panel integration",
                "limits": ["original historical ticker preserved; not added to current predictions",
                           "earlier TCSG GDR US87238U2033 excluded: it is a different security",
                           "current ISS identity alone cannot establish historical identity; dated MOEX notices define segment"]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def _verified_bytes(directory: Path, record: dict) -> bytes:
    """Check a retained source against its frozen manifest, without networking."""
    filename = record.get("file", "")
    if not filename or Path(filename).name != filename:
        raise RuntimeError("Replay manifest contains an unsafe or missing filename")
    data = (directory / filename).read_bytes()
    if hashlib.sha256(data).hexdigest() != record.get("sha256"):
        raise RuntimeError(f"Replay SHA-256 mismatch: {filename}")
    return data


def replay_verified_history(output: Path, quotes: Path, *,
                            acknowledged_predecessors: set[str], through: str):
    """Replay only frozen inputs; reject source, baseline or ledger changes.

    This function is read-only. The CLI writes the derived quote grid only if
    needed, preserving the original download manifest and its retrieval dates.
    """
    manifest = json.loads((output / "manifest.json").read_text())
    # JSON roundtrip normalizes dataclass source tuples into persisted lists.
    expected = json.loads(json.dumps(transition_records()))
    same_ticker = json.loads(json.dumps([asdict(event) for event in SAME_TICKER_TRANSITIONS]))
    if manifest.get("transitions") != expected or manifest.get("inherited_same_ticker_custody_assumptions") != same_ticker:
        raise RuntimeError("Replay transition ledger differs from current verified ledger")
    if manifest.get("through") != through or set(manifest.get("acknowledged", [])) != acknowledged_predecessors:
        raise RuntimeError("Replay date or custody acknowledgements differ from frozen run")
    if hashlib.sha256(quotes.read_bytes()).hexdigest() != manifest.get("baseline_quote_sha256"):
        raise RuntimeError("Replay baseline quote SHA-256 mismatch")
    history = manifest.get("history", [])
    if len(history) != len(VERIFIED_TRANSITIONS) or {item["ticker"] for item in history} != {event.successor for event in VERIFIED_TRANSITIONS}:
        raise RuntimeError("Replay has missing or duplicate successor histories")
    records = {item["ticker"]: item for item in history}
    frames = []
    for event in VERIFIED_TRANSITIONS:
        item = records[event.successor]
        identity = json.loads(_verified_bytes(output, item["identity"]))["description"]
        fields = {row["name"]: row["value"] for row in
                  (dict(zip(identity["columns"], values)) for values in identity["data"])}
        if fields.get("ISIN") != event.new_isin or fields.get("SECID") != event.successor:
            raise RuntimeError("Replay security identity differs from verified ledger")
        if not item.get("pages"):
            raise RuntimeError("Replay requires retained ISS page snapshots")
        for page in item["pages"]:
            _verified_bytes(output / "iss_pages", page)
        from io import BytesIO
        frame = pd.read_csv(BytesIO(_verified_bytes(output, item)), float_precision="round_trip")
        frame["TRADEDATE"] = pd.to_datetime(frame.TRADEDATE)
        if len(frame) != item["rows"] or not frame.ticker.eq(event.successor).all() or not frame.ISIN.eq(event.new_isin).all():
            raise RuntimeError("Replay CSV rows or security identity differ from manifest")
        if not frame.TRADEDATE.between(pd.Timestamp(event.first_tradable_date), pd.Timestamp(through)).all():
            raise RuntimeError("Replay CSV outside verified interval")
        frames.append(frame)
    return bridge_held_security_quotes(pd.read_parquet(quotes), pd.concat(frames, ignore_index=True),
                                       acknowledged_predecessors=acknowledged_predecessors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--till", default="2026-11-06")
    parser.add_argument("--output", type=Path, default=HERE / "continuity_data")
    parser.add_argument("--quotes", type=Path, default=HERE / "quotes.parquet")
    parser.add_argument("--acknowledge", nargs="*", default=[])
    parser.add_argument("--recover-aliases", action="store_true",
                        help="Also recover verified same-ISIN raw segments without changing current forecasts")
    parser.add_argument("--aliases-only", action="store_true",
                        help="Recover historical aliases and leave existing conversion outputs untouched")
    parser.add_argument("--offline", action="store_true",
                        help="Replay frozen CSVs and hash-verified snapshots; never fetch revised market data")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.offline:
        if args.recover_aliases or args.aliases_only:
            parser.error("--offline replays conversion quotes only; alias CSVs are already retained")
        bridged, audit = replay_verified_history(args.output, args.quotes,
            acknowledged_predecessors=set(args.acknowledge), through=args.till)
        target = args.output / "quotes_conversion_scenario.parquet"
        if target.exists() and pd.read_parquet(target).equals(bridged):
            print("Offline source checks passed; derived quote grid unchanged", flush=True)
        else:
            bridged.to_parquet(target, index=False)
            print("Offline source checks passed; rebuilt derived quote grid", flush=True)
        return
    if args.recover_aliases or args.aliases_only:
        recover_aliases(args.output / "aliases")
    if args.aliases_only:
        return
    frames, sources = [], []
    for event in VERIFIED_TRANSITIONS:
        actual_isin, identity = collect_identity(event.successor, event.new_isin, args.output)
        frame, pages = collect_history(event.successor, event.first_tradable_date, args.till, args.output / "iss_pages")
        frame["ISIN"] = actual_isin
        file = args.output / f"{event.successor}.csv"
        frame.to_csv(file, index=False)
        first = frame.loc[frame.CLOSE.notna(), "TRADEDATE"].min()
        if first != pd.Timestamp(event.first_tradable_date):
            raise RuntimeError(f"Observed trading start contradicts verified ledger for {event.successor}: {first}")
        frames.append(frame)
        sources.append({"ticker": event.successor, "rows": len(frame), "first_observed": str(first.date()), "identity": identity, "pages": pages,
                        "file": file.name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
        print(event.successor, len(frame), str(first.date()), flush=True)
    successors = pd.concat(frames, ignore_index=True)
    successors.to_parquet(args.output / "successor_quotes.parquet", index=False)
    bridged, audit = bridge_held_security_quotes(pd.read_parquet(args.quotes), successors,
                                                acknowledged_predecessors=set(args.acknowledge))
    bridged.to_parquet(args.output / "quotes_conversion_scenario.parquet", index=False)
    manifest = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "source": "MOEX ISS TQBR historical observations",
                "through": args.till, "baseline_quote_sha256": hashlib.sha256(args.quotes.read_bytes()).hexdigest(),
                "transitions": transition_records(), "acknowledged": args.acknowledge, "applied": audit, "history": sources,
                "inherited_same_ticker_custody_assumptions": [asdict(event) for event in SAME_TICKER_TRANSITIONS],
                "limits": ["explicit completed-conversion custody scenario, not automatic ticker aliases",
                           "no causal model features or new successor predictions generated",
                           "FIXR proposed 1000:1 consolidation not applied: MOEX n97730 postponed it; actual observed price history shows no such jump"]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
