"""Explicit custody scenarios for transformed securities; never automatic aliases.

The bridge expresses successor prices per old receipt for already-held positions.
It does not add successor instruments to the model's prediction universe. It is
valid only for the explicitly acknowledged custody/election scenario, with
fractional share settlement, zero conversion fees and no residual old claim.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SecurityTransition:
    predecessor: str
    successor: str
    old_isin: str
    new_isin: str
    successor_shares_per_old: float
    first_tradable_date: str
    eligibility: str
    sources: tuple[str, ...]
    last_predecessor_session: str | None = None


VERIFIED_TRANSITIONS = (
    SecurityTransition(
        "AGRO", "RAGR", "US7496552057", "RU000A0JQUZ6", 7.0135, "2025-02-17",
        "Eligible Russian-depository GDR owner whose RAGR shares were actually credited before trading; not foreign custody or uncompleted applications",
        ("https://www.moex.com/n77722",
         "https://www.tbank.ru/invest/social/profile/Rusagro/25b9175a-f89d-412f-b752-b71ef66f5ee3/",
         "https://www.rusagrogroup.ru/ru/investoram/shares-distribution/",
         "https://bcs.ru/newsdepo/1024099")),
    SecurityTransition(
        "FIXP", "FIXR", "US33835G2057", "RU000A10B5G8", 158., "2025-08-20",
        "Holder voluntarily tendered through Russian exchange custody, offer accepted and successor shares credited; not an automatic conversion and not OTC settlement",
        ("https://www.moex.com/n92917",
         "https://ir.fix-price.com/upload/iblock/22f/27rf3pxxb3qjfztprwn375cd44xgse9y/Press%20release_Exchange_ENG.pdf",
         "https://media.fix-price.com/news/fix-price-obyavlyaet-itogi-obmena-gdr-na-aktsii-pao-fiks-prays/?back=8786")),
)


# Same-code events are recorded separately: they must not go through the price
# bridge (which would overwrite real quotes). Existing backtests already cross
# this boundary and are conditional on receipt conversion for old positions.
SAME_TICKER_TRANSITIONS = (
    SecurityTransition(
        "OZON", "OZON", "US69269L1044", "RU000A10CW95", 1., "2025-11-11",
        "Existing Russian-custody ADS automatically converted 3-20 October 2025 and shares credited before resumption; not an uncompleted foreign-custody conversion",
        ("https://www.moex.com/n93647", "https://www.moex.com/n93976", "https://www.moex.com/n95178",
         "https://ir.ozon.com/redomiciliation",
         "https://www.nsd.ru/ru/news/o-publikatsii-daty-sostavleniya-spiska-depozitariev-osuschestvlyayuschih-uchet-prav-na-amerikanskie-depozitarnye-raspiski-na-obyknovennye-aktsii-ozon-holdings-plc-isin-us69269l1044-262654/"),
        last_predecessor_session="2025-09-22"),
)


@dataclass(frozen=True)
class TickerAlias:
    """A renamed trading code for one legal security, not a receipt exchange."""
    predecessor: str
    successor: str
    isin: str
    history_start: str
    last_predecessor_session: str
    first_successor_session: str
    announced_at: str
    sources: tuple[str, ...]


VERIFIED_ALIASES = (
    TickerAlias("TCSG", "T", "RU000A107UL4", "2024-03-18", "2024-11-27",
                "2024-11-28", "2024-11-14T15:08:00+03:00",
                ("https://www.moex.com/n74886", "https://www.moex.com/n68303",
                 "https://www.moex.com/n67851")),
)


def validate_alias_history(history: pd.DataFrame, alias: TickerAlias) -> pd.DataFrame:
    """Validate a same-ISIN segment, preserving its actual historical ticker.

    The start boundary excludes the earlier foreign GDR with the *same* TCSG
    code but a different ISIN. Quotes remain unadjusted in the CSV; the ordinary
    split ledger must be applied exactly once if this segment enters a panel.
    """
    required = {"TRADEDATE", "ticker", "ISIN", "OPEN", "CLOSE"}
    if not required.issubset(history):
        raise ValueError("Alias history needs dates, prices, ticker and ISIN")
    result = history.copy()
    result["TRADEDATE"] = pd.to_datetime(result.TRADEDATE, errors="raise")
    if result.empty or result.TRADEDATE.isna().any() or result.TRADEDATE.duplicated().any():
        raise ValueError("Missing or duplicate alias dates")
    if not result.ticker.eq(alias.predecessor).all() or not result.ISIN.eq(alias.isin).all():
        raise ValueError("Alias security identity differs from verified ledger")
    if not result.TRADEDATE.between(pd.Timestamp(alias.history_start),
                                   pd.Timestamp(alias.last_predecessor_session)).all():
        raise ValueError("Alias history crosses the verified legal-security interval")
    for col in ("OPEN", "CLOSE"):
        if (result[col].notna() & (~np.isfinite(result[col]) | result[col].le(0))).any():
            raise ValueError("Invalid alias quote")
    result["security_isin"] = alias.isin
    result["successor_ticker"] = alias.successor
    result["alias_announced_at"] = alias.announced_at
    return result.sort_values("TRADEDATE")


def transition_records():
    return [asdict(event) for event in VERIFIED_TRANSITIONS]


def same_ticker_transition_records():
    return [asdict(event) for event in SAME_TICKER_TRANSITIONS]


def bridge_held_security_quotes(quotes: pd.DataFrame, successors: pd.DataFrame,
                               *, acknowledged_predecessors: set[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Make a separate scenario quote grid, preserving every pre-transition quote.

    Explicit acknowledgement is mandatory per predecessor. Existing OHLC under
    the old code after the new trading date is an error, not silently overwritten.
    Successor quotes must be unadjusted per-share units for the verified ISIN;
    subsequent splits must be handled explicitly before this function is used.
    Missing successor quotes stay missing, including the whole trading pause.
    """
    required = {"TRADEDATE", "ticker", "OPEN", "CLOSE"}
    if not required.issubset(quotes) or not required.issubset(successors):
        raise ValueError("Quote tables need date/ticker/open/close")
    known = {e.predecessor for e in VERIFIED_TRANSITIONS}
    if not acknowledged_predecessors <= known:
        raise ValueError("Unknown transition acknowledgement")
    out = quotes.copy()
    src = successors.copy()
    for frame in (out, src):
        frame["TRADEDATE"] = pd.to_datetime(frame.TRADEDATE, errors="raise")
        if frame.TRADEDATE.isna().any() or frame.ticker.isna().any():
            raise ValueError("Missing quote date or ticker")
        if frame.TRADEDATE.dt.tz is not None or not frame.TRADEDATE.eq(frame.TRADEDATE.dt.normalize()).all():
            raise ValueError("Quote dates must be naive calendar dates")
        if frame.duplicated(["TRADEDATE", "ticker"]).any():
            raise ValueError("Duplicate quote keys")
        for col in ("OPEN", "CLOSE"):
            if ((frame[col].notna()) & (~np.isfinite(frame[col]) | frame[col].le(0))).any():
                raise ValueError("Invalid observed quote")
    out["quote_ticker"] = out.ticker
    out["continuity_assumption"] = "none"
    if "entry_allowed" not in out:
        out["entry_allowed"] = True
    elif not out.entry_allowed.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("entry_allowed must contain explicit booleans")
    audit = []
    for event in VERIFIED_TRANSITIONS:
        if event.predecessor not in acknowledged_predecessors:
            continue
        following = src.loc[src.ticker.eq(event.successor)].copy()
        if following.empty:
            raise ValueError(f"Missing successor history: {event.successor}")
        if "ISIN" not in following or not following.ISIN.eq(event.new_isin).all():
            raise ValueError(f"Wrong or missing successor ISIN: {event.successor}")
        boundary = pd.Timestamp(event.first_tradable_date)
        following = following.loc[following.TRADEDATE.ge(boundary)].set_index("TRADEDATE")
        old_mask = out.ticker.eq(event.predecessor) & out.TRADEDATE.ge(boundary)
        if out.loc[old_mask, ["OPEN", "CLOSE"]].notna().any().any():
            raise ValueError(f"Observed predecessor quotes overlap successor: {event.predecessor}")
        selected = out.loc[old_mask, "TRADEDATE"]
        replacement = following.reindex(pd.DatetimeIndex(selected))
        for col in ("OPEN", "CLOSE"):
            out.loc[old_mask, col] = replacement[col].to_numpy() * event.successor_shares_per_old
        # Preserve monetary turnover; express volume in the old-receipt unit.
        if "VALUE" in out and "VALUE" in replacement:
            out.loc[old_mask, "VALUE"] = replacement.VALUE.to_numpy()
        if "VOLUME" in out and "VOLUME" in replacement:
            out.loc[old_mask, "VOLUME"] = replacement.VOLUME.to_numpy() / event.successor_shares_per_old
        out.loc[old_mask, "quote_ticker"] = event.successor
        out.loc[old_mask, "continuity_assumption"] = "eligible_received_fractional_successor"
        # These synthetic old-receipt prices liquidate existing positions only.
        # An accidental old-ticker forecast must never authorize a new purchase.
        out.loc[old_mask, "entry_allowed"] = False
        record = asdict(event)
        record["observed_successor_dates"] = int(replacement.CLOSE.notna().sum())
        record["basis"] = "one old receipt; successor price multiplied by shares_per_old"
        record["assumptions"] = ["custody eligibility and completed distribution/exchange acknowledged",
                                  "fractional shares; actual account rounding omitted",
                                  "zero conversion fees; no separately valued residual GDR claim",
                                  "no successor-based predictions added; only existing holdings valued"]
        audit.append(record)
    return out, audit
