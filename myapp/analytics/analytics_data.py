import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd


@dataclass
class QueryEvent:
    query_id: int
    text: str
    term_count: int
    timestamp: datetime
    algorithm: str
    session_id: Optional[str]
    user_agent: Optional[str]
    browser: Optional[str]
    os: Optional[str]
    ip: Optional[str]


@dataclass
class ClickEvent:
    query_id: int
    doc_id: str
    rank: Optional[int]
    timestamp: datetime
    session_id: Optional[str]
    dwell_time_ms: Optional[float] = None


@dataclass
class SessionContext:
    session_id: str
    ip: Optional[str]
    user_agent: Optional[str]
    browser: Optional[str]
    os: Optional[str]
    first_seen: datetime
    last_seen: datetime
    queries: List[int] = field(default_factory=list)
    pending_click_ts: Optional[datetime] = None


class AnalyticsData:
    """In-memory analytics store."""

    def __init__(self) -> None:
        self.fact_clicks: Dict[str, int] = {}
        self.query_events: List[QueryEvent] = []
        self.click_events: List[ClickEvent] = []
        self.sessions: Dict[str, SessionContext] = {}
        self.query_result_map: Dict[int, Dict[str, int]] = {}
        self._query_counter = 0

    def next_query_id(self) -> int:
        self._query_counter += 1
        return self._query_counter

    def save_query_terms(
        self,
        terms: str,
        *,
        algorithm: str = "tfidf",
        session_id: Optional[str] = None,
        user_agent: Optional[str] = None,
        browser: Optional[str] = None,
        os_name: Optional[str] = None,
        ip: Optional[str] = None,
    ) -> int:
        query_id = self.next_query_id()
        event = QueryEvent(
            query_id=query_id,
            text=terms,
            term_count=len((terms or "").split()),
            timestamp=datetime.utcnow(),
            algorithm=algorithm,
            session_id=session_id,
            user_agent=user_agent,
            browser=browser,
            os=os_name,
            ip=ip,
        )
        self.query_events.append(event)

        if session_id:
            session = self.sessions.get(session_id)
            if session:
                session.queries.append(query_id)
                session.last_seen = event.timestamp
            else:
                self.sessions[session_id] = SessionContext(
                    session_id=session_id,
                    ip=ip,
                    user_agent=user_agent,
                    browser=browser,
                    os=os_name,
                    first_seen=event.timestamp,
                    last_seen=event.timestamp,
                    queries=[query_id],
                )
        return query_id

    def register_results(self, query_id: int, results: List[Any]) -> None:
        mapping: Dict[str, int] = {}
        for idx, item in enumerate(results):
            pid = None
            if hasattr(item, "pid"):
                pid = getattr(item, "pid")
            elif isinstance(item, dict):
                pid = item.get("pid")
            if pid:
                mapping[pid] = idx + 1
        self.query_result_map[query_id] = mapping

    def record_click(self, query_id: int, doc_id: str, session_id: Optional[str]) -> None:
        rank_map = self.query_result_map.get(query_id, {})
        rank = rank_map.get(doc_id)
        now = datetime.utcnow()
        event = ClickEvent(
            query_id=query_id,
            doc_id=doc_id,
            rank=rank,
            timestamp=now,
            session_id=session_id,
        )
        self.click_events.append(event)
        self.fact_clicks[doc_id] = self.fact_clicks.get(doc_id, 0) + 1

        if session_id:
            session = self.sessions.get(session_id)
            if session:
                session.pending_click_ts = now
                session.last_seen = now

    def record_return_to_results(self, session_id: Optional[str]) -> None:
        if not session_id:
            return
        session = self.sessions.get(session_id)
        if not session or not session.pending_click_ts:
            return
        dwell = (datetime.utcnow() - session.pending_click_ts).total_seconds() * 1000.0
        # Assign dwell time to the last click event for this session
        for event in reversed(self.click_events):
            if event.session_id == session_id and event.dwell_time_ms is None:
                event.dwell_time_ms = dwell
                break
        session.pending_click_ts = None

    def record_request_context(
        self,
        session_id: str,
        *,
        ip: Optional[str],
        user_agent: Optional[str],
        browser: Optional[str],
        os_name: Optional[str],
    ) -> SessionContext:
        now = datetime.utcnow()
        session = self.sessions.get(session_id)
        if session:
            session.last_seen = now
            session.ip = session.ip or ip
            session.user_agent = session.user_agent or user_agent
            session.browser = session.browser or browser
            session.os = session.os or os_name
            return session
        session = SessionContext(
            session_id=session_id,
            ip=ip,
            user_agent=user_agent,
            browser=browser,
            os=os_name,
            first_seen=now,
            last_seen=now,
        )
        self.sessions[session_id] = session
        return session

    def plot_number_of_views(self):
        data = [{'Document ID': doc_id, 'Number of Views': count} for doc_id, count in self.fact_clicks.items()]
        df = pd.DataFrame(data)
        if df.empty:
            df = pd.DataFrame({'Document ID': [], 'Number of Views': []})
        chart = alt.Chart(df).mark_bar().encode(
            x='Document ID',
            y='Number of Views'
        ).properties(
            title='Number of Views per Document'
        )
        return chart.to_html()

    def top_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        counter: Dict[str, int] = {}
        for event in self.query_events:
            counter[event.text] = counter.get(event.text, 0) + 1
        return sorted(
            [{'query': q, 'count': c} for q, c in counter.items()],
            key=lambda item: item['count'],
            reverse=True
        )[:limit]

    def top_clicked_docs(self, limit: int = 10) -> List[Dict[str, Any]]:
        return sorted(
            [{'doc_id': doc_id, 'count': count} for doc_id, count in self.fact_clicks.items()],
            key=lambda item: item['count'],
            reverse=True
        )[:limit]

    def browser_stats(self) -> Dict[str, int]:
        stats: Dict[str, int] = {}
        for session in self.sessions.values():
            if session.browser:
                stats[session.browser] = stats.get(session.browser, 0) + 1
        return stats

    def os_stats(self) -> Dict[str, int]:
        stats: Dict[str, int] = {}
        for session in self.sessions.values():
            if session.os:
                stats[session.os] = stats.get(session.os, 0) + 1
        return stats

    def click_distribution_by_rank(self) -> Dict[int, int]:
        distribution: Dict[int, int] = {}
        for event in self.click_events:
            if event.rank:
                distribution[event.rank] = distribution.get(event.rank, 0) + 1
        return dict(sorted(distribution.items()))

    def dwell_times(self) -> List[float]:
        return [event.dwell_time_ms for event in self.click_events if event.dwell_time_ms]

    def totals(self) -> Dict[str, int]:
        return {
            "queries": len(self.query_events),
            "clicks": len(self.click_events),
            "sessions": len(self.sessions),
        }

    def click_through_rate(self) -> Optional[float]:
        query_count = len(self.query_events)
        if not query_count:
            return None
        return len(self.click_events) / query_count

    def average_results_per_query(self) -> Optional[float]:
        if not self.query_events:
            return None
        total_results = sum(len(self.query_result_map.get(event.query_id, {})) for event in self.query_events)
        return total_results / len(self.query_events)

    def average_dwell_time_ms(self) -> Optional[float]:
        dwell = self.dwell_times()
        if not dwell:
            return None
        return sum(dwell) / len(dwell)

    def algorithm_distribution(self) -> Dict[str, int]:
        counter = Counter()
        for event in self.query_events:
            counter[event.algorithm or "unknown"] += 1
        return dict(counter)

    def search_volume_series(self, days: int = 7) -> List[Dict[str, Any]]:
        if days <= 0:
            return []
        today = datetime.utcnow().date()
        buckets: Dict[str, int] = { (today - timedelta(days=offset)).isoformat(): 0 for offset in range(days - 1, -1, -1)}
        for event in self.query_events:
            label = event.timestamp.date().isoformat()
            if label in buckets:
                buckets[label] += 1
        return [{"date": date_label, "count": buckets[date_label]} for date_label in buckets]

    def fact_query_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for event in self.query_events:
            result_count = len(self.query_result_map.get(event.query_id, {}))
            rows.append(
                {
                    "query_id": event.query_id,
                    "session_id": event.session_id,
                    "request_id": None,
                    "query_text": event.text,
                    "term_count": event.term_count,
                    "ranking_algorithm": event.algorithm,
                    "result_count": result_count,
                    "timestamp": event.timestamp.isoformat(),
                    "browser": event.browser,
                    "os_name": event.os,
                    "ip_address": event.ip,
                }
            )
        return rows

    def fact_click_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for event in self.click_events:
            rows.append(
                {
                    "query_id": event.query_id,
                    "session_id": event.session_id,
                    "doc_id": event.doc_id,
                    "rank_position": event.rank,
                    "click_ts": event.timestamp.isoformat(),
                    "dwell_time_ms": event.dwell_time_ms,
                }
            )
        return rows

    def dim_session_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for session in self.sessions.values():
            rows.append(
                {
                    "session_id": session.session_id,
                    "ip_address": session.ip,
                    "browser": session.browser,
                    "os_name": session.os,
                    "user_agent": session.user_agent,
                    "first_seen_ts": session.first_seen.isoformat(),
                    "last_seen_ts": session.last_seen.isoformat(),
                    "query_count": len(session.queries),
                }
            )
        return rows

    def export_table(self, table_name: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        table_map = {
            "fact_query": (
                [
                    "query_id",
                    "session_id",
                    "request_id",
                    "query_text",
                    "term_count",
                    "ranking_algorithm",
                    "result_count",
                    "timestamp",
                    "browser",
                    "os_name",
                    "ip_address",
                ],
                self.fact_query_rows(),
            ),
            "fact_click": (
                [
                    "query_id",
                    "session_id",
                    "doc_id",
                    "rank_position",
                    "click_ts",
                    "dwell_time_ms",
                ],
                self.fact_click_rows(),
            ),
            "dim_session": (
                [
                    "session_id",
                    "ip_address",
                    "browser",
                    "os_name",
                    "user_agent",
                    "first_seen_ts",
                    "last_seen_ts",
                    "query_count",
                ],
                self.dim_session_rows(),
            ),
        }

        if table_name not in table_map:
            raise ValueError(f"Unknown analytics table '{table_name}'")
        return table_map[table_name]


class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        return json.dumps(self)
