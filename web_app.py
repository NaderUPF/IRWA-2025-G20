import csv
import io
import os
import uuid
import zipfile
from json import JSONEncoder

import httpagentparser  # for getting the user agent as json
from flask import Flask, render_template, session, jsonify, Response, send_file
from flask import request

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***


# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = os.getenv("SECRET_KEY")
# open browser dev tool to see the cookies
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME")
# instantiate our search engine
search_engine = SearchEngine()
# instantiate our in memory persistence
analytics_data = AnalyticsData()
# instantiate RAG generator
rag_generator = RAGGenerator()


def _ensure_session_id():
    if 'analytics_session_id' not in session:
        session['analytics_session_id'] = str(uuid.uuid4())
    return session['analytics_session_id']


def _parse_user_agent(user_agent_raw):
    agent = httpagentparser.detect(user_agent_raw or "")
    browser = agent.get('browser', {}).get('name') if agent.get('browser') else None
    os_name = agent.get('platform', {}).get('name') if agent.get('platform') else None
    return agent, browser, os_name


def _render_csv(fieldnames, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()

# load documents corpus into memory.
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
file_path = path + "/" + os.getenv("DATA_FILE_PATH")
corpus = load_corpus(file_path)
# Log first element of corpus to verify it loaded correctly:
print("\nCorpus is loaded... \n First element:\n", list(corpus.values())[0])


# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests. Example:
    session['some_var'] = "Some value that is kept in session"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent, browser, os_name = _parse_user_agent(user_agent)

    session_id = _ensure_session_id()
    analytics_data.record_request_context(
        session_id,
        ip=user_ip,
        user_agent=user_agent,
        browser=browser,
        os_name=os_name,
    )

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))
    print(session)

    last_algorithm = session.get('last_algorithm', search_engine.default_algorithm)
    last_search_query = session.get('last_search_query', '')

    return render_template(
        'index.html',
        page_title="Welcome",
        selected_algorithm=last_algorithm,
        last_search_query=last_search_query,
    )


@app.route('/search', methods=['POST'])
def search_form_post():
    session_id = _ensure_session_id()
    analytics_data.record_return_to_results(session_id)

    search_query = request.form['search-query']
    ranking_algorithm = request.form.get('ranking-algorithm', search_engine.default_algorithm)

    session['last_search_query'] = search_query
    session['last_algorithm'] = ranking_algorithm

    user_agent = request.headers.get('User-Agent')
    user_ip = request.remote_addr
    _, browser, os_name = _parse_user_agent(user_agent)

    query_id = analytics_data.save_query_terms(
        search_query,
        algorithm=ranking_algorithm,
        session_id=session_id,
        user_agent=user_agent,
        browser=browser,
        os_name=os_name,
        ip=user_ip,
    )

    results = search_engine.search(search_query, query_id, corpus, algorithm=ranking_algorithm)
    analytics_data.register_results(query_id, results)
    session['last_query_id'] = query_id

    # generate RAG response based on user query and retrieved results
    rag_response = rag_generator.generate_response(search_query, results)
    print("RAG response:", rag_response)

    found_count = len(results)
    session['last_found_count'] = found_count

    print(session)

    return render_template(
        'results.html',
        results_list=results,
        page_title="Results",
        found_counter=found_count,
        rag_response=rag_response,
        selected_algorithm=ranking_algorithm,
        search_query=search_query,
        query_id=query_id,
    )


@app.route('/doc_details', methods=['GET'])
def doc_details():
    clicked_doc_id = request.args.get("pid")
    if not clicked_doc_id:
        return render_template('doc_details.html', document=None, error="Missing document identifier."), 400

    print("doc details session: ")
    print(session)

    if clicked_doc_id not in corpus:
        return render_template('doc_details.html', document=None, error="Document not found."), 404

    document: Document = corpus[clicked_doc_id]

    query_id_param = request.args.get("search_id")
    try:
        query_id = int(query_id_param) if query_id_param else None
    except ValueError:
        query_id = None

    session_id = _ensure_session_id()
    analytics_data.record_click(query_id or session.get('last_query_id'), clicked_doc_id, session_id)

    return render_template('doc_details.html', document=document, query_id=query_id)


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with yourdashboard ###
    :return:
    """

    docs = []
    for doc_id in analytics_data.fact_clicks:
        row: Document = corpus[doc_id]
        count = analytics_data.fact_clicks[doc_id]
        doc = StatsDocument(pid=row.pid, title=row.title, description=row.description, url=row.url, count=count)
        docs.append(doc)
    
    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    avg_dwell = analytics_data.dwell_times()
    avg_dwell_ms = sum(avg_dwell) / len(avg_dwell) if avg_dwell else None

    return render_template(
        'stats.html',
        page_title="Analytics",
        clicks_data=docs,
        top_queries=analytics_data.top_queries(),
        top_docs=analytics_data.top_clicked_docs(),
        browser_stats=analytics_data.browser_stats(),
        os_stats=analytics_data.os_stats(),
        click_distribution=analytics_data.click_distribution_by_rank(),
        avg_dwell_ms=avg_dwell_ms,
        totals={
            'queries': len(analytics_data.query_events),
            'clicks': len(analytics_data.click_events),
            'sessions': len(analytics_data.sessions),
        }
    )


@app.route('/dashboard', methods=['GET'])
def dashboard():
    visited_docs = []
    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = corpus[doc_id]
        doc = ClickedDoc(doc_id, d.description, analytics_data.fact_clicks[doc_id])
        visited_docs.append(doc)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)

    totals = analytics_data.totals()
    ctr = analytics_data.click_through_rate()
    avg_results = analytics_data.average_results_per_query()
    avg_dwell = analytics_data.average_dwell_time_ms()
    algo_mix = analytics_data.algorithm_distribution()
    volume_series = analytics_data.search_volume_series(days=7)
    click_distribution = analytics_data.click_distribution_by_rank()
    browser_stats = analytics_data.browser_stats()
    os_stats = analytics_data.os_stats()

    usage_highlights = []
    if ctr is not None:
        usage_highlights.append(f"{ctr * 100:.1f}% of searches lead to at least one click.")
    if avg_dwell is not None:
        usage_highlights.append(f"Average dwell time on a document is {(avg_dwell / 1000):.1f} seconds before users return to results.")
    if avg_results is not None:
        usage_highlights.append(f"Each query returns roughly {avg_results:.1f} documents, giving users a meaningful comparison set.")
    if not usage_highlights:
        usage_highlights.append("Not enough interactions yet. Run a few searches to populate the dashboard.")

    return render_template(
        'dashboard.html',
        page_title="Analytics Dashboard",
        visited_docs=visited_docs,
        top_queries=analytics_data.top_queries(),
        click_distribution=click_distribution,
        browser_stats=browser_stats,
        os_stats=os_stats,
        totals=totals,
        ctr=ctr,
        avg_results=avg_results,
        avg_dwell=avg_dwell,
        algo_mix=algo_mix,
        volume_series=volume_series,
        usage_highlights=usage_highlights,
    )


# New route added for generating an examples of basic Altair plot (used for dashboard)
@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    return analytics_data.plot_number_of_views()


@app.route('/analytics/export', methods=['GET'])
def export_analytics():
    query_payload = [
        {
            "query_id": event.query_id,
            "text": event.text,
            "term_count": event.term_count,
            "timestamp": event.timestamp.isoformat(),
            "algorithm": event.algorithm,
            "session_id": event.session_id,
            "browser": event.browser,
            "os": event.os,
            "ip": event.ip,
        }
        for event in analytics_data.query_events
    ]

    click_payload = [
        {
            "query_id": event.query_id,
            "doc_id": event.doc_id,
            "rank": event.rank,
            "timestamp": event.timestamp.isoformat(),
            "session_id": event.session_id,
            "dwell_time_ms": event.dwell_time_ms,
        }
        for event in analytics_data.click_events
    ]

    sessions_payload = [
        {
            "session_id": session_ctx.session_id,
            "ip": session_ctx.ip,
            "browser": session_ctx.browser,
            "os": session_ctx.os,
            "first_seen": session_ctx.first_seen.isoformat(),
            "last_seen": session_ctx.last_seen.isoformat(),
            "queries": session_ctx.queries,
        }
        for session_ctx in analytics_data.sessions.values()
    ]

    return jsonify(
        {
            "queries": query_payload,
            "clicks": click_payload,
            "sessions": sessions_payload,
        }
    )


@app.route('/analytics/export/csv', methods=['GET'])
def export_analytics_csv():
    table = request.args.get('table')
    valid_tables = ('fact_query', 'fact_click', 'dim_session')

    if table and table not in valid_tables and table != 'all':
        return jsonify({"error": f"Invalid table '{table}'. Choose from {valid_tables} or 'all'."}), 400

    def _single_table_response(table_name: str):
        fieldnames, rows = analytics_data.export_table(table_name)
        csv_data = _render_csv(fieldnames, rows)
        response = Response(csv_data, mimetype='text/csv')
        response.headers['Content-Disposition'] = f'attachment; filename={table_name}.csv'
        return response

    if table in valid_tables:
        return _single_table_response(table)

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zf:
        for table_name in valid_tables:
            fieldnames, rows = analytics_data.export_table(table_name)
            csv_data = _render_csv(fieldnames, rows)
            zf.writestr(f"{table_name}.csv", csv_data)
    archive.seek(0)

    return send_file(
        archive,
        mimetype='application/zip',
        as_attachment=True,
        download_name='analytics_export.zip',
    )


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"))
