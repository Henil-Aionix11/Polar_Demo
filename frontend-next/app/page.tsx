"use client";

import { useEffect, useState } from "react";
import clsx from "classnames";

type LoadResponse = {
  dataset_id?: string;
  session: string;
  columns: Record<string, string>;
  row_count: number;
  preview: Array<Record<string, unknown>>;
};

type NLExprResponse = {
  code: string;
  preview: Array<Record<string, unknown>>;
  row_count?: number;
  error?: string | null;
};

type PageResponse = {
  rows: Array<Record<string, unknown>>;
  total: number;
};

type OpenResponse = LoadResponse;

const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const DEFAULT_PAGE_SIZE = 500;
const STORAGE_DATASET_ID = "dataset_id";

export default function Page() {
  const [s3Path, setS3Path] = useState("s3://training-data-kg/100mb.xlsx");
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [session, setSession] = useState<string | null>(null);
  const [columns, setColumns] = useState<Record<string, string>>({});
  const [rowCount, setRowCount] = useState(0);
  const [question, setQuestion] = useState("");
  const [code, setCode] = useState("--");
  const [preview, setPreview] = useState<Array<Record<string, unknown>>>([]);
  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState(DEFAULT_PAGE_SIZE);
  const [activeCode, setActiveCode] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [paging, setPaging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<
    { role: "user" | "assistant"; text: string; kind?: "code" | "text" | "error" }[]
  >([]);
  const [showLoadModal, setShowLoadModal] = useState(false);

  const totalPages = Math.max(1, Math.ceil((rowCount || preview.length || 1) / limit));
  const currentPage = Math.floor(offset / limit) + 1;

  const formatValue = (value: unknown) => {
    if (value === null || value === undefined) return "";
    if (Array.isArray(value)) return JSON.stringify(value);
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  };

  const stripRowId = (rows: Array<Record<string, unknown>>) =>
    rows.map((r) => {
      const { __row_id, ...rest } = r;
      return rest;
    });

  const stripColumns = (cols: Record<string, string>) => {
    const { __row_id, ...rest } = cols;
    return rest;
  };

  const fetchJson = async (path: string, body: any) => {
    const res = await fetch(`${apiBase}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Request failed");
    }
    return data;
  };

  const loadDataset = async () => {
    setLoading(true);
    setError(null);
    try {
      const data: LoadResponse = await fetchJson("/dataset/load", { path: s3Path });
      setDatasetId(data.dataset_id || null);
      if (data.dataset_id) {
        localStorage.setItem(STORAGE_DATASET_ID, data.dataset_id);
      }
      setSession(data.session);
      setColumns(stripColumns(data.columns || {}));
      setRowCount(data.row_count || 0);
      setPreview(stripRowId(data.preview || []));
      setCode("--");
      setActiveCode(null);
      setMessages([]);
      setOffset(0);
      setLimit(DEFAULT_PAGE_SIZE);
      if (data.session) {
        await fetchPage(0, DEFAULT_PAGE_SIZE, data.session, null);
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const runNLQ = async () => {
    if (!session) {
      setError("Load a dataset first");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data: NLExprResponse = await fetchJson("/nlq/expr", {
        session,
        question,
      });
      if (data.error) {
        setError(data.error);
      }
      setCode(data.code || "--");
      setActiveCode(data.code || null);
      setPreview(stripRowId(data.preview || []));
      if (data.row_count !== undefined) {
        setRowCount(data.row_count);
      }
      setMessages((m) => [
        ...m,
        { role: "user", text: question, kind: "text" },
        { role: "assistant", text: data.code || data.error || "", kind: data.error ? "error" : "code" },
      ]);
      if (session && !data.error) {
        const requestedLimit = Math.min(rowCount || limit || DEFAULT_PAGE_SIZE, DEFAULT_PAGE_SIZE);
        await fetchPage(0, requestedLimit, session, data.code || null);
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchPage = async (
    newOffset: number,
    newLimit?: number,
    explicitSession?: string,
    codeOverride?: string | null
  ) => {
    const activeSession = explicitSession || session;
    if (!activeSession) {
      setError("Load a dataset first");
      return;
    }
    const effectiveLimit = newLimit ?? limit;
    setPaging(true);
    setError(null);
    try {
      const codeToUse = codeOverride !== undefined ? codeOverride : activeCode;
      const data: PageResponse = await fetchJson("/dataset/page", {
        session: activeSession,
        offset: newOffset,
        limit: effectiveLimit,
        code: codeToUse || undefined,
      });
      setPreview(stripRowId(data.rows || []));
      setRowCount(data.total || rowCount);
      setOffset(newOffset);
      if (codeOverride !== undefined) {
        setActiveCode(codeOverride || null);
      }
      if (newLimit !== undefined) {
        setLimit(newLimit);
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setPaging(false);
    }
  };

  const openDataset = async (id: string) => {
    setLoading(true);
    setError(null);
    try {
      const data: OpenResponse = await fetchJson("/dataset/open", { dataset_id: id });
      setDatasetId(data.dataset_id || id);
      if (data.dataset_id) {
        localStorage.setItem(STORAGE_DATASET_ID, data.dataset_id);
      }
      setSession(data.session);
      setColumns(stripColumns(data.columns || {}));
      setRowCount(data.row_count || 0);
      setPreview(stripRowId(data.preview || []));
      setCode("--");
      setActiveCode(null);
      setMessages([]);
      setOffset(0);
      setLimit(DEFAULT_PAGE_SIZE);
      if (data.session) {
        await fetchPage(0, DEFAULT_PAGE_SIZE, data.session, null);
      }
    } catch (e: any) {
      localStorage.removeItem(STORAGE_DATASET_ID);
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const storedId = typeof window !== "undefined" ? localStorage.getItem(STORAGE_DATASET_ID) : null;
    if (storedId) {
      void openDataset(storedId);
    }
  }, []);

  return (
    <div className="h-screen bg-white overflow-hidden">
      <main className="h-full grid grid-cols-1 lg:grid-cols-[440px_1fr] gap-4 p-4">
        {/* Left rail: chat only */}
        <aside className="rounded-2xl border border-slate-100 shadow-[0_10px_30px_rgba(15,23,42,0.04)] bg-white flex flex-col h-full overflow-hidden">
          <div className="flex-1 overflow-auto px-4 py-4 space-y-3 bg-slate-50">
            {messages.length === 0 && (
              <div className="text-xs text-slate-400 text-center mt-6">Ask a question to transform the dataset.</div>
            )}
            {messages.map((m, idx) => {
              const isUser = m.role === "user";
              const isCode = m.kind === "code";
              const isError = m.kind === "error";
              return (
                <div
                  key={idx}
                  className={clsx(
                    "max-w-[92%] rounded-2xl px-4 py-3 text-sm shadow-sm border",
                    isUser
                      ? "ml-auto bg-slate-900 text-white border-slate-900/70"
                      : "mr-auto bg-white text-slate-800 border-slate-200"
                  )}
                >
                  <div className="text-[11px] uppercase tracking-wide text-slate-400 mb-1">
                    {isUser ? "You" : isError ? "Error" : isCode ? "Generated Polars" : "Assistant"}
                  </div>
                  {isCode ? (
                    <div className="rounded-xl bg-slate-900 text-slate-100 font-mono text-xs px-3 py-2 whitespace-pre overflow-auto shadow-inner">
                      {m.text}
                    </div>
                  ) : (
                    <div className="whitespace-pre-wrap leading-relaxed">{m.text}</div>
                  )}
                </div>
              );
            })}
          </div>
          <div className="p-4 border-t border-slate-200 bg-white">
            <div className="text-xs font-semibold text-slate-600 mb-2">Natural language query</div>
            <div className="relative">
              <textarea
                className="w-full rounded-2xl border border-slate-200 px-3 py-3 pr-24 text-sm min-h-[90px] focus:ring-2 focus:ring-slate-200 focus:border-slate-300 bg-slate-50"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Type a question or transformation request..."
              />
              <div className="absolute inset-y-3 right-3 flex items-center gap-2">
                <button
                  className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"
                  onClick={() => setShowLoadModal(true)}
                >
                  Upload
                </button>
                <button
                  className="rounded-xl bg-slate-900 text-white text-xs px-3 py-2 font-semibold hover:bg-black disabled:opacity-50"
                  onClick={runNLQ}
                  disabled={loading}
                >
                  {loading ? "..." : "Send"}
                </button>
              </div>
            </div>
            {/* <div className="flex gap-2 mt-3">
              <button
                className="flex-1 rounded-xl bg-slate-900 text-white text-sm px-4 py-3 font-semibold hover:bg-black shadow-sm"
                onClick={runNLQ}
                disabled={loading}
              >
                {loading ? "Generating..." : "Generate & Preview"}
              </button>
              <button
                className="rounded-xl bg-slate-100 text-slate-700 text-sm px-4 py-3 font-semibold border border-slate-200"
                onClick={() => {
                  setPreview([]);
                  setCode("--");
                  setQuestion("");
                  setActiveCode(null);
                  setMessages([]);
                }}
              >
                Clear
              </button>
            </div> */}
          </div>
        </aside>

        {/* Right rail: preview table only */}
        <section className="rounded-2xl border border-slate-100 shadow-[0_10px_30px_rgba(15,23,42,0.04)] bg-white flex flex-col h-full overflow-hidden">
          <div className="p-4 border-b border-slate-200 bg-slate-50">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <div className="flex items-center gap-2 text-sm text-slate-700">
                <span className="px-3 py-1 rounded-full border border-slate-200 bg-white">
                  Showing {preview.length} / {rowCount || preview.length} rows
                </span>
                <span className="px-3 py-1 rounded-full border border-slate-200 bg-white">
                  Cols: {Object.keys(columns).length || (preview[0] ? Object.keys(preview[0]).length : 0)}
                </span>
                <span className="px-3 py-1 rounded-full border border-slate-200 bg-white">
                  Page {currentPage} / {totalPages}
                </span>
                <div className="flex items-center gap-2">
                  <button
                    className="rounded-lg border border-slate-300 bg-white px-3 py-1 text-xs font-semibold text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                    onClick={() => fetchPage(Math.max(0, offset - limit))}
                    disabled={paging || offset === 0}
                  >
                    Prev
                  </button>
                  <button
                    className="rounded-lg border border-slate-300 bg-white px-3 py-1 text-xs font-semibold text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                    onClick={() => fetchPage(offset + limit)}
                    disabled={paging || offset + limit >= (rowCount || preview.length)}
                  >
                    Next
                  </button>
                </div>
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-600">
                <span className="px-3 py-1 rounded-full border border-slate-200 bg-white">{s3Path || "No source set"}</span>
                <button
                  className="rounded-lg border border-slate-300 bg-white px-3 py-1 font-semibold text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                  onClick={() => loadDataset()}
                  disabled={loading}
                >
                  Refresh
                </button>
              </div>
            </div>
          </div>

          <div className="flex-1 overflow-hidden flex flex-col">
            <div className="flex-1 overflow-auto">
              {preview.length === 0 ? (
                <div className="h-full flex items-center justify-center text-sm text-slate-500 p-6">No data</div>
              ) : (
                <div className="h-full w-full overflow-auto">
                  <table className="min-w-max table-auto text-sm border-collapse">
                    <thead className="sticky top-0 bg-slate-100 shadow-sm z-10">
                      <tr>
                        {Object.keys(preview[0]).map((col) => (
                          <th
                            key={col}
                            className="border border-slate-200 px-4 py-2 text-left text-xs font-semibold text-slate-700 whitespace-nowrap bg-slate-100"
                          >
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {preview.map((row, idx) => (
                        <tr key={idx} className={idx % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                          {Object.keys(preview[0]).map((col) => (
                            <td
                              key={col}
                              className="border border-slate-200 px-4 py-2 align-top text-slate-800 whitespace-nowrap min-w-[160px]"
                            >
                              {formatValue(row[col])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </section>
      </main>

      {showLoadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div className="absolute inset-0 bg-slate-900/50 backdrop-blur-sm" onClick={() => setShowLoadModal(false)} />
          <div className="relative bg-white rounded-2xl shadow-2xl w-full max-w-md p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-slate-900">Load dataset from S3</h3>
              <button
                className="text-slate-500 hover:text-slate-800"
                onClick={() => setShowLoadModal(false)}
              >
                ✕
              </button>
            </div>
            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-600">S3 URI</label>
              <input
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                value={s3Path}
                onChange={(e) => setS3Path(e.target.value)}
                placeholder="s3://bucket/key.xlsx"
              />
            </div>
            <div className="flex justify-end gap-2">
              <button
                className="px-4 py-2 rounded-lg border border-slate-200 text-slate-700 bg-white"
                onClick={() => setShowLoadModal(false)}
              >
                Cancel
              </button>
              <button
                className="px-4 py-2 rounded-lg bg-indigo-600 text-white font-semibold hover:bg-indigo-700"
                onClick={async () => {
                  await loadDataset();
                  setShowLoadModal(false);
                }}
                disabled={loading}
              >
                {loading ? "Loading..." : "Load"}
              </button>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="fixed bottom-4 left-1/2 -translate-x-1/2 bg-rose-50 text-rose-700 border border-rose-200 px-4 py-2 rounded-lg shadow">
          {error}
        </div>
      )}
    </div>
  );
}


