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
  total_count?: number;
  updated_cells?: Array<{ row_id: number; column: string; old_value: string | null; new_value: string | null }>;
  undo_count?: number;  // Number of available undo levels (0-10)
  error?: string | null;
};

type UndoResponse = {
  success: boolean;
  dataset_id: string;
  session: string;
  columns: Record<string, string>;
  row_count: number;
  preview: Array<Record<string, unknown>>;
  message: string;
  undo_count: number;  // Remaining undo levels after this undo
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
    { role: "user" | "assistant"; text: string; kind?: "code" | "text" | "error" | "preview" | "update"; previewData?: Array<Record<string, unknown>>; totalCount?: number; activeCode?: string; updatedCount?: number }[]
  >([]);
  const [showLoadModal, setShowLoadModal] = useState(false);
  // Highlighted cells state: Map of "rowId-colName" -> color
  const [highlightedCells, setHighlightedCells] = useState<Map<string, string>>(new Map());
  // Stack of highlight keys for each operation (for proper undo)
  // Each entry is an array of cell keys that were highlighted in that operation
  const [highlightStack, setHighlightStack] = useState<string[][]>([]);
  // Filter-related state
  const [filteredTotalCount, setFilteredTotalCount] = useState(0);
  // Raw preview with __row_id for highlighting
  const [rawPreview, setRawPreview] = useState<Array<Record<string, unknown>>>([]);
  // Track updated row IDs for "View Updated Records" button
  const [updatedRowIds, setUpdatedRowIds] = useState<Set<number>>(new Set());
  // Track updated column for each update operation
  const [lastUpdatedColumn, setLastUpdatedColumn] = useState<string | null>(null);
  // Undo state: number of available undo levels (0-10)
  const [undoCount, setUndoCount] = useState(0);

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

  // Random highlight colors for updated cells
  const getRandomHighlightColor = () => {
    const colors = [
      'rgba(255, 182, 193, 0.6)', // light pink
      'rgba(173, 216, 230, 0.6)', // light blue
      'rgba(144, 238, 144, 0.6)', // light green
      'rgba(255, 255, 224, 0.6)', // light yellow
      'rgba(230, 230, 250, 0.6)', // lavender
      'rgba(255, 218, 185, 0.6)', // peach
      'rgba(175, 238, 238, 0.6)', // pale turquoise
      'rgba(255, 160, 122, 0.6)', // light salmon
    ];
    return colors[Math.floor(Math.random() * colors.length)];
  };

  // Handle "Show All" button click from chat preview (shows filtered data)
  const handleShowAllInTable = (code: string | null) => {
    if (session) {
      setActiveCode(code);
      fetchPage(0, limit, session, code);
    }
  };

  // Handle "Show All Data" button - reset to full dataset view (keeping chat history)
  const handleShowAllData = async () => {
    if (session) {
      setActiveCode(null); // Clear filter code
      await fetchPage(0, limit, session, null); // Load original data
    }
  };

  // Handle "View Updated Records" button - shows data with highlights visible
  const handleViewUpdatedRows = async () => {
    if (session) {
      setActiveCode(null); // Show all data
      await fetchPage(0, limit, session, null); // Load original data - highlights will show based on row_ids
    }
  };

  // Clear all highlights (for when user wants to reset)
  const handleClearHighlights = () => {
    setHighlightedCells(new Map());
    setUpdatedRowIds(new Set());
    setHighlightStack([]); // Clear the stack too
  };

  // Handle Undo button - restore previous state and remove ONLY last operation's highlights
  const handleUndo = async () => {
    if (!datasetId || !session) {
      setError("No dataset loaded");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data: UndoResponse = await fetchJson("/dataset/undo", {
        dataset_id: datasetId,
        session,
      });
      if (data.success) {
        // Get current highlight stack to compute new values
        const currentStack = [...highlightStack];

        if (currentStack.length > 0) {
          // Get keys from the last operation to remove
          const lastOperationKeys = currentStack[currentStack.length - 1];
          const newStack = currentStack.slice(0, -1);

          // Build new highlighted cells map without the undone operation's keys
          const newHighlightedCells = new Map(highlightedCells);
          lastOperationKeys.forEach((key) => newHighlightedCells.delete(key));

          // Rebuild row IDs from remaining stack
          const newRowIds = new Set<number>();
          newStack.forEach((opKeys) => {
            opKeys.forEach((key) => {
              const rowId = parseInt(key.split("-")[0], 10);
              if (!isNaN(rowId)) newRowIds.add(rowId);
            });
          });

          // Apply all state updates
          setHighlightStack(newStack);
          setHighlightedCells(newHighlightedCells);
          setUpdatedRowIds(newRowIds);
        }

        // Update session and data
        setSession(data.session);
        setColumns(stripColumns(data.columns || {}));
        setRowCount(data.row_count || 0);
        setRawPreview(data.preview || []);
        setPreview(stripRowId(data.preview || []));
        setUndoCount(data.undo_count || 0);

        // Add undo confirmation message to chat
        setMessages((m) => [
          ...m,
          {
            role: "assistant",
            text: data.message || "Changes reverted successfully.",
            kind: "text",
          },
        ]);

        // Refresh full page data
        if (data.session) {
          await fetchPage(0, limit, data.session, null);
        }
      } else {
        setError(data.message || "Undo failed");
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
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
      setRawPreview(data.preview || []);
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

      // Check if this is an update operation (has updated_cells)
      const isUpdateOperation = data.updated_cells && data.updated_cells.length > 0;

      // Handle updated cells highlighting - use SAME color for all cells in one operation
      if (isUpdateOperation) {
        console.log("Updated cells received:", data.updated_cells); // Debug log

        const operationColor = getRandomHighlightColor(); // Pick ONE color for this entire operation
        const newHighlights = new Map<string, string>();
        const newUpdatedRowIds = new Set<number>();
        const operationKeys: string[] = []; // Track keys for this operation (for undo)

        data.updated_cells!.forEach((cell) => {
          const key = `${cell.row_id}-${cell.column}`;
          newHighlights.set(key, operationColor); // Same color for all cells
          newUpdatedRowIds.add(cell.row_id);
          operationKeys.push(key); // Store for undo stack
        });

        // Track updated column name
        if (data.updated_cells!.length > 0) {
          setLastUpdatedColumn(data.updated_cells![0].column);
        }

        // Merge with existing updated row IDs (for tracking across operations)
        setUpdatedRowIds((prev) => {
          const merged = new Set(prev);
          newUpdatedRowIds.forEach((id) => merged.add(id));
          return merged;
        });

        // For mutations, reload data FIRST to get updated __row_id values
        if (session) {
          await fetchPage(0, limit, session, null); // Load original data with new values
        }

        // THEN set highlights (after rawPreview is updated) - PERSISTS FOR SESSION (no auto-clear)
        setHighlightedCells((prev) => {
          // Merge with existing highlights (different operations keep their colors)
          const merged = new Map(prev);
          newHighlights.forEach((color, key) => merged.set(key, color));
          console.log("Highlighting cells:", Array.from(merged.keys())); // Debug log
          return merged;
        });

        // Push this operation's keys to the highlight stack (for proper undo)
        setHighlightStack((prevStack) => [...prevStack, operationKeys]);
      }

      // Store filtered total count
      const totalCount = data.total_count || 0;
      setFilteredTotalCount(totalCount);

      // Add user message
      setMessages((m) => [
        ...m,
        { role: "user", text: question, kind: "text" },
      ]);

      // Add assistant response - either error, update, or preview
      if (data.error) {
        setMessages((m) => [
          ...m,
          { role: "assistant", text: data.error || "", kind: "error" },
        ]);
      } else if (isUpdateOperation) {
        // For updates, show update count with View Updated Records button
        const updateCount = data.updated_cells!.length;
        setMessages((m) => [
          ...m,
          {
            role: "assistant",
            text: data.code || "",
            kind: "update",
            updatedCount: updateCount,
          },
        ]);
        // Set undo count from backend response
        setUndoCount(data.undo_count || 0);
      } else {
        // For filters/queries, show preview message with Show All button data
        const chatPreviewRows = stripRowId(data.preview || []).slice(0, 10);
        setMessages((m) => [
          ...m,
          {
            role: "assistant",
            text: data.code || "",
            kind: "preview",
            previewData: chatPreviewRows,
            totalCount: totalCount,
            activeCode: data.code || undefined,
          },
        ]);
      }

      // NOTE: Do NOT auto-load filtered data in table - user must click "Show All"
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
      setRawPreview(data.rows || []);
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
      setRawPreview(data.preview || []);
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
          <div className="flex-1 overflow-auto px-4 py-4 space-y-3 bg-slate-50 thin-scrollbar">
            {messages.length === 0 && (
              <div className="text-xs text-slate-400 text-center mt-6">Ask a question to transform the dataset.</div>
            )}
            {messages.map((m, idx) => {
              const isUser = m.role === "user";
              const isCode = m.kind === "code";
              const isError = m.kind === "error";
              const isPreview = m.kind === "preview";
              const isUpdate = m.kind === "update";
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
                    {isUser ? "You" : isError ? "Error" : isUpdate ? "Updated" : isPreview ? "Results" : isCode ? "Generated Polars" : "Assistant"}
                  </div>
                  {isUpdate ? (
                    <div className="space-y-2">
                      {/* Update success message */}
                      <div className="flex items-center gap-2 text-green-700 bg-green-50 px-3 py-2 rounded-lg">
                        <span className="text-lg">✓</span>
                        <span className="font-semibold">{m.updatedCount?.toLocaleString()} cells updated successfully</span>
                      </div>

                      {/* Generated code (collapsed) */}
                      <details className="text-xs">
                        <summary className="cursor-pointer text-slate-500 hover:text-slate-700">View generated code</summary>
                        <div className="rounded-lg bg-slate-900 text-slate-100 font-mono text-xs px-3 py-2 mt-1 whitespace-pre overflow-auto">
                          {m.text}
                        </div>
                      </details>

                    </div>
                  ) : isCode ? (
                    <div className="rounded-xl bg-slate-900 text-slate-100 font-mono text-xs px-3 py-2 whitespace-pre overflow-auto shadow-inner">
                      {m.text}
                    </div>
                  ) : isPreview && m.previewData ? (
                    <div className="space-y-2">
                      {/* Generated code (collapsed) */}
                      <details className="text-xs">
                        <summary className="cursor-pointer text-slate-500 hover:text-slate-700">View generated code</summary>
                        <div className="rounded-lg bg-slate-900 text-slate-100 font-mono text-xs px-3 py-2 mt-1 whitespace-pre overflow-auto">
                          {m.text}
                        </div>
                      </details>

                      {/* Mini preview table */}
                      {m.previewData.length > 0 ? (
                        <div className="overflow-auto max-h-[250px] rounded-lg border border-slate-200 thin-scrollbar">
                          <table className="min-w-full text-xs border-collapse">
                            <thead className="sticky top-0 bg-slate-100">
                              <tr>
                                {Object.keys(m.previewData[0]).map((col) => (
                                  <th key={col} className="border-b border-slate-200 px-2 py-1 text-left font-semibold text-slate-600 whitespace-nowrap">
                                    {col}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {m.previewData.map((row, rowIdx) => (
                                <tr key={rowIdx} className={rowIdx % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                                  {Object.keys(m.previewData![0]).map((col) => (
                                    <td key={col} className="border-b border-slate-100 px-2 py-1 whitespace-nowrap text-slate-700">
                                      {formatValue(row[col])}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <div className="text-slate-500 text-xs py-2">No matching rows found.</div>
                      )}

                      {/* Show All button - only enabled for last message */}
                      {m.totalCount !== undefined && m.totalCount > 0 && (
                        <button
                          onClick={() => handleShowAllInTable(m.activeCode || null)}
                          disabled={idx !== messages.length - 1}
                          className={clsx(
                            "w-full mt-2 rounded-lg text-white text-xs px-3 py-2 font-semibold transition-colors",
                            idx === messages.length - 1
                              ? "bg-indigo-600 hover:bg-indigo-700 cursor-pointer"
                              : "bg-slate-400 cursor-not-allowed opacity-50"
                          )}
                        >
                          Show All ({m.totalCount.toLocaleString()} rows) in Table
                        </button>
                      )}
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

                <button
                  className="rounded-lg border border-indigo-300 bg-indigo-50 px-3 py-1 font-semibold text-indigo-700 hover:bg-indigo-100 disabled:opacity-50"
                  onClick={handleShowAllData}
                  disabled={loading || paging}
                >
                  Show All Data
                </button>
                {/* Undo button - shown when undo is available */}
                <button
                  className={clsx(
                    "rounded-lg px-3 py-1 font-semibold flex items-center gap-1 disabled:opacity-50 transition-colors",
                    undoCount > 0
                      ? "border border-amber-400 bg-amber-50 text-amber-700 hover:bg-amber-100"
                      : "border border-slate-200 bg-slate-100 text-slate-400 cursor-not-allowed"
                  )}
                  onClick={handleUndo}
                  disabled={loading || paging || undoCount === 0}
                  title={undoCount > 0 ? `Undo (${undoCount} level${undoCount > 1 ? 's' : ''} available)` : "No undo available"}
                >
                  <span>↩</span>
                  <span>Undo{undoCount > 0 ? ` (${undoCount})` : ""}</span>
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
                      {(() => {
                        // Create pairs of [preview row, raw row] for sorting
                        const rowPairs = preview.map((row, idx) => ({
                          row,
                          rawRow: rawPreview[idx],
                          idx,
                        }));

                        // Sort to put highlighted rows at the top
                        const sortedPairs = [...rowPairs].sort((a, b) => {
                          const aRowId = a.rawRow?.__row_id;
                          const bRowId = b.rawRow?.__row_id;

                          // Check if any column in this row is highlighted
                          const aHasHighlight = Object.keys(preview[0] || {}).some(
                            (col) => highlightedCells.has(`${aRowId}-${col}`)
                          );
                          const bHasHighlight = Object.keys(preview[0] || {}).some(
                            (col) => highlightedCells.has(`${bRowId}-${col}`)
                          );

                          if (aHasHighlight && !bHasHighlight) return -1;
                          if (!aHasHighlight && bHasHighlight) return 1;
                          return 0;
                        });

                        return sortedPairs.map(({ row, rawRow, idx: originalIdx }, sortedIdx) => {
                          const rowId = rawRow?.__row_id;
                          return (
                            <tr key={originalIdx} className={sortedIdx % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                              {Object.keys(preview[0]).map((col) => {
                                const cellKey = `${rowId}-${col}`;
                                const highlightColor = highlightedCells.get(cellKey);
                                return (
                                  <td
                                    key={col}
                                    className="border border-slate-200 px-4 py-2 align-top text-slate-800 whitespace-nowrap min-w-[160px] transition-colors duration-300"
                                    style={highlightColor ? { backgroundColor: highlightColor } : undefined}
                                  >
                                    {formatValue(row[col])}
                                  </td>
                                );
                              })}
                            </tr>
                          );
                        });
                      })()}
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


