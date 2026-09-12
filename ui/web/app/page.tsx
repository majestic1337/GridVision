"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type ChatSummary = {
  id: string;
  title?: string | null;
  created_at: string;
  updated_at: string;
};

type Source = {
  chunk_id: string;
  doc_slug: string;
  title?: string | null;
  source_uri?: string | null;
  page_label: string;
  content_preview: string;
  score: number;
  type: string;
  chunk_index?: number | null;
};

type Attachment = {
  asset_id: string;
  type: string;
  file_path: string;
  page_label: string;
  element_id: string;
  url: string;
  score?: number | null;
  doc_slug?: string | null;
  title?: string | null;
  source_uri?: string | null;
  doc_id?: string | null;
};

type Message = {
  id: string;
  chat_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  sources?: Source[] | null;
  attachments?: Attachment[] | null;
};

type DebugInfo = {
  rerank_used?: boolean;
  rerank_active?: boolean;
  rerank_enabled?: boolean;
  rerank_disabled_reason?: string | null;
  rerank_error?: string | null;
  rerank_model?: string | null;
  lcel_used?: boolean;
  image_mode?: boolean;
};

type ModelStatus = {
  key: string;
  label: string;
  ok: boolean;
  detail?: string | null;
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const VISUAL_KEYWORDS = [
  "diagram",
  "schematic",
  "figure",
  "table",
  "image",
  "photo",
  "picture",
  "chart",
  "plot"
];
const THINKING_PHASES = [
  "🔎 Retrieving relevant sources…",
  "🧩 Linking images and tables…",
  "🧠 Reranking context…",
  "✍️ Drafting response…"
];
const buildThinkingContent = (phaseIndex: number) => {
  const lines = ["RAG progress:"];
  THINKING_PHASES.forEach((phase, idx) => {
    if (idx < phaseIndex) {
      lines.push(`✅ ${phase}`);
    } else if (idx === phaseIndex) {
      lines.push(`⏳ ${phase}`);
    } else {
      lines.push(`• ${phase}`);
    }
  });
  return lines.join("\n");
};
const EXAMPLE_QUERIES = [
  "Which figure shows the spiral tapered screw extractor?",
  "What does the manual say about sealing compound?",
  "Show the diagram for a collimating telescope.",
  "What does the table 5-5 indicate about pounds per sq. in.?"
];

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return res.json();
}

function formatTimestamp(ts: string) {
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString();
}

function useAutoScroll(deps: unknown[]) {
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, deps);
  return ref;
}

function isVisualRequest(text: string) {
  const q = text.toLowerCase();
  return VISUAL_KEYWORDS.some((k) => q.includes(k));
}

export default function Home() {
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [lastDebug, setLastDebug] = useState<DebugInfo | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus[]>([]);
  const [expandedImage, setExpandedImage] = useState<{
    url: string;
    label: string;
  } | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const thinkingIntervalRef = useRef<number | null>(null);
  const skipNextLoadRef = useRef(false);

  const containsCyrillic = input.trim().length > 0 && /[\u0400-\u04FF]/.test(input);
  const scrollRef = useAutoScroll([messages, streaming]);

  const activeChat = useMemo(
    () => chats.find((c) => c.id === activeChatId) || null,
    [chats, activeChatId]
  );

  const rerankStatus = useMemo(() => {
    if (!lastDebug) return null;
    if (lastDebug.image_mode) {
      return { label: "Rerank: n/a (image)", detail: null };
    }
    const enabled = lastDebug.rerank_enabled;
    const active = lastDebug.rerank_active;
    const used = lastDebug.rerank_used;
    const detail = lastDebug.rerank_disabled_reason || lastDebug.rerank_error || null;
    let label = "Rerank: ";
    if (used) {
      label += "on";
    } else if (enabled === false) {
      label += "off";
    } else if (active === false) {
      label += "off";
    } else if (lastDebug.rerank_error) {
      label += "error";
    } else {
      label += "off";
    }
    return { label, detail };
  }, [lastDebug]);

  const loadChats = async () => {
    const data = await fetchJson<ChatSummary[]>("/chats");
    setChats(data);
    if (!activeChatId && data.length > 0) {
      setActiveChatId(data[0].id);
    }
  };

  const loadChat = async (chatId: string) => {
    const data = await fetchJson<{ chat: ChatSummary; messages: Message[] }>(
      `/chats/${chatId}`
    );
    setActiveChatId(data.chat.id);
    setMessages(data.messages);
  };

  useEffect(() => {
    loadChats().catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    let active = true;
    const loadStatus = async () => {
      try {
        const data = await fetchJson<{ models: ModelStatus[] }>("/status");
        if (active) setModelStatus(data.models || []);
      } catch {
        if (active) setModelStatus([]);
      }
    };
    loadStatus();
    const interval = window.setInterval(loadStatus, 60000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    if (activeChatId) {
      if (skipNextLoadRef.current) {
        skipNextLoadRef.current = false;
        return;
      }
      loadChat(activeChatId).catch((err) => setError(err.message));
    } else {
      setMessages([]);
    }
  }, [activeChatId]);

  const createChat = async () => {
    const chat = await fetchJson<ChatSummary>("/chats", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    });
    setChats((prev) => [chat, ...prev]);
    setActiveChatId(chat.id);
    setMessages([]);
  };

  const deleteChat = async (chatId: string) => {
    try {
      await fetchJson(`/chats/${chatId}`, { method: "DELETE" });
      const remaining = chats.filter((chat) => chat.id !== chatId);
      setChats(remaining);
      if (activeChatId === chatId) {
        setActiveChatId(remaining[0]?.id ?? null);
        setMessages([]);
      }
    } catch (err: any) {
      setError(err.message || "Failed to delete chat");
    }
  };

  const stopThinking = () => {
    if (thinkingIntervalRef.current !== null) {
      window.clearInterval(thinkingIntervalRef.current);
      thinkingIntervalRef.current = null;
    }
  };

  const startThinking = (assistantLocalId: string) => {
    stopThinking();
    let phaseIndex = 0;
    const updateThinking = (index: number) => {
      const content = buildThinkingContent(index);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantLocalId ? { ...m, content } : m
        )
      );
    };
    updateThinking(phaseIndex);
    thinkingIntervalRef.current = window.setInterval(() => {
      phaseIndex = Math.min(phaseIndex + 1, THINKING_PHASES.length - 1);
      updateThinking(phaseIndex);
    }, 900);
  };

  const handleSend = async () => {
    if ((input.trim().length === 0 && !imageFile) || streaming) return;
    if (containsCyrillic) {
      setError("Only English queries are allowed.");
      return;
    }
    setError(null);
    setStreaming(true);
    setLastDebug(null);

    const content = input.trim();
    const wantsVisuals = isVisualRequest(content);
    setInput("");

    const localUserId = `local-user-${Date.now()}`;
    const localAssistantId = `local-assistant-${Date.now()}`;
    const nowIso = new Date().toISOString();
    const localChatId = activeChatId ?? "local";

    const optimisticUser: Message = {
      id: localUserId,
      chat_id: localChatId,
      role: "user",
      content: content || "[Image prompt]",
      created_at: nowIso
    };

    const optimisticAssistant: Message = {
      id: localAssistantId,
      chat_id: localChatId,
      role: "assistant",
      content: buildThinkingContent(0),
      created_at: nowIso
    };

    setMessages((prev) => [...prev, optimisticUser, optimisticAssistant]);
    startThinking(localAssistantId);

    let chatId = activeChatId;
    if (!chatId) {
      const chat = await fetchJson<ChatSummary>("/chats", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
      });
      setChats((prev) => [chat, ...prev]);
      skipNextLoadRef.current = true;
      chatId = chat.id;
      setActiveChatId(chat.id);
    }

    if (localChatId === "local") {
      setMessages((prev) =>
        prev.map((m) =>
          m.chat_id === "local" ? { ...m, chat_id: chatId as string } : m
        )
      );
    }

    try {
      if (imageFile) {
        const formData = new FormData();
        formData.append("file", imageFile);
        formData.append("content", content);
        formData.append("show_assets", "true");

        const res = await fetch(`${API_BASE}/chats/${chatId}/messages/image`, {
          method: "POST",
          body: formData
        });

        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || "Image request failed");
        }
        const data = await res.json();
        const userMessage = data.user_message as Message;
        const assistantMessage = data.assistant_message as Message;
        setLastDebug(data.debug || null);
        setMessages((prev) =>
          prev.map((m) => {
            if (m.id === localUserId) return userMessage || m;
            if (m.id === localAssistantId) return assistantMessage || m;
            return m;
          })
        );
        stopThinking();
        setStreaming(false);
        setImageFile(null);
        await loadChats();
        return;
      }

      const res = await fetch(`${API_BASE}/chats/${chatId}/messages/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content,
          show_assets: wantsVisuals,
          use_lcel: true
        })
      });

      if (!res.body) {
        throw new Error("Streaming not supported by browser.");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let assistantId = "";

      const updateAssistant = (delta: string) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: m.content + delta } : m
          )
        );
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const evt of events) {
          const line = evt.split("\n").find((l) => l.startsWith("data:"));
          if (!line) continue;
          const payload = line.replace(/^data:\s*/, "");
          if (!payload) continue;
          const data = JSON.parse(payload);

          if (data.type === "start") {
            stopThinking();
            const userMessage = data.user_message as Message;
            assistantId = data.assistant_message_id;
            const assistantMessage: Message = {
              id: assistantId,
              chat_id: chatId,
              role: "assistant",
              content: "",
              created_at: new Date().toISOString()
            };
            setMessages((prev) =>
              prev.map((m) => {
                if (m.id === localUserId) {
                  return userMessage || m;
                }
                if (m.id === localAssistantId) {
                  return assistantMessage;
                }
                return m;
              })
            );
          }

          if (data.type === "token") {
            updateAssistant(data.content || "");
          }

          if (data.type === "final") {
            const assistantMessage = data.assistant_message as Message;
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantMessage.id ? assistantMessage : m))
            );
            setLastDebug(data.debug || null);
            stopThinking();
            setStreaming(false);
            await loadChats();
          }
        }
      }
      stopThinking();
      setStreaming(false);
    } catch (err: any) {
      setError(err.message || "Streaming error");
      stopThinking();
      setStreaming(false);
      setImageFile(null);
      setLastDebug(null);
    }
  };

  return (
    <div className="app-shell">
      {modelStatus.length > 0 && (
        <div className="model-status" aria-label="Model status indicators">
          {modelStatus.map((item) => {
            const title = item.ok
              ? `${item.label}: ok`
              : `${item.label}: ${item.detail || "error"}`;
            return (
              <span
                key={item.key}
                className={`status-dot ${item.ok ? "ok" : "bad"}`}
                title={title}
                aria-label={title}
              />
            );
          })}
        </div>
      )}
      <aside className="sidebar">
        <div className="brand">
          <h1>GridVision Chat</h1>
          <span>ChatGPT-like workspace for schematics & manuals</span>
        </div>
        <button className="new-chat" onClick={createChat}>
          + New chat
        </button>
        <div className="chat-list">
          {chats.map((chat) => (
            <div
              key={chat.id}
              className={`chat-item ${chat.id === activeChatId ? "active" : ""}`}
              onClick={() => setActiveChatId(chat.id)}
            >
              <div className="chat-title">
                {chat.title?.trim() || "Untitled chat"}
              </div>
              <div className="chat-meta">{formatTimestamp(chat.updated_at)}</div>
              <button
                className="chat-delete"
                onClick={(e) => {
                  e.stopPropagation();
                  deleteChat(chat.id);
                }}
                aria-label="Delete chat"
                title="Delete chat"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      </aside>

      <main className="main-panel">
        <section className="chat-window" ref={scrollRef}>
          {messages.length === 0 ? (
            <div className="empty-state">
              <h2>Ask for a diagram or component</h2>
              <p>
                Submit a query to see relevant pages and inline schematics right
                in the chat.
              </p>
            </div>
          ) : (
            <div className="messages">
              {messages.map((msg, idx) => {
                const prevUser = [...messages]
                  .slice(0, idx)
                  .reverse()
                  .find((m) => m.role === "user");
                const attachments = msg.attachments || [];
                const primary = attachments.slice(0, 1);
                const secondary = attachments.slice(1);
                const isUser = msg.role === "user";
                const isTableAsset = (att: Attachment) =>
                  att.file_path.toLowerCase().endsWith(".html") ||
                  att.type.toLowerCase().includes("table");
                const primaryHasTable = primary.some(isTableAsset);
                const secondaryHasTable = secondary.some(isTableAsset);

                const primaryBlock =
                  primary.length > 0 ? (
                    <div
                      className={`attachments ${
                        isUser
                          ? "attachments-user"
                          : primaryHasTable
                          ? "attachments-table"
                          : "attachments-primary"
                      }`}
                    >
                      {primary.map((att) => {
                        const url = `${API_BASE}${att.url}`;
                        const isHtml = att.file_path.toLowerCase().endsWith(".html");
                        const isImage = /\.(png|jpg|jpeg|webp)$/i.test(att.file_path);
                        const isTable = isTableAsset(att);
                        return (
                          <div
                            key={att.asset_id}
                            className={`attachment-card ${
                              isTable ? "attachment-card-table" : ""
                            }`}
                          >
                            {isHtml ? (
                              <iframe
                                src={url}
                                title={att.asset_id}
                                className="attachment-frame"
                              />
                            ) : isImage ? (
                              <img
                                src={url}
                                alt={att.asset_id}
                                onClick={() =>
                                  setExpandedImage({
                                    url,
                                    label: `${att.title || att.doc_slug || "image_asset"} · page ${att.page_label}`
                                  })
                                }
                              />
                            ) : (
                              <div className="attachment-label">
                                Unsupported asset type
                              </div>
                            )}
                            <div className="attachment-label">
                              {att.title || att.doc_slug || "image_asset"} · page {att.page_label}
                            </div>
                            <div className="attachment-label">
                              id {att.asset_id}
                              {typeof att.score === "number" ? ` · score ${att.score.toFixed(3)}` : ""}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : null;

                const secondaryBlock =
                  secondary.length > 0 ? (
                    <details className="sources">
                      <summary>Additional visuals</summary>
                      <div
                        className={`attachments ${
                          secondaryHasTable ? "attachments-table" : ""
                        }`}
                      >
                        {secondary.map((att) => {
                          const url = `${API_BASE}${att.url}`;
                          const isHtml = att.file_path.toLowerCase().endsWith(".html");
                          const isImage = /\.(png|jpg|jpeg|webp)$/i.test(att.file_path);
                          const isTable = isTableAsset(att);
                          return (
                            <div
                              key={att.asset_id}
                              className={`attachment-card ${
                                isTable ? "attachment-card-table" : ""
                              }`}
                            >
                              {isHtml ? (
                                <iframe
                                  src={url}
                                  title={att.asset_id}
                                  className="attachment-frame"
                                />
                              ) : isImage ? (
                                <img
                                  src={url}
                                  alt={att.asset_id}
                                  onClick={() =>
                                    setExpandedImage({
                                      url,
                                      label: `${att.title || att.doc_slug || "image_asset"} · page ${att.page_label}`
                                    })
                                  }
                                />
                              ) : (
                                <div className="attachment-label">
                                  Unsupported asset type
                                </div>
                              )}
                              <div className="attachment-label">
                                {att.title || att.doc_slug || "image_asset"} · page {att.page_label}
                              </div>
                              <div className="attachment-label">
                                id {att.asset_id}
                                {typeof att.score === "number" ? ` · score ${att.score.toFixed(3)}` : ""}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </details>
                  ) : null;

                return (
                  <div key={msg.id} className={`message ${msg.role}`}>
                    {isUser && primaryBlock}
                    <div className="bubble">{msg.content}</div>
                    {!isUser && primaryBlock}
                    {secondaryBlock}
                    {msg.sources && msg.sources.length > 0 && (
                      <details className="sources">
                        <summary>Sources</summary>
                        <ul>
                          {msg.sources.map((src) => (
                            <li key={`${src.chunk_id}-${src.page_label}`}>
                              {src.title || src.doc_slug} · page {src.page_label}
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                    <div className="message-meta">{formatTimestamp(msg.created_at)}</div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <section className="composer">
          {error && <div className="sources">Error: {error}</div>}
          {containsCyrillic && (
            <div className="sources">Only English queries are allowed.</div>
          )}
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="English only. Describe which schematic you need or ask a question..."
          />
          <div className="composer-suggestions">
            <div className="suggestions-title">
              Try one of these (click to fill, press Enter to send):
            </div>
            <div className="suggestions-list">
              {EXAMPLE_QUERIES.map((example) => (
                <button
                  key={example}
                  type="button"
                  className="example-chip"
                  onClick={() => {
                    setInput(example);
                    inputRef.current?.focus();
                  }}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
          <div className="composer-actions">
            <div className="composer-toggles">
              <span className="status-pill">
                {streaming ? "Streaming…" : "Ready"}
              </span>
              {rerankStatus && (
                <span
                  className="status-pill"
                  title={rerankStatus.detail || undefined}
                >
                  {rerankStatus.label}
                  {rerankStatus.detail && rerankStatus.detail.length <= 24
                    ? ` · ${rerankStatus.detail}`
                    : ""}
                </span>
              )}
              <label className="file-upload" aria-label="Attach image">
                <input
                  className="file-input"
                  type="file"
                  accept="image/*"
                  onChange={(e) => {
                    const file = e.target.files?.[0] || null;
                    setImageFile(file);
                  }}
                />
                <span className="file-icon" aria-hidden="true">📎</span>
                <span className="file-text">
                  {imageFile ? imageFile.name : "Attach image"}
                </span>
              </label>
            </div>
            <button
              className="send-btn"
              onClick={handleSend}
              disabled={streaming || containsCyrillic}
            >
              Send
            </button>
          </div>
        </section>
      </main>
      {expandedImage && (
        <div
          className="image-modal"
          onClick={() => setExpandedImage(null)}
          role="dialog"
          aria-modal="true"
        >
          <div
            className="image-modal-content"
            onClick={(e) => e.stopPropagation()}
          >
            <img src={expandedImage.url} alt={expandedImage.label} />
            <div className="image-modal-label">{expandedImage.label}</div>
            <button
              className="image-modal-close"
              onClick={() => setExpandedImage(null)}
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
