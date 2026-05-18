import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type Session = { id: number; title: string; created_at: string }
type Message = { role: string; content: string; created_at?: string }
type Citation = { source: string; page?: number | null; distance: number; snippet_preview: string }
type Step = { kind: string; detail?: string; tool_name?: string; result_preview?: string; [key: string]: unknown }
type Doc = { filename: string; size_bytes: number; modified_at: number }
type RagSettings = { embedding_model: string; top_k: number; chunk_size: number; chunk_overlap: number }

const API = '/api'

function App() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [sessionId, setSessionId] = useState(1)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [systemPrompt, setSystemPrompt] = useState('')
  const [agent, setAgent] = useState(false)
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('准备就绪')
  const [citations, setCitations] = useState<Citation[]>([])
  const [steps, setSteps] = useState<Step[]>([])
  const [docs, setDocs] = useState<Doc[]>([])
  const [settings, setSettings] = useState<RagSettings | null>(null)
  const fileRef = useRef<HTMLInputElement | null>(null)
  const bottomRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => { void bootstrap() }, [])
  useEffect(() => { void loadHistory(sessionId) }, [sessionId])
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  async function json<T>(url: string, init?: RequestInit): Promise<T> {
    const res = await fetch(url, init)
    if (!res.ok) throw new Error(await res.text())
    return res.json() as Promise<T>
  }

  async function bootstrap() {
    try {
      const [s, d, r] = await Promise.all([
        json<Session[]>(`${API}/sessions`),
        json<Doc[]>(`${API}/documents`),
        json<RagSettings>(`${API}/rag/settings`),
      ])
      setSessions(s); setDocs(d); setSettings(r)
      if (s[0]) setSessionId(s[0].id)
    } catch (e) { setStatus(e instanceof Error ? e.message : '初始化失败') }
  }

  async function loadHistory(id: number) {
    try {
      setMessages(await json<Message[]>(`${API}/history?session_id=${id}`))
      setCitations([]); setSteps([])
    } catch (e) { setStatus(e instanceof Error ? e.message : '加载历史失败') }
  }

  async function createSession() {
    const s = await json<Session>(`${API}/sessions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: `新会话 ${sessions.length + 1}` }),
    })
    setSessions((v) => [...v, s]); setSessionId(s.id); setStatus('新会话已创建')
  }

  async function deleteSession() {
    if (sessions.length <= 1) return setStatus('至少保留一个会话')
    await fetch(`${API}/sessions/${sessionId}`, { method: 'DELETE' })
    const rest = sessions.filter((s) => s.id !== sessionId)
    setSessions(rest); if (rest[0]) setSessionId(rest[0].id)
  }

  async function clearHistory() {
    await fetch(`${API}/history?session_id=${sessionId}`, { method: 'DELETE' })
    setMessages([]); setCitations([]); setSteps([]); setStatus('历史已清空')
  }

  async function submit(e: FormEvent) {
    e.preventDefault()
    const text = input.trim(); if (!text || busy) return
    setInput(''); setBusy(true); setStatus(agent ? 'Agent 思考中...' : '模型回复中...')
    setCitations([]); setSteps([])
    setMessages((v) => [...v, { role: 'user', content: text }, { role: 'assistant', content: '' }])
    try {
      if (agent) await sendAgent(text); else await sendStream(text)
      setStatus('回复完成')
    } catch (err) {
      const msg = err instanceof Error ? err.message : '发送失败'
      setStatus(msg); replaceLast(`出错了：${msg}`)
    } finally { setBusy(false) }
  }

  async function sendAgent(text: string) {
    const data = await json<{ reply: string; steps: Step[]; rag_citations: Citation[] }>(`${API}/chat/agent`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, system_prompt: systemPrompt, session_id: sessionId }),
    })
    setCitations(data.rag_citations || []); setSteps(data.steps || []); replaceLast(data.reply)
  }

  async function sendStream(text: string) {
    const res = await fetch(`${API}/chat/stream`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, system_prompt: systemPrompt, session_id: sessionId }),
    })
    if (!res.ok || !res.body) throw new Error(await res.text())
    const reader = res.body.getReader(); const decoder = new TextDecoder()
    let buffer = ''; let meta = false
    while (true) {
      const { value, done } = await reader.read(); if (done) break
      buffer += decoder.decode(value, { stream: true })
      if (!meta) {
        const idx = buffer.indexOf('\n'); if (idx < 0) continue
        const line = buffer.slice(0, idx); buffer = buffer.slice(idx + 1); meta = true
        if (line.startsWith('::META::')) setCitations(JSON.parse(line.slice(8)).rag_citations || [])
        else append(line)
      }
      if (buffer) { append(buffer); buffer = '' }
    }
  }

  function append(text: string) { setMessages((v) => v.map((m, i) => i === v.length - 1 ? { ...m, content: m.content + text } : m)) }
  function replaceLast(text: string) { setMessages((v) => v.map((m, i) => i === v.length - 1 ? { ...m, content: text } : m)) }

  async function upload() {
    const file = fileRef.current?.files?.[0]; if (!file) return
    const data = new FormData(); data.append('file', file); setStatus('文档处理中...')
    const res = await fetch(`${API}/upload`, { method: 'POST', body: data })
    if (!res.ok) throw new Error(await res.text())
    const body = await res.json() as { message: string }
    setStatus(body.message); setDocs(await json<Doc[]>(`${API}/documents`))
    if (fileRef.current) fileRef.current.value = ''
  }

  async function removeDoc(name: string) {
    setStatus('删除并重建向量库...')
    const res = await fetch(`${API}/documents/${encodeURIComponent(name)}`, { method: 'DELETE' })
    if (!res.ok) throw new Error(await res.text())
    setDocs(await json<Doc[]>(`${API}/documents`)); setStatus('文档已删除')
  }

  async function rebuild() {
    const r = await json<{ message: string }>(`${API}/documents/rebuild`, { method: 'POST' })
    setStatus(r.message)
  }

  return <main className="app-shell">
    <aside className="sidebar">
      <div className="brand"><b>AI</b><div><strong>Python AI Starter</strong><small>v3.2+</small></div></div>
      <button className="primary full" onClick={createSession}>新建会话</button>
      <div className="session-list">{sessions.map((s) => <button key={s.id} className={s.id === sessionId ? 'session active' : 'session'} onClick={() => setSessionId(s.id)}><span>{s.title}</span><small>#{s.id}</small></button>)}</div>
      <div className="sidebar-actions"><button onClick={clearHistory}>清空历史</button><button onClick={deleteSession}>删除会话</button></div>
    </aside>
    <section className="chat-panel">
      <header className="topbar"><div><h1>{sessions.find((s) => s.id === sessionId)?.title || '默认会话'}</h1><p>{status}</p></div><label><input checked={agent} onChange={(e) => setAgent(e.target.checked)} type="checkbox" /> Agent 模式</label></header>
      <div className="messages">{messages.length ? messages.map((m, i) => <article className={`message ${m.role}`} key={i}><span>{m.role === 'user' ? '你' : '助手'}</span><p>{m.content || '...'}</p></article>) : <div className="empty-state"><h2>开始一次 AI 对话</h2><p>上传资料后，可用流式问答或 Agent 工具问答。</p></div>}<div ref={bottomRef} /></div>
      {citations.length > 0 && <section className="card"><h2>RAG 引用</h2>{citations.map((c, i) => <div className="citation" key={i}><strong>{c.source}</strong><small>{typeof c.page === 'number' ? `第 ${c.page + 1} 页` : '页码未知'} · 距离 {c.distance.toFixed(4)}</small><p>{c.snippet_preview}</p></div>)}</section>}
      {steps.length > 0 && <section className="card"><h2>Agent 步骤</h2>{steps.map((s, i) => <details key={i}><summary>{s.tool_name || s.detail || s.kind}</summary><pre>{JSON.stringify(s, null, 2)}</pre></details>)}</section>}
      <form className="composer" onSubmit={submit}><textarea placeholder="系统提示词" value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} /><div className="input-row"><input placeholder="输入你的问题..." value={input} onChange={(e) => setInput(e.target.value)} /><button className="primary" disabled={busy}>{busy ? '发送中' : '发送'}</button></div></form>
    </section>
    <aside className="right-panel"><section className="card"><h2>知识库</h2><input accept=".pdf,.txt" ref={fileRef} type="file" /><div className="button-row"><button onClick={() => void upload()}>上传入库</button><button onClick={() => void rebuild()}>重建</button></div><div className="document-list">{docs.length ? docs.map((d) => <div className="document" key={d.filename}><span>{d.filename}</span><button onClick={() => void removeDoc(d.filename)}>删除</button></div>) : <p>暂无文档</p>}</div></section><section className="card"><h2>RAG 配置</h2>{settings ? <dl><dt>Embedding</dt><dd>{settings.embedding_model}</dd><dt>Top K</dt><dd>{settings.top_k}</dd><dt>Chunk</dt><dd>{settings.chunk_size} / {settings.chunk_overlap}</dd></dl> : <p>加载中...</p>}</section></aside>
  </main>
}

export default App
