import React, { useEffect, useRef, useState } from 'react'
import './index.css'

const PERSONAS = [
  { id: 'teacher', name: 'Teacher' },
  { id: 'senior', name: 'Senior' },
  { id: 'parent', name: 'Parent' },
  { id: 'counselor', name: 'Counselor' },
  { id: 'friend', name: 'Friend' },
]

const EXAMPLES = [
  'Explain how indexing works in databases with a simple example',
  'How do JWT refresh tokens work in real-world systems?',
  'What is machine learning in simple words?',
  'What are the best free resources to learn cloud computing for job readiness?',
  'How should I prepare for my first software engineering internship?',
  'I feel scared of failing in life',
  'I feel lonely even when I am surrounded by people',
  'I overthink every decision and feel mentally tired',
  'Talk to me like a friend, I just need some company',
  'What food gives energy for studying?',
  'How can I fix my sleep schedule before exams?',
]

const TITLES = [
  'Expert Collective.',
  'Five Minds. One Answer.',
  'The Intelligence Deck.',
  'Unified Perspectives.',
  'The Collective Brain.',
]

const randomTitle = TITLES[Math.floor(Math.random() * TITLES.length)]

const renderText = (text) => {
  if (!text) return ''
  const parts = text.split(/(\*\*.*?\*\*)/g)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i}>{part.slice(2, -2)}</strong>
    }
    return part
  })
}

export default function App() {
  const STORAGE_KEY = 'multi_agent_ui_state_v2'
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [logs, setLogs] = useState([])
  const [showLogs, setShowLogs] = useState(false)
  const [personaOutputs, setPersonaOutputs] = useState({})
  const [response, setResponse] = useState(null)
  const [focusPersona, setFocusPersona] = useState(null)
  const [personaChats, setPersonaChats] = useState({})
  const [livePriorityPersonas, setLivePriorityPersonas] = useState([])
  const [completedPersonas, setCompletedPersonas] = useState({})
  const [runMetrics, setRunMetrics] = useState({})
  const [carouselIndex, setCarouselIndex] = useState(0)
  const esRef = useRef(null)
  const pendingFocusedQueryRef = useRef(null)

  useEffect(() => {
    return () => {
      if (esRef.current) esRef.current.close()
    }
  }, [])

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (!raw) return
      const saved = JSON.parse(raw)
      if (typeof saved.query === 'string') setQuery(saved.query)
      if (typeof saved.focusPersona === 'string' || saved.focusPersona === null) setFocusPersona(saved.focusPersona)
      if (saved.response && typeof saved.response === 'object') setResponse(saved.response)
      if (saved.personaOutputs && typeof saved.personaOutputs === 'object') setPersonaOutputs(saved.personaOutputs)
      if (saved.personaChats && typeof saved.personaChats === 'object') setPersonaChats(saved.personaChats)
      if (saved.runMetrics && typeof saved.runMetrics === 'object') setRunMetrics(saved.runMetrics)
    } catch (_) {}
  }, [])

  useEffect(() => {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ query, focusPersona, response, personaOutputs, personaChats, runMetrics })
      )
    } catch (_) {}
  }, [query, focusPersona, response, personaOutputs, personaChats, runMetrics])

  function startQuery(q, focus = null) {
    if (!q.trim()) return
    setQuery(q)
    if (esRef.current) esRef.current.close()

    setLoading(true)
    setResponse(null)
    setRunMetrics({})
    setLivePriorityPersonas([])

    if (!focus) {
      setPersonaOutputs({})
      setCompletedPersonas({})
      setCarouselIndex(0)
      setFocusPersona(null)
    } else {
      setCompletedPersonas((prev) => ({ ...prev, [focus]: false }))
      setPersonaOutputs((prev) => ({ ...prev, [focus]: '' }))
    }

    setLogs([])
    if (focus) {
      setFocusPersona(focus)
      pendingFocusedQueryRef.current = q
    }

    const effectiveFocus = focus || focusPersona
    const focusParam = effectiveFocus ? `&focus=${effectiveFocus}` : ''
    const es = new EventSource(`http://localhost:8000/chat/stream?query=${encodeURIComponent(q)}${focusParam}`)
    esRef.current = es

    es.onmessage = (e) => {
      const data = JSON.parse(e.data)

      if (data.type === 'log') setLogs((prev) => [...prev, data.message])
      if (data.type === 'routing') setLivePriorityPersonas(data.priority_personas || [])
      if (data.type === 'metric') setRunMetrics((prev) => ({ ...prev, [data.node]: data.payload }))

      if (data.type === 'token') {
        setPersonaOutputs((prev) => ({
          ...prev,
          [data.id]: (prev[data.id] || '') + data.text,
        }))
      }

      if (data.type === 'persona') {
        setPersonaOutputs((prev) => ({ ...prev, [data.id]: data.text }))
        setCompletedPersonas((prev) => ({ ...prev, [data.id]: true }))
      }

      if (data.type === 'final') {
        const activeFocus = effectiveFocus
        if (activeFocus && pendingFocusedQueryRef.current) {
          const userText = pendingFocusedQueryRef.current
          setPersonaChats((prev) => ({
            ...prev,
            [activeFocus]: [
              ...(prev[activeFocus] || []),
              { role: 'user', text: userText },
              { role: 'assistant', text: data.answer || '' },
            ],
          }))
          pendingFocusedQueryRef.current = null
        }
        setResponse(data)
        if (data.node_metrics) setRunMetrics(data.node_metrics)
        setLoading(false)
        es.close()
      }

      if (data.type === 'error') {
        setLogs((prev) => [...prev, `[ERR] ${data.message}`])
        setLoading(false)
        es.close()
      }
    }

    es.onerror = () => {
      setLoading(false)
      es.close()
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (loading) return
    if (focusPersona) startQuery(query, focusPersona)
    else startQuery(query)
  }

  const carouselItems = [
    { id: 'combined', name: 'Synthesized Answer', text: response?.answer || 'Synthesizing perspectives...' },
    ...PERSONAS.map((p) => ({ id: p.id, name: `${p.name}'s Mind`, text: personaOutputs[p.id] || 'Agent is thinking...' })),
  ]

  return (
    <div className="page">
      <div className="hero">
        <div className="hero-label">Multi-Agent Intelligence {focusPersona ? `| Focusing on ${focusPersona}` : ''}</div>
        <h1>{randomTitle}</h1>
        <p className="hero-sub">Five expert agents collaborating on your query. Slide through the deck to explore their individual minds.</p>
      </div>




      <div className={`examples ${loading ? 'disabled' : ''}`}>
        {EXAMPLES.map((ex, i) => (
          <div key={i} className="ex-pill" onClick={() => !loading && startQuery(ex)} style={{ cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.5 : 1 }}>
            {ex}
          </div>
        ))}
      </div>

      <form className="search-wrap" onSubmit={handleSubmit}>
        <input className="search-input" type="text" placeholder={focusPersona ? `Message ${focusPersona}...` : "What's on your mind today?"} value={query} onChange={(e) => setQuery(e.target.value)} disabled={loading} />
        <button className="search-btn" type="submit" disabled={loading}>{loading ? 'Running Experts...' : (focusPersona ? `Chat with ${focusPersona}` : 'Ask All 5')}</button>
        {focusPersona && (
          <button type="button" className="search-btn" onClick={() => { setFocusPersona(null); setQuery('') }} disabled={loading}>
            Exit Chat Mode
          </button>
        )}
      </form>

      {query && (
        <div className="query-banner">
          <div className="query-label">Current Query</div>
          <div className="query-text">"{query}"</div>
        </div>
      )}





      {livePriorityPersonas.length > 0 && (
        <div className="priority-banner">
          <span className="priority-banner-label">Priority Now</span>
          <span className="priority-names">{livePriorityPersonas.join(' + ')}</span>
        </div>
      )}

      {(loading || response || Object.keys(personaOutputs).length > 0 || Object.keys(personaChats).length > 0) && (
        <div className="carousel-deck-container">
          <button onClick={() => setCarouselIndex((i) => (i - 1 + carouselItems.length) % carouselItems.length)} className="pane-nav-arrow left">{'<'}</button>
          <button onClick={() => setCarouselIndex((i) => (i + 1) % carouselItems.length)} className="pane-nav-arrow right">{'>'}</button>

          <div className="deck-wrapper">
            {carouselItems.map((item, idx) => {
              let position = 'hidden'
              if (idx === carouselIndex) position = 'active'
              else if (idx === (carouselIndex - 1 + carouselItems.length) % carouselItems.length) position = 'prev'
              else if (idx === (carouselIndex + 1) % carouselItems.length) position = 'next'
              if (position === 'hidden') return null

              const prioritySet = response?.priority_personas || livePriorityPersonas
              const isPriority = prioritySet?.includes(item.id)
              const isCombined = item.id === 'combined'
              const isDone = isCombined ? !!response?.answer : !!completedPersonas[item.id]
              const isCardLoading = loading && !isDone

              const colorMap = {
                teacher: 'brand-pink',
                senior: 'brand-teal',
                parent: 'brand-lavender',
                counselor: 'brand-peach',
                friend: 'brand-ochre',
                combined: 'surface-card'
              }
              const cardColorClass = colorMap[item.id] || ''

              return (
                <div key={item.id} className={`stacked-card ${position} ${cardColorClass} ${isCardLoading ? 'loading' : ''} ${isDone ? 'done' : ''} ${isPriority ? 'priority-aura' : ''} ${isCombined ? 'is-combined' : ''}`} onClick={() => !loading && position !== 'active' && setCarouselIndex(idx)}>
                  <div className="card-header">
                    <div className="card-title-group">
                      <span className="expert-tag" style={{ backgroundColor: `var(--${cardColorClass})` }}>{item.id.toUpperCase()}</span>
                      <h2 className="card-persona-title">{item.name}</h2>
                    </div>
                    <div className="card-status-group">
                      {isPriority && <span className="priority-label">LEAD EXPERT</span>}
                      {focusPersona === item.id && <span className="focus-tag">FOCUS</span>}
                    </div>
                  </div>

                  <div className="card-body">
                    {focusPersona === item.id ? (
                      <div className="persona-chat-window">
                        {(personaChats[item.id] || []).map((m, i) => (
                          <div key={i} className={`chat-bubble ${m.role}`}>{renderText(m.text)}</div>
                        ))}
                        {loading && <div className="chat-bubble assistant">Thinking...</div>}
                        <form className="in-card-chat-form" onSubmit={(e) => { e.preventDefault(); if (loading || !query.trim()) return; startQuery(query, item.id) }}>
                          <input className="in-card-chat-input" type="text" placeholder={`Message ${item.id}...`} value={query} onChange={(e) => setQuery(e.target.value)} disabled={loading} />
                          <button className="focus-btn" type="submit" disabled={loading || !query.trim()}>Send</button>
                        </form>
                      </div>
                    ) : (
                      renderText(item.text)
                    )}
                  </div>

                  {position === 'active' && !loading && (
                    <div className="card-actions">
                      <button className="focus-btn" onClick={() => {
                        setFocusPersona(item.id)
                        setCarouselIndex(idx)
                        setQuery('')
                        setPersonaChats((prev) => {
                          const existing = prev[item.id] || []
                          if (existing.length > 0) return prev
                          const seedText = (item.text && item.text !== 'Agent is thinking...' && item.text !== 'Synthesizing perspectives...')
                            ? item.text
                            : `You are now chatting with ${item.id}. Ask anything and I will stay in this mode.`
                          return { ...prev, [item.id]: [{ role: 'assistant', text: seedText }] }
                        })
                      }}>
                        Continue Conversation
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginLeft: '8px' }}>
                          <path d="M5 12h14"></path>
                          <path d="M12 5l7 7-7 7"></path>
                        </svg>
                      </button>
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          <div className="carousel-dots">
            {carouselItems.map((_, i) => (
              <div key={i} className={`dot ${i === carouselIndex ? 'active' : ''}`} onClick={() => setCarouselIndex(i)} />
            ))}
          </div>

          <div className="bottom-meta">
            <div className="meta-left">
              <button className="text-link" onClick={() => setShowLogs(!showLogs)}>{showLogs ? 'Hide Logs' : 'View Logs'}</button>
              <button className="text-link danger" style={{ marginLeft: '16px', color: '#ef4444' }} onClick={() => {
                localStorage.removeItem(STORAGE_KEY)
                window.location.reload()
              }}>Reset Session</button>
            </div>
            {response && <div className="stats">{Math.round(response.confidence * 100)}% Match | {response.latency?.toFixed(1)}s | Mode: {response.generation_mode || 'n/a'}</div>}
          </div>

          {Object.keys(runMetrics).length > 0 && (
            <div className="metrics-grid">
              {Object.entries(runMetrics).map(([k, v]) => (
                <div key={k} className="metric-card">
                  <div className="metric-title">{k}</div>
                  <div className="metric-val">{typeof v?.latency === 'number' ? `${v.latency.toFixed(2)}s` : 'n/a'}</div>
                </div>
              ))}
            </div>
          )}

          {showLogs && (
            <div className="log-drawer">
              {logs.map((l, i) => <div key={i} className="log-entry">{l}</div>)}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
