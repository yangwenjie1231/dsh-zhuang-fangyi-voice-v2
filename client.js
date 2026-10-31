/**
 * 庄方宜语音 v2 · 设置面板（浏览器半边）
 *
 * 只做一件事：列出宿主发现的 GPT-SoVITS v5 权重，让用户选择。
 * 扫描/校验/切换全在宿主与 Python 侧，这里不重复实现。
 */
window.__ModuleLoader__.load({
  id: 'dsh-zhuang-fangyi-voice-v2',
  factory: require => {
    const module = { exports: {} }
    const exports = module.exports
    const React = require('react')
    const { createElement: h, useState, useEffect, useCallback } = React

    const ROUTE = '/api/zhuang-fangyi/voice'
    const NS = 'settings.zhuangFangyiVoice'

    const DICT = {
      zh: {
        nav: '庄方宜语音 v2',
        title: '庄方宜语音 v2',
        subtitle: 'GPT-SoVITS v5 · 选择要使用的 SoVITS 权重',
        scan: '重新扫描',
        loading: '正在读取模型…',
        empty: '没有找到 v5 权重',
        emptyHint: '先运行 python download_models.py --torch-only，或把 e8/e16/e24/e32 放进模型目录',
        current: '当前使用',
        recommended: '推荐',
        selected: '使用中',
        use: '使用',
        epoch: 'epoch',
        size: '大小',
        refresh: '刷新',
        start: '启动服务',
        error: '操作失败',
        v5only: '仅支持 GPT-SoVITS v5dev；v2/v2Pro 请使用 v1 插件'
      },
      en: {
        nav: 'Zhuang Fangyi Voice v2',
        title: 'Zhuang Fangyi Voice v2',
        subtitle: 'GPT-SoVITS v5 · choose a SoVITS weight',
        scan: 'Rescan',
        loading: 'Loading models…',
        empty: 'No v5 weights found',
        emptyHint: 'Run python download_models.py --torch-only first',
        current: 'In use',
        recommended: 'Recommended',
        selected: 'Selected',
        use: 'Use',
        epoch: 'epoch',
        size: 'Size',
        refresh: 'Refresh',
        start: 'Start service',
        error: 'Failed',
        v5only: 'Only GPT-SoVITS v5dev is supported; use the v1 plugin for v2/v2Pro'
      }
    }

    function pickLocale (ctx) {
      const l = ctx?.locale?.get?.() ?? ctx?.locale?.current
      return String(l ?? 'zh').toLowerCase().startsWith('en') ? 'en' : 'zh'
    }

    function mb (n) {
      return `${(Number(n) / 1024 / 1024).toFixed(0)} MB`
    }

    async function api (path, init) {
      const res = await fetch(ROUTE + path, {
        credentials: 'same-origin',
        ...init
      })
      const body = await res.json().catch(() => ({}))
      if (!res.ok || body.ok !== true) {
        throw new Error(body.error ?? `HTTP ${res.status}`)
      }
      return body
    }

    function Section (props) {
      const t = key => (DICT[pickLocale(props?.ctx)] ?? DICT.zh)[key] ?? key
      const [state, setState] = useState({ models: [], current: null, service: null })
      const [busy, setBusy] = useState(false)
      const [error, setError] = useState(null)

      const load = useCallback(async () => {
        setBusy(true)
        setError(null)
        try {
          setState(await api('/models'))
        } catch (e) {
          setError(String(e.message ?? e))
        } finally {
          setBusy(false)
        }
      }, [])

      useEffect(() => { void load() }, [load])

      const choose = async model => {
        setBusy(true)
        setError(null)
        try {
          await api('/model', {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ model: model.path })
          })
          await load()
        } catch (e) {
          setError(String(e.message ?? e))
        } finally {
          setBusy(false)
        }
      }

      const start = async () => {
        setBusy(true)
        setError(null)
        try {
          await api('/start', { method: 'POST' })
        } catch (e) {
          setError(String(e.message ?? e))
        } finally {
          setBusy(false)
        }
      }

      const rows = state.models ?? []
      return h('div', { className: 'zf-voice' },
        h('h2', null, t('title')),
        h('p', { className: 'zf-voice-sub' }, t('subtitle')),
        h('p', { className: 'zf-voice-note' }, t('v5only')),
        h('div', { className: 'zf-voice-actions' },
          h('button', { type: 'button', onClick: load, disabled: busy }, t('scan')),
          h('button', { type: 'button', onClick: start, disabled: busy }, t('start'))
        ),
        error !== null && h('p', { className: 'zf-voice-error' }, `${t('error')}: ${error}`),
        busy && h('p', null, t('loading')),
        !busy && rows.length === 0 && h('div', { className: 'zf-voice-empty' },
          h('strong', null, t('empty')),
          h('p', null, t('emptyHint'))
        ),
        rows.length > 0 && h('ul', { className: 'zf-voice-list' }, rows.map(m =>
          h('li', {
            key: m.path,
            className: m.selected ? 'zf-voice-item is-selected' : 'zf-voice-item'
          },
          h('div', { className: 'zf-voice-main' },
            h('strong', null, `e${m.epoch}`),
            m.recommended && h('span', { className: 'zf-voice-badge' }, t('recommended')),
            m.selected && h('span', { className: 'zf-voice-badge is-current' }, t('selected'))
          ),
          h('div', { className: 'zf-voice-meta' },
            h('span', null, `${t('size')} ${mb(m.size)}`),
            h('code', null, m.filename)
          ),
          !m.selected && h('button', {
            type: 'button',
            onClick: () => choose(m),
            disabled: busy
          }, t('use'))
          ))
        )
      )
    }

    function styleText () {
      return `
.zf-voice{display:flex;flex-direction:column;gap:12px}
.zf-voice h2{margin:0;font-size:18px}
.zf-voice-sub,.zf-voice-note{margin:0;color:var(--dsh-text-secondary,#666);font-size:13px}
.zf-voice-actions{display:flex;gap:8px}
.zf-voice-list{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:8px}
.zf-voice-item{border:1px solid var(--dsh-border,#ddd);border-radius:10px;padding:10px 12px;display:grid;grid-template-columns:1fr auto;gap:6px 12px;align-items:center}
.zf-voice-item.is-selected{border-color:var(--dsh-accent,#3b82f6);background:color-mix(in srgb,var(--dsh-accent,#3b82f6) 8%,transparent)}
.zf-voice-main{display:flex;align-items:center;gap:8px}
.zf-voice-meta{grid-column:1;display:flex;gap:12px;align-items:center;color:var(--dsh-text-secondary,#666);font-size:12px;overflow:hidden}
.zf-voice-meta code{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.zf-voice-item>button{grid-row:1/3;grid-column:2}
.zf-voice-badge{font-size:11px;padding:1px 6px;border-radius:999px;background:var(--dsh-fill,#eee)}
.zf-voice-badge.is-current{background:var(--dsh-accent,#3b82f6);color:#fff}
.zf-voice-empty{padding:16px;border:1px dashed var(--dsh-border,#ddd);border-radius:10px}
.zf-voice-error{color:#b91c1c;margin:0}
`
    }

    function apply (ctx) {
      ctx.effect(() => {
        const el = document.createElement('style')
        el.id = 'zf-voice-style'
        el.textContent = styleText()
        document.head.appendChild(el)
        return () => { try { el.remove() } catch { /* ignore */ } }
      }, 'zhuang-fangyi-voice-v2: styles')

      try {
        ctx.slots.inject('settings.section', () => ctx.slots.register(
          { name: 'settings.section', id: 'zhuang-fangyi-voice-v2', order: 131, label: () => '庄方宜语音 v2', locale: NS },
          Section
        ))
      } catch (error) {
        console.warn('[zhuang-fangyi-voice-v2] settings.section 注册失败:', error?.message ?? error)
      }
    }

    exports.__test = { Section, styleText, DICT, ROUTE }
    exports.apply = apply
    exports.inject = ['slots', 'locale']
    exports.name = 'dsh-zhuang-fangyi-voice-v2'
    return module.exports
  }
})
