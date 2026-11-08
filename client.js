/**
 * 庄方宜语音 v2 · 设置面板（DSH 0.2.2 客户端半边）
 *
 * 分工（0.2.2 的约定）：
 *   · **持久化**交给宿主 —— Config schema 里 `sovitsPath` 标了 `.volatile()`，
 *     宿主会为插件行自动渲染配置表单并写回 profile。
 *   · **本面板**只做两件 Config 表单做不到的事：扫描模型目录、热气切换权重。
 *     数据直接来自 Python 服务（它已带 CORS），不经过宿主转发。
 *
 * 注册位置：`plugins.item`（0.2.2 的设置入口，旧 `settings.section` 已不是）。
 */
window.__ModuleLoader__.load({
  id: 'dsh-zhuang-fangyi-voice-v2',
  factory: require => {
    const module = { exports: {} }
    const exports = module.exports
    const React = require('react')
    const { createElement: h, useState, useEffect, useCallback } = React

    const NS = 'settings.zhuangFangyiVoice'
    const PANEL_ID = 'zhuang-fangyi-voice-v2'
    const DEFAULT_BASE = 'http://127.0.0.1:8766'

    const DICT = {
      zh: {
        title: '庄方宜语音 v2',
        description: 'GPT-SoVITS v5 · 选择要使用的 SoVITS 权重',
        v5only: '仅支持 GPT-SoVITS v5dev；v2/v2Pro 请使用 v1 插件',
        hint: '选中后立即生效；要永久保存，请在上方配置表单里填写 sovitsPath。',
        scan: '重新扫描',
        loading: '正在读取模型…',
        offline: '推理服务未启动',
        offlineHint: '先在插件配置里启动服务，或调用一次合成后再打开此页。',
        empty: '没有找到 v5 权重',
        emptyHint: '先运行 python download_models.py --torch-only',
        recommended: '推荐',
        selected: '使用中',
        use: '使用',
        size: '大小',
        error: '操作失败'
      },
      en: {
        title: 'Zhuang Fangyi voice v2',
        description: 'GPT-SoVITS v5 · choose a SoVITS weight',
        v5only: 'Only GPT-SoVITS v5dev is supported; use the v1 plugin for v2/v2Pro',
        hint: 'Switching applies immediately; set sovitsPath in the config form above to persist it.',
        scan: 'Rescan',
        loading: 'Loading models…',
        offline: 'Inference service is not running',
        offlineHint: 'Start it from the plugin config, or synthesize once, then reopen this page.',
        empty: 'No v5 weights found',
        emptyHint: 'Run python download_models.py --torch-only first',
        recommended: 'Recommended',
        selected: 'In use',
        use: 'Use',
        size: 'Size',
        error: 'Failed'
      }
    }

    function mb (n) {
      return `${(Number(n) / 1024 / 1024).toFixed(0)} MB`
    }

    async function call (path, init) {
      const base = (typeof window !== 'undefined' && window.__ZFH_VOICE_BASE__) || DEFAULT_BASE
      const res = await fetch(base + path, {
        ...init,
        headers: { 'content-type': 'application/json', ...(init?.headers ?? {}) }
      })
      const body = await res.json().catch(() => ({}))
      if (!res.ok || body.ok !== true) throw new Error(body.error ?? `HTTP ${res.status}`)
      return body
    }

    /**
     * 面板组件。用地址栏同源可见的固定端口读列表，失败即视为服务未启动，
     * 不伪造空列表。
     */
    function VoiceCard (props) {
      const t = props.t ?? (key => key)
      const [models, setModels] = useState([])
      const [current, setCurrent] = useState(null)
      const [busy, setBusy] = useState(true)
      const [offline, setOffline] = useState(false)
      const [error, setError] = useState(null)

      const load = useCallback(async () => {
        setBusy(true)
        setError(null)
        try {
          const body = await call('/v2/models', { method: 'GET' })
          setModels(Array.isArray(body.models) ? body.models : [])
          setCurrent(body.current ?? null)
          setOffline(false)
        } catch (e) {
          setOffline(true)
          setModels([])
          setCurrent(null)
        } finally {
          setBusy(false)
        }
      }, [])

      useEffect(() => { void load() }, [load])

      const choose = async path => {
        setBusy(true)
        setError(null)
        try {
          await call('/v2/model', {
            method: 'POST',
            body: JSON.stringify({ model: path })
          })
          await load()
        } catch (e) {
          setError(String(e.message ?? e))
        } finally {
          setBusy(false)
        }
      }

      return h('div', { className: 'zf-voice' },
        h('p', { className: 'zf-voice-sub' }, t('description')),
        h('p', { className: 'zf-voice-note' }, t('v5only')),
        h('p', { className: 'zf-voice-hint' }, t('hint')),
        h('div', { className: 'zf-voice-actions' },
          h('button', { type: 'button', onClick: load, disabled: busy }, t('scan'))
        ),
        error !== null && h('p', { className: 'zf-voice-error' }, `${t('error')}: ${error}`),
        busy && models.length === 0 && !offline && h('p', null, t('loading')),
        offline && h('div', { className: 'zf-voice-empty' },
          h('strong', null, t('offline')),
          h('p', null, t('offlineHint'))
        ),
        !offline && !busy && models.length === 0 && h('div', { className: 'zf-voice-empty' },
          h('strong', null, t('empty')),
          h('p', null, t('emptyHint'))
        ),
        models.length > 0 && h('ul', { className: 'zf-voice-list' }, models.map(m =>
          h('li', {
            key: m.path,
            className: m.path === current ? 'zf-voice-item is-selected' : 'zf-voice-item'
          },
          h('div', { className: 'zf-voice-main' },
            h('strong', null, `e${m.epoch}`),
            m.recommended && h('span', { className: 'zf-voice-badge' }, t('recommended')),
            m.path === current && h('span', { className: 'zf-voice-badge is-current' }, t('selected'))
          ),
          h('div', { className: 'zf-voice-meta' },
            h('span', null, `${t('size')} ${mb(m.size)}`),
            h('code', null, m.filename)
          ),
          m.path !== current && h('button', {
            type: 'button',
            disabled: busy,
            onClick: () => choose(m.path)
          }, t('use'))
          ))
        )
      )
    }

    function styleText () {
      return `
.zf-voice{display:flex;flex-direction:column;gap:10px}
.zf-voice-sub,.zf-voice-note,.zf-voice-hint{margin:0;color:var(--dsw-alias-text-secondary,#666);font-size:13px}
.zf-voice-actions{display:flex;gap:8px}
.zf-voice-list{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:8px}
.zf-voice-item{border:1px solid var(--dsw-alias-border,#ddd);border-radius:10px;padding:10px 12px;display:grid;grid-template-columns:1fr auto;gap:6px 12px;align-items:center}
.zf-voice-item.is-selected{border-color:var(--dsw-alias-accent,#3b82f6)}
.zf-voice-main{display:flex;align-items:center;gap:8px}
.zf-voice-meta{grid-column:1;display:flex;gap:12px;align-items:center;color:var(--dsw-alias-text-secondary,#666);font-size:12px;overflow:hidden}
.zf-voice-meta code{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.zf-voice-item>button{grid-row:1/3;grid-column:2}
.zf-voice-badge{font-size:11px;padding:1px 6px;border-radius:999px;background:var(--dsw-alias-fill,#eee)}
.zf-voice-badge.is-current{background:var(--dsw-alias-accent,#3b82f6);color:#fff}
.zf-voice-empty{padding:16px;border:1px dashed var(--dsw-alias-border,#ddd);border-radius:10px}
.zf-voice-error{color:#b91c1c;margin:0}
`
    }

    const inject = ['slots', 'locale', 'configForms']

    function apply (ctx) {
      const t = ctx.locale.bind(NS)
      ctx.effect(() => ctx.locale.register(NS, DICT), 'zhuang-fangyi-voice-v2: dictionaries')
      ctx.effect(() => {
        const el = document.createElement('style')
        el.id = 'zf-voice-style'
        el.textContent = styleText()
        document.head.appendChild(el)
        return () => { try { el.remove() } catch { /* ignore */ } }
      }, 'zhuang-fangyi-voice-v2: styles')

      ctx.effect(() => ctx.configForms.whileServed([PANEL_ID], () =>
        ctx.slots.inject('plugins.item', () => ctx.slots.register({
          name: 'plugins.item',
          id: PANEL_ID,
          order: 131,
          label: () => t('title'),
          locale: NS,
          inject: () => ({ t })
        }, VoiceCard))
      ), 'zhuang-fangyi-voice-v2: settings page')
    }

    exports.__test = { VoiceCard, styleText, DICT, NS, PANEL_ID, inject }
    exports.apply = apply
    exports.inject = inject
    exports.name = 'dsh-zhuang-fangyi-voice-v2'
    return module.exports
  }
})