/**
 * 庄方宜语音 v2 · 设置面板（DSH 0.2.2 客户端半边）
 *
 * 按 0.2.2 官方约定：
 *   · 注册进 Plugins 页的 `plugins.item` 插槽（旧的 `settings.section` 已不是当前入口）
 *   · 需要 `slots` / `locale` / `configForms` 三个服务
 *   · 配置写入走 configForms 表单，不自己改 profile YAML
 *   · 模型扫描来自宿主 `zfhVoiceV2` 服务，经面板注入的只读快照
 */
window.__ModuleLoader__.load({
  id: 'dsh-zhuang-fangyi-voice-v2',
  factory: require => {
    const module = { exports: {} }
    const exports = module.exports
    const React = require('react')
    const { createElement: h, useState, useEffect, useCallback, useSyncExternalStore } = React

    const NS = 'settings.zhuangFangyiVoice'
    const PANEL_ID = 'zhuang-fangyi-voice-v2'

    const DICT = {
      zh: {
        title: '庄方宜语音 v2',
        description: 'GPT-SoVITS v5 · 选择要使用的 SoVITS 权重',
        v5only: '仅支持 GPT-SoVITS v5dev；v2/v2Pro 请使用 v1 插件',
        scan: '重新扫描',
        loading: '正在读取模型…',
        empty: '没有找到 v5 权重',
        emptyHint: '先运行 python download_models.py --torch-only，或把 e8/e16/e24/e32 放进模型目录',
        recommended: '推荐',
        selected: '使用中',
        use: '使用',
        size: '大小',
        error: '操作失败',
        saveFailed: '本部署没有接受该选择，已保留供你修改。'
      },
      en: {
        title: 'Zhuang Fangyi voice v2',
        description: 'GPT-SoVITS v5 · choose a SoVITS weight',
        v5only: 'Only GPT-SoVITS v5dev is supported; use the v1 plugin for v2/v2Pro',
        scan: 'Rescan',
        loading: 'Loading models…',
        empty: 'No v5 weights found',
        emptyHint: 'Run python download_models.py --torch-only first',
        recommended: 'Recommended',
        selected: 'In use',
        use: 'Use',
        size: 'Size',
        error: 'Failed',
        saveFailed: 'This deployment did not accept the choice; it was left for you to correct.'
      }
    }

    function mb (n) {
      return `${(Number(n) / 1024 / 1024).toFixed(0)} MB`
    }

    /** 面板组件：模型列表 + 选择。数据来自注入的 hooks/actions。 */
    function VoiceCard (props) {
      const t = props.t ?? (key => key)
      const models = props.models ?? []
      const current = props.current ?? null
      const busy = props.busy === true
      const error = props.error ?? null
      const onSelect = props.onSelect
      const onRescan = props.onRescan

      return h('div', { className: 'zf-voice' },
        h('p', { className: 'zf-voice-sub' }, t('description')),
        h('p', { className: 'zf-voice-note' }, t('v5only')),
        h('div', { className: 'zf-voice-actions' },
          h('button', { type: 'button', onClick: onRescan, disabled: busy }, t('scan'))
        ),
        error !== null && h('p', { className: 'zf-voice-error' }, `${t('error')}: ${error}`),
        busy && h('p', null, t('loading')),
        !busy && models.length === 0 && h('div', { className: 'zf-voice-empty' },
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
            onClick: () => onSelect(m.path)
          }, t('use'))
          ))
        )
      )
    }

    function styleText () {
      return `
.zf-voice{display:flex;flex-direction:column;gap:12px}
.zf-voice-sub,.zf-voice-note{margin:0;color:var(--dsw-alias-text-secondary,#666);font-size:13px}
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

    /** 必填服务：0.2.2 的 slots / locale / configForms。 */
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

      // 0.2.2：设置项在 Plugins 页，按 namespace 是否由宿主提供来挂载。
      ctx.effect(() => ctx.configForms.whileServed([PANEL_ID], () =>
        ctx.slots.inject('plugins.item', () => ctx.slots.register({
          name: 'plugins.item',
          id: PANEL_ID,
          order: 131,
          label: () => t('title'),
          locale: NS,
          inject: () => ({ t, vendor: 'zhuang-fangyi-voice-v2' })
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
