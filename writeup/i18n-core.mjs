export const selectLanguage = (requested, isEditor = false) => !isEditor && requested === 'zh' ? 'zh' : 'en';
export const plain = text => String(text).replace(/\*\*(.+?)\*\*/g, '$1');
export function translator(catalog, language = 'zh') {
  function translate(text) {
    if (typeof text !== 'string' || language !== 'zh') return text;
    if (catalog[text]) return catalog[text];
    const patterns = [
      [/^Seed (\d+)$/, '随机种子 $1'],
      [/^Loop (\d+) · (.*?) → (.*?)$/, (_, n, a, b) => `第 ${n} 轮 · ${translate(a)} → ${translate(b)}`],
      [/^Task · (.*)$/, (_, task) => `任务 · ${translate(task)}`],
      [/^Go to (.*)$/, (_, title) => `跳转到${translate(title)}`],
      [/^(\d+) of (\d+)$/, '$1 / $2'],
      [/^Done in ([\d.]+) s$/, '$1 秒完成'],
      [/^Not finished in (\d+) s$/, '$1 秒内未完成'],
      [/^([\d.]+) s delay$/, '延迟 $1 秒'],
      [/^· step (\d+)\/(\d+)$/, '· 第 $1 / $2 步'],
      [/^· denoising (\d+)\/(\d+)$/, '· 去噪 $1 / $2 步'],
      [/^· (\d+) of 50 executed$/, '· 已执行 $1 / 50 步'],
      [/^· succeeds in (\d+) of (\d+) random layouts$/, '· $2 种随机布局中成功 $1 次'],
      [/^· layout (\d+) of 30$/, '· 第 $1 / 30 种布局'],
      [/^average of (\d+) checkpoints?$/, '$1 个检查点的平均值'],
      [/^Updated (.*?)\. Scores, progress and videos follow training as it runs\.$/, '更新于 $1。分数、进度和视频随训练更新。'],
      [/^[Ss]tep ([\d,]+)$/, '第 $1 步'],
      [/^Read (.*)$/, (_, title) => `阅读：${translate(title)}`],
      [/^Open the (\d+) component notes$/, '展开 $1 条组件注释'],
      [/^Hide the (\d+) component notes$/, '收起 $1 条组件注释'],
      [/^training, step ([\d,]+) of ([\d,]+)$/, '训练中，第 $1 / $2 步'],
      [/^From: (.*)$/, (_, name) => `起点：${translate(name)}`],
      [/^Bar: (.*), (\d+) of (\d+)\.$/, (_, label, n, of) => `进度条：${translate(label)}，${n} / ${of}。`],
    ];
    for (const [pattern, replacement] of patterns) if (pattern.test(text)) return text.replace(pattern, replacement);
    if (text.includes(' · ')) return text.split(' · ').map(translate).join(' · ');
    return text;
  }
  return translate;
}
export function sentencePairs(en, zh) {
  const split = (text, locale) => {
    const pieces = [...new Intl.Segmenter(locale, { granularity: 'sentence' }).segment(text)].map(s => s.segment);
    const result = [];
    for (let i = 0; i < pieces.length; i++) {
      let part = pieces[i];
      while (((part.match(/\*\*/g) || []).length % 2) && i + 1 < pieces.length) {
        if (pieces[i + 1].startsWith('**')) { part += '**'; pieces[i + 1] = pieces[i + 1].slice(2); break; }
        part += pieces[++i];
      }
      if (part) result.push(part);
    }
    return result;
  };
  const a = split(en, 'en'), b = split(zh, 'zh');
  return a.length === b.length ? a.map((text, i) => ({ en: text, zh: b[i] })) : [{ en, zh }];
}

const referenceAliases = {
  'VLM loss': ['VLM 损失'], 'EAM loss': ['EAM 损失'],
  'flow loss': ['流匹配损失', '流损失', 'flow 损失'],
  'knowledge insulation': ['知识隔离'], 'attention mask': ['注意力掩码'],
  'transformer panel': ['Transformer 面板', 'transformer 面板'],
  'flow-matching panel': ['流匹配面板'], 'ingestion chapter': ['输入章节', '摄取章节'],
  'Ingestion': ['输入'], 'Decision model': ['决策模型'], 'Training losses': ['训练损失'],
  'High-level policy': ['高层策略'], 'World model': ['世界模型'],
};
export function localizeReferences(references, language) {
  return language !== 'zh' ? references : [...references, ...references.flatMap(ref =>
    (referenceAliases[ref.text] || []).map(text => ({...ref, text, locale: 'zh'})))];
}
export function firstSentence(text) {
  return new Intl.Segmenter(undefined, {granularity:'sentence'}).segment(text)[Symbol.iterator]().next().value?.segment.trim() || '';
}
