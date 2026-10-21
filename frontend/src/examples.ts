export const paperExamples = [
  {
    id: 'gcn', name: 'GCN', title: 'Semi-Supervised Classification with Graph Convolutional Networks',
    authors: 'Thomas N. Kipf、Max Welling · ICLR 2017',
    paper: 'https://openreview.net/forum?id=SJU4ayYgl', repository: 'https://github.com/tkipf/gcn',
    pdf: '/examples/gcn.pdf',
    goal: '使用作者源码，在 CPU 上复现 GCN 的 Cora 节点分类实验，按论文配置准备环境和数据，保存准确率、训练曲线与报告。',
    description: '从 Cora 节点分类开始，体验论文理解、作者源码准备、环境适配和结果比较。',
    questions: ['请解释归一化邻接矩阵在代码中的实现位置。', '核对 Cora 的数据划分、隐藏维度、dropout 和早停规则。', '只修改 Cora 的隐藏维度，先给出代码差异预览。'],
  },
  {
    id: 'sgc', name: 'SGC', title: 'Simplifying Graph Convolutional Networks',
    authors: 'Felix Wu 等 · ICML 2019',
    paper: 'https://proceedings.mlr.press/v97/wu19e.html', repository: 'https://github.com/Tiiiger/SGC',
    pdf: '/examples/sgc.pdf',
    goal: '使用作者源码，在 CPU 上复现 SGC 的 Cora 节点分类实验，核对传播阶数、学习率与权重衰减，保存准确率、训练曲线与报告。',
    description: '使用简化图卷积模型，体验参数核对、CPU 实验和代码修改。',
    questions: ['SGC 与 GCN 的计算流程有什么区别？请结合论文和源码解释。', '请找到 Cora 的传播阶数、学习率和权重衰减参数。', '保持原实验记录，预览修改传播阶数的代码方案。'],
  },
] as const

export type PaperExample = typeof paperExamples[number]

export function exampleNote(example: PaperExample): string {
  return `# ${example.name} 体验指引\n\n论文：${example.title}\n作者：${example.authors}\n论文来源：${example.paper}\n作者仓库：${example.repository}\n\n## 建议实验\n\n${example.goal}\n\n## 复现助手提问示例\n\n${example.questions.map(q => '- ' + q).join('\n')}\n\n在页面确认准备好的环境、数据和命令后运行实验。如果想要运行与 GPU 相关的实验，强烈推荐本地部署。\n`
}
