import { readFileSync } from 'node:fs'
import { defineConfig } from 'vitepress'
import { withSidebar } from 'vitepress-sidebar'

const vitePressConfig = defineConfig({
  base: '/blog/',
  title: '学习记录',
  description: 'AI 编译器、算子优化与推理系统知识库',
  lang: 'zh-CN',
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '知识库', link: '/notes/' },
      { text: '编程基础', link: '/notes/cpp/' },
      { text: '编译器', link: '/notes/compile/' },
      { text: '算子优化', link: '/notes/kernels/' },
      { text: '推理系统', link: '/notes/inference/' },
      { text: '实践', items: [
        { text: '实践记录', link: '/posts/' },
        { text: '项目实战', link: '/projects/' },
      ] },
    ],
    search: { provider: 'local' },
    outline: { label: '目录', level: [2, 3] },
    docFooter: { prev: '上一篇', next: '下一篇' },
    socialLinks: [{ icon: 'github', link: 'https://github.com/jnfkdsn' }],
  },
})

// Domain landing pages show modules; a course shows its own chapter tree.
// A folder title already links to its index, so do not repeat that index as a child.
const overviewScopes = new Set(['notes', 'notes/compile', 'notes/kernels'])
const scopes = [
  'notes', 'notes/cpp', 'notes/compile',
  'notes/compile/traditional', 'notes/compile/ai-basics', 'notes/compile/mlir',
  'notes/compile/tvm', 'notes/compile/pytorch-compiler',
  'notes/compile/triton', 'notes/compile/npu-ir',
  'notes/kernels', 'notes/kernels/cuda', 'notes/kernels/ascend-c',
  'notes/kernels/algorithms', 'notes/inference', 'notes/inference/quantization',
  'posts', 'projects',
]

export default withSidebar(vitePressConfig, scopes.map(scope => ({
  documentRootPath: '/docs',
  scanStartPath: scope,
  resolvePath: `/${scope}/`,
  basePath: `/${scope}/`,
  collapsed: true,
  collapseDepth: 1,
  excludeFilesByFrontmatterFieldName: 'excludeFromSidebar',
  ...(overviewScopes.has(scope) ? { excludeByFolderDepth: 2 } : {}),
  useTitleFromFileHeading: true,
  useFolderTitleFromIndexFile: true,
  useFolderLinkFromIndexFile: true,
  includeRootIndexFile: false,
  rootGroupText: readFileSync(`docs/${scope}/index.md`, 'utf8').match(/^# (.+)$/m)?.[1] ?? scope,
  rootGroupLink: 'index.md',
  rootGroupCollapsed: false,
  includeFolderIndexFile: false,
  sortMenusByFrontmatterOrder: true,
  frontmatterOrderDefaultValue: 999,
})))
