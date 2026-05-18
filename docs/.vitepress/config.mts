import { defineConfig } from 'vitepress'
import { withSidebar } from 'vitepress-sidebar'

const vitePressConfig = defineConfig({
  base: '/blog/',
  title: '学习记录',
  description: 'CUDA / AI Infra 学习笔记与实践',
  lang: 'zh-CN',

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '知识库', link: '/notes/' },
      { text: 'CUDA', link: '/notes/cuda/' },
      { text: 'Triton', link: '/notes/triton/' },
      { text: '推理系统', link: '/notes/infer/' },
      { text: '实践记录', link: '/posts/' },
      { text: '项目实战', link: '/projects/' },
    ],

    search: {
      provider: 'local',
    },

    outline: {
      label: '目录',
      level: [2, 3],
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇',
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/jnfkdsn' }
    ],
  }
})

export default withSidebar(vitePressConfig, [
  {
    documentRootPath: '/docs',
    scanStartPath: 'notes',
    resolvePath: '/notes/',
    useTitleFromFileHeading: true,
    useFolderTitleFromIndexFile: true,
    useFolderLinkFromIndexFile: true,
    includeFolderIndexFile: true,
    sortMenusByFrontmatterOrder: true,
    frontmatterOrderDefaultValue: 999,
  },
  {
    documentRootPath: '/docs',
    scanStartPath: 'posts',
    resolvePath: '/posts/',
    useTitleFromFileHeading: true,
    useFolderTitleFromIndexFile: true,
    useFolderLinkFromIndexFile: true,
    includeFolderIndexFile: true,
    sortMenusByFrontmatterOrder: true,
    frontmatterOrderDefaultValue: 999,
  },
  {
    documentRootPath: '/docs',
    scanStartPath: 'projects',
    resolvePath: '/projects/',
    useTitleFromFileHeading: true,
    useFolderTitleFromIndexFile: true,
    useFolderLinkFromIndexFile: true,
    includeFolderIndexFile: true,
    sortMenusByFrontmatterOrder: true,
    frontmatterOrderDefaultValue: 999,
  },
])
