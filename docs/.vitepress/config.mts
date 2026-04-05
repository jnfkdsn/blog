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
      { text: 'CUDA 笔记', link: '/cuda/' },
      { text: '实践记录', link: '/posts/' },
    ],

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
    scanStartPath: 'cuda',
    resolvePath: '/cuda/',
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
])
