import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'AI Infra 学习记录',
  description: 'CUDA / AI Infra 学习笔记与实践',
  lang: 'zh-CN',

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: 'CUDA 笔记', link: '/cuda/' },
      { text: '实践记录', link: '/posts/' },
    ],

    sidebar: {
      '/cuda/': [
        {
          text: 'CUDA 学习笔记',
          items: [
            { text: '概览', link: '/cuda/' },
            { text: 'CMake 构建实践', link: '/cuda/cmake' },
          ]
        }
      ],
      '/posts/': [
        {
          text: '实践记录',
          items: [
            { text: '概览', link: '/posts/' },
            { text: '第一个 CUDA Kernel', link: '/posts/first-kernel' },
          ]
        }
      ]
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
      // { icon: 'github', link: 'https://github.com/你的用户名' }
    ],
  }
})
