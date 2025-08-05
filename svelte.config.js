// svelte.config.js
import { mdsvex } from 'mdsvex';
import remarkMath   from 'remark-math';
import rehypeKatex  from 'rehype-katex';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import adapter from '@sveltejs/adapter-node';

export default {
  extensions: ['.svelte', '.md', '.svx'],

  preprocess: [
    mdsvex({
      extensions: ['.md', '.svx'],
      remarkPlugins: [remarkMath],    // 👈 will now be used
      rehypePlugins: [rehypeKatex]    // 👈 will now be used
    }),
    vitePreprocess()
  ],

  kit: { adapter: adapter() }
};
