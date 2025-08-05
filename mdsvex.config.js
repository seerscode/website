// mdsvex.config.js

import { visit } from 'unist-util-visit';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config = {
  extensions: ['.svelte.md', '.md', '.svx'],
  
  // Add the plugins here
  remarkPlugins: [remarkMath],
  rehypePlugins: [rehypeKatex],

  smartypants: {
    dashes: 'oldschool'
  },
};

export default config;