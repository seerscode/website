
import { fetchMarkdownPosts } from '$lib/posts.js';

export const load = async () => {
  const posts = await fetchMarkdownPosts();

  return {
    posts
  };
};
