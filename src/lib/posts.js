
export const fetchMarkdownPosts = async () => {
  // Get all markdown files in src/posts (relative to this file in src/lib)
  const allPostFiles = import.meta.glob('../posts/*.md');
  const iterablePostFiles = Object.entries(allPostFiles);

  const allPosts = await Promise.all(
    iterablePostFiles.map(async ([path, resolver]) => {
      const module = await resolver();
      const meta = module.metadata ?? module.meta ?? {};
      const postSlug = path.replace(/^.*posts[/\\]/, '').replace(/\.md$/, '');

      return {
        meta,
        path: `/blog/${postSlug}`,
      };
    })
  );

  const validPosts = allPosts.filter((p) => p.meta && (p.meta.date || p.meta.title));
  const sortedPosts = validPosts.sort((a, b) => {
    const dateA = a.meta.date ? new Date(a.meta.date) : new Date(0);
    const dateB = b.meta.date ? new Date(b.meta.date) : new Date(0);
    return dateB - dateA;
  });

  return sortedPosts;
};
