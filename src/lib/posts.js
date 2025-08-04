
export const fetchMarkdownPosts = async () => {
  // Get all markdown files in /src/posts
  const allPostFiles = import.meta.glob('/src/posts/*.md');
  const iterablePostFiles = Object.entries(allPostFiles);

  const allPosts = await Promise.all(
    iterablePostFiles.map(async ([path, resolver]) => {
      const { metadata } = await resolver();
      // Extract the slug from the path. Example: /src/posts/my-post.md -> my-post
      const postSlug = path.slice(11, -3);

      return {
        meta: metadata,
        path: `/blog/${postSlug}`,
      };
    })
  );

  // Sort posts by date (newest first)
  const sortedPosts = allPosts.sort((a, b) => {
    return new Date(b.meta.date) - new Date(a.meta.date);
  });

  return sortedPosts;
};
