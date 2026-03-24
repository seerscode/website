const fetchMarkdownPosts = async () => {
  const allPostFiles = /* @__PURE__ */ Object.assign({ "../posts/challenges-in-superposition.md": () => import("../../../chunks/challenges-in-superposition.js"), "../posts/fractal-partitioning-theory.md": () => import("../../../chunks/fractal-partitioning-theory.js"), "../posts/global-workspace-theory.md": () => import("../../../chunks/global-workspace-theory.js"), "../posts/hard-problem-machines.md": () => import("../../../chunks/hard-problem-machines.js"), "../posts/induction-heads.md": () => import("../../../chunks/induction-heads.js"), "../posts/llms-architecture.md": () => import("../../../chunks/llms-architecture.js"), "../posts/measuring-phi.md": () => import("../../../chunks/measuring-phi.js"), "../posts/opening-the-black-box-awaken-l.md": () => import("../../../chunks/opening-the-black-box-awaken-l.js"), "../posts/qualia-and-silicon.md": () => import("../../../chunks/qualia-and-silicon.js"), "../posts/speed-of-a-thought.md": () => import("../../../chunks/speed-of-a-thought.js") });
  const iterablePostFiles = Object.entries(allPostFiles);
  const allPosts = await Promise.all(
    iterablePostFiles.map(async ([path, resolver]) => {
      const module = await resolver();
      const meta = module.metadata ?? module.meta ?? {};
      const postSlug = path.replace(/^.*posts[/\\]/, "").replace(/\.md$/, "");
      return {
        meta,
        path: `/blog/${postSlug}`
      };
    })
  );
  const validPosts = allPosts.filter((p) => p.meta && (p.meta.date || p.meta.title));
  const sortedPosts = validPosts.sort((a, b) => {
    const dateA = a.meta.date ? new Date(a.meta.date) : /* @__PURE__ */ new Date(0);
    const dateB = b.meta.date ? new Date(b.meta.date) : /* @__PURE__ */ new Date(0);
    return dateB - dateA;
  });
  return sortedPosts;
};
const load = async () => {
  const posts = await fetchMarkdownPosts();
  return {
    posts
  };
};
export {
  load
};
