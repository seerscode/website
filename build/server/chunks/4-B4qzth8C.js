const fetchMarkdownPosts = async () => {
  const allPostFiles = /* @__PURE__ */ Object.assign({ "../posts/challenges-in-superposition.md": () => import('./challenges-in-superposition-C7IbdZpz.js'), "../posts/fractal-partitioning-theory.md": () => import('./fractal-partitioning-theory-zrDs81ti.js'), "../posts/global-workspace-theory.md": () => import('./global-workspace-theory-BqRrSG8I.js'), "../posts/hard-problem-machines.md": () => import('./hard-problem-machines-gF6FH3cO.js'), "../posts/induction-heads.md": () => import('./induction-heads-Dnf2ll9J.js'), "../posts/llms-architecture.md": () => import('./llms-architecture-EXKChhxD.js'), "../posts/measuring-phi.md": () => import('./measuring-phi-BfW33BLs.js'), "../posts/opening-the-black-box-awaken-l.md": () => import('./opening-the-black-box-awaken-l-BsnNYvm6.js'), "../posts/qualia-and-silicon.md": () => import('./qualia-and-silicon-CQi39-j6.js'), "../posts/speed-of-a-thought.md": () => import('./speed-of-a-thought-DasIpdWR.js') });
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

var _page = /*#__PURE__*/Object.freeze({
  __proto__: null,
  load: load
});

const index = 4;
let component_cache;
const component = async () => component_cache ??= (await import('./_page.svelte-PFnQvTES.js')).default;
const universal_id = "src/routes/blog/+page.js";
const imports = ["_app/immutable/nodes/4.C3sZT_SY.js","_app/immutable/chunks/C1FmrZbK.js","_app/immutable/chunks/CcGnvI8A.js","_app/immutable/chunks/Y_klFubo.js","_app/immutable/chunks/IHki7fMi.js"];
const stylesheets = [];
const fonts = [];

export { component, fonts, imports, index, stylesheets, _page as universal, universal_id };
//# sourceMappingURL=4-B4qzth8C.js.map
