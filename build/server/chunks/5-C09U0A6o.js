import { e as error } from './index-B_P6GQpZ.js';

const __variableDynamicImportRuntimeHelper = (glob, path, segs) => {
  const v = glob[path];
  if (v) {
    return typeof v === "function" ? v() : Promise.resolve(v);
  }
  return new Promise((_, reject) => {
    (typeof queueMicrotask === "function" ? queueMicrotask : setTimeout)(
      reject.bind(
        null,
        new Error(
          "Unknown variable dynamic import: " + path + (path.split("/").length !== segs ? ". Note that variables only represent file names one level deep." : "")
        )
      )
    );
  });
};
async function load({ params }) {
  try {
    const post = await __variableDynamicImportRuntimeHelper(/* @__PURE__ */ Object.assign({ "../../../posts/challenges-in-superposition.md": () => import('./challenges-in-superposition-C7IbdZpz.js'), "../../../posts/fractal-partitioning-theory.md": () => import('./fractal-partitioning-theory-zrDs81ti.js'), "../../../posts/global-workspace-theory.md": () => import('./global-workspace-theory-BqRrSG8I.js'), "../../../posts/hard-problem-machines.md": () => import('./hard-problem-machines-gF6FH3cO.js'), "../../../posts/induction-heads.md": () => import('./induction-heads-Dnf2ll9J.js'), "../../../posts/llms-architecture.md": () => import('./llms-architecture-EXKChhxD.js'), "../../../posts/measuring-phi.md": () => import('./measuring-phi-BfW33BLs.js'), "../../../posts/opening-the-black-box-awaken-l.md": () => import('./opening-the-black-box-awaken-l-BsnNYvm6.js'), "../../../posts/qualia-and-silicon.md": () => import('./qualia-and-silicon-CQi39-j6.js'), "../../../posts/speed-of-a-thought.md": () => import('./speed-of-a-thought-DasIpdWR.js') }), `../../../posts/${params.slug}.md`, 5);
    return {
      content: post.default,
      meta: post.metadata
    };
  } catch (e) {
    throw error(404, `Could not find post: ${params.slug}`);
  }
}

var _page = /*#__PURE__*/Object.freeze({
  __proto__: null,
  load: load
});

const index = 5;
let component_cache;
const component = async () => component_cache ??= (await import('./_page.svelte-BzGUgRJH.js')).default;
const universal_id = "src/routes/blog/[slug]/+page.js";
const imports = ["_app/immutable/nodes/5.BTy50nQ3.js","_app/immutable/chunks/C1FmrZbK.js","_app/immutable/chunks/CYgJF_JY.js","_app/immutable/chunks/CcGnvI8A.js","_app/immutable/chunks/IHki7fMi.js"];
const stylesheets = ["_app/immutable/assets/5.CngFlqL_.css"];
const fonts = [];

export { component, fonts, imports, index, stylesheets, _page as universal, universal_id };
//# sourceMappingURL=5-C09U0A6o.js.map
