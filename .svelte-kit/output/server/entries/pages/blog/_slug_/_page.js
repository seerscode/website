import { error } from "@sveltejs/kit";
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
    const post = await __variableDynamicImportRuntimeHelper(/* @__PURE__ */ Object.assign({ "../../../posts/challenges-in-superposition.md": () => import("../../../../chunks/challenges-in-superposition.js"), "../../../posts/fractal-partitioning-theory.md": () => import("../../../../chunks/fractal-partitioning-theory.js"), "../../../posts/global-workspace-theory.md": () => import("../../../../chunks/global-workspace-theory.js"), "../../../posts/hard-problem-machines.md": () => import("../../../../chunks/hard-problem-machines.js"), "../../../posts/induction-heads.md": () => import("../../../../chunks/induction-heads.js"), "../../../posts/llms-architecture.md": () => import("../../../../chunks/llms-architecture.js"), "../../../posts/measuring-phi.md": () => import("../../../../chunks/measuring-phi.js"), "../../../posts/opening-the-black-box-awaken-l.md": () => import("../../../../chunks/opening-the-black-box-awaken-l.js"), "../../../posts/qualia-and-silicon.md": () => import("../../../../chunks/qualia-and-silicon.js"), "../../../posts/speed-of-a-thought.md": () => import("../../../../chunks/speed-of-a-thought.js") }), `../../../posts/${params.slug}.md`, 5);
    return {
      content: post.default,
      meta: post.metadata
    };
  } catch (e) {
    throw error(404, `Could not find post: ${params.slug}`);
  }
}
export {
  load
};
