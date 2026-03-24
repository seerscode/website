import { c as create_ssr_component, d as each, e as escape, f as add_attribute } from "../../../chunks/ssr.js";
const Page = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { data } = $$props;
  if ($$props.data === void 0 && $$bindings.data && data !== void 0) $$bindings.data(data);
  return `${$$result.head += `<!-- HEAD_svelte-1f7fyrj_START -->${$$result.title = `<title>Research | Marcio Diaz</title>`, ""}<!-- HEAD_svelte-1f7fyrj_END -->`, ""} <h1 class="text-3xl font-serif font-bold text-slate-900 mb-10" data-svelte-h="svelte-9f9b9i">Research &amp; Writings</h1> <div class="space-y-10">${each(data.posts, (post) => {
    return `<article><p class="text-sm text-slate-500 mb-2 font-mono">${escape(new Date(post.meta.date).toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric"
    }))}</p> <h2 class="text-xl font-serif font-semibold text-slate-900 mb-3"><a${add_attribute("href", post.path, 0)} class="hover:text-slate-800 transition border-b border-slate-300 hover:border-slate-600">${escape(post.meta.title)} </a></h2> <p class="text-slate-600">${escape(post.meta.excerpt)}</p> </article>`;
  })}</div>`;
});
export {
  Page as default
};
