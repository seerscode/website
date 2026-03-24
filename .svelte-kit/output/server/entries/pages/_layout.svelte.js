import { c as create_ssr_component, e as escape } from "../../chunks/ssr.js";
const Layout = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  return `<div class="min-h-screen bg-slate-50"><header class="py-6 border-b border-slate-200 bg-white/80 backdrop-blur" data-svelte-h="svelte-1ipk6ys"><div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8"><nav class="flex justify-between items-center"><a href="/" class="text-lg font-serif font-semibold text-slate-800 hover:text-slate-900 transition">Marcio Diaz</a> <div class="space-x-6 text-sm"><a href="/blog" class="text-slate-600 hover:text-slate-900 transition duration-150">Research</a> <a href="/about" class="text-slate-600 hover:text-slate-900 transition duration-150">About</a></div></nav></div></header> <main class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-12">${slots.default ? slots.default({}) : ``}</main> <footer class="py-8 mt-16 border-t border-slate-200"><div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-sm text-slate-500">© ${escape((/* @__PURE__ */ new Date()).getFullYear())} Marcio Diaz · Machine Consciousness Research
      <span class="mx-2" data-svelte-h="svelte-jbleqq">·</span> <a href="https://x.com/marciodiaz_ai" target="_blank" rel="noopener noreferrer" class="text-slate-600 hover:text-slate-900 transition" data-svelte-h="svelte-10mfsms">Twitter</a></div></footer></div>`;
});
export {
  Layout as default
};
