import * as universal from '../entries/pages/blog/_slug_/_page.js';

export const index = 5;
let component_cache;
export const component = async () => component_cache ??= (await import('../entries/pages/blog/_slug_/_page.svelte.js')).default;
export { universal };
export const universal_id = "src/routes/blog/[slug]/+page.js";
export const imports = ["_app/immutable/nodes/5.BTy50nQ3.js","_app/immutable/chunks/C1FmrZbK.js","_app/immutable/chunks/CYgJF_JY.js","_app/immutable/chunks/CcGnvI8A.js","_app/immutable/chunks/IHki7fMi.js"];
export const stylesheets = ["_app/immutable/assets/5.CngFlqL_.css"];
export const fonts = [];
