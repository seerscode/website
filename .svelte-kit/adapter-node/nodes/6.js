

export const index = 6;
let component_cache;
export const component = async () => component_cache ??= (await import('../entries/pages/chat/_page.svelte.js')).default;
export const imports = ["_app/immutable/nodes/6.CuJsNgSK.js","_app/immutable/chunks/CcGnvI8A.js","_app/immutable/chunks/Y_klFubo.js","_app/immutable/chunks/IHki7fMi.js"];
export const stylesheets = ["_app/immutable/assets/6.DIBNJ4XM.css"];
export const fonts = [];
