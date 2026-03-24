const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set(["FPT_CS_Edition.docx","Fractal Partitioning Theory.pdf","favicon.png"]),
	mimeTypes: {".pdf":"application/pdf",".png":"image/png"},
	_: {
		client: {start:"_app/immutable/entry/start.BmfacvVg.js",app:"_app/immutable/entry/app.xN8CwLoZ.js",imports:["_app/immutable/entry/start.BmfacvVg.js","_app/immutable/chunks/CHawhTRN.js","_app/immutable/chunks/CcGnvI8A.js","_app/immutable/chunks/CYgJF_JY.js","_app/immutable/entry/app.xN8CwLoZ.js","_app/immutable/chunks/C1FmrZbK.js","_app/immutable/chunks/CcGnvI8A.js","_app/immutable/chunks/IHki7fMi.js"],stylesheets:[],fonts:[],uses_env_dynamic_public:false},
		nodes: [
			__memo(() => import('./chunks/0-BryJWnA-.js')),
			__memo(() => import('./chunks/1-Fni4itfA.js')),
			__memo(() => import('./chunks/2-BKrdXV0M.js')),
			__memo(() => import('./chunks/3-CK8VSCCu.js')),
			__memo(() => import('./chunks/4-B4qzth8C.js')),
			__memo(() => import('./chunks/5-C09U0A6o.js')),
			__memo(() => import('./chunks/6-CxccsXmL.js')),
			__memo(() => import('./chunks/7-Bv-Aj2xC.js'))
		],
		remotes: {
			
		},
		routes: [
			{
				id: "/",
				pattern: /^\/$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			},
			{
				id: "/about",
				pattern: /^\/about\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 3 },
				endpoint: null
			},
			{
				id: "/api/chat",
				pattern: /^\/api\/chat\/?$/,
				params: [],
				page: null,
				endpoint: __memo(() => import('./chunks/_server-DwaxhvHS.js'))
			},
			{
				id: "/api/persona-post",
				pattern: /^\/api\/persona-post\/?$/,
				params: [],
				page: null,
				endpoint: __memo(() => import('./chunks/_server-C_KSW1N_.js'))
			},
			{
				id: "/blog",
				pattern: /^\/blog\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 4 },
				endpoint: null
			},
			{
				id: "/blog/[slug]",
				pattern: /^\/blog\/([^/]+?)\/?$/,
				params: [{"name":"slug","optional":false,"rest":false,"chained":false}],
				page: { layouts: [0,], errors: [1,], leaf: 5 },
				endpoint: null
			},
			{
				id: "/chat",
				pattern: /^\/chat\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 6 },
				endpoint: null
			},
			{
				id: "/social",
				pattern: /^\/social\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 7 },
				endpoint: null
			}
		],
		prerendered_routes: new Set([]),
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();

const prerendered = new Set([]);

const base = "";

export { base, manifest, prerendered };
//# sourceMappingURL=manifest.js.map
