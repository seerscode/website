
// this file is generated — do not edit it


/// <reference types="@sveltejs/kit" />

/**
 * Environment variables [loaded by Vite](https://vitejs.dev/guide/env-and-mode.html#env-files) from `.env` files and `process.env`. Like [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), this module cannot be imported into client-side code. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured).
 * 
 * _Unlike_ [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), the values exported from this module are statically injected into your bundle at build time, enabling optimisations like dead code elimination.
 * 
 * ```ts
 * import { API_KEY } from '$env/static/private';
 * ```
 * 
 * Note that all environment variables referenced in your code should be declared (for example in an `.env` file), even if they don't have a value until the app is deployed:
 * 
 * ```
 * MY_FEATURE_FLAG=""
 * ```
 * 
 * You can override `.env` values from the command line like so:
 * 
 * ```sh
 * MY_FEATURE_FLAG="enabled" npm run dev
 * ```
 */
declare module '$env/static/private' {
	export const NVM_RC_VERSION: string;
	export const GOMODCACHE: string;
	export const _ZO_DOCTOR: string;
	export const PUPPETEER_CACHE_DIR: string;
	export const VSCODE_CRASH_REPORTER_PROCESS_TYPE: string;
	export const NODE: string;
	export const INIT_CWD: string;
	export const NVM_CD_FLAGS: string;
	export const TERM: string;
	export const SHELL: string;
	export const VSCODE_PROCESS_TITLE: string;
	export const TMPDIR: string;
	export const HOMEBREW_REPOSITORY: string;
	export const npm_config_global_prefix: string;
	export const CONDA_SHLVL: string;
	export const CONDA_PROMPT_MODIFIER: string;
	export const GSETTINGS_SCHEMA_DIR_CONDA_BACKUP: string;
	export const CURSOR_WORKSPACE_LABEL: string;
	export const PIP_CACHE_DIR: string;
	export const MallocNanoZone: string;
	export const CURSOR_TRACE_ID: string;
	export const COLOR: string;
	export const NO_COLOR: string;
	export const npm_config_noproxy: string;
	export const npm_config_local_prefix: string;
	export const NX_CACHE_DIRECTORY: string;
	export const CYPRESS_CACHE_FOLDER: string;
	export const USER: string;
	export const NVM_DIR: string;
	export const CCACHE_DIR: string;
	export const COMMAND_MODE: string;
	export const npm_config_globalconfig: string;
	export const YARN_CACHE_FOLDER: string;
	export const CONDA_EXE: string;
	export const SSH_AUTH_SOCK: string;
	export const __CF_USER_TEXT_ENCODING: string;
	export const npm_execpath: string;
	export const BUN_INSTALL_CACHE_DIR: string;
	export const HOMEBREW_CACHE: string;
	export const ELECTRON_RUN_AS_NODE: string;
	export const npm_config_devdir: string;
	export const PATH: string;
	export const GSETTINGS_SCHEMA_DIR: string;
	export const npm_package_json: string;
	export const _: string;
	export const npm_config_userconfig: string;
	export const npm_config_init_module: string;
	export const __CFBundleIdentifier: string;
	export const CONDA_PREFIX: string;
	export const npm_command: string;
	export const CP_HOME_DIR: string;
	export const PWD: string;
	export const VSCODE_HANDLES_UNCAUGHT_ERRORS: string;
	export const npm_lifecycle_event: string;
	export const EDITOR: string;
	export const VSCODE_ESM_ENTRYPOINT: string;
	export const npm_package_name: string;
	export const CONDA_PKGS_DIRS: string;
	export const CURSOR_AGENT: string;
	export const npm_config_npm_version: string;
	export const PLAYWRIGHT_BROWSERS_PATH: string;
	export const XPC_FLAGS: string;
	export const CURSOR_EXTENSION_HOST_ROLE: string;
	export const FORCE_COLOR: string;
	export const npm_config_node_gyp: string;
	export const GEM_SPEC_CACHE: string;
	export const npm_package_version: string;
	export const XPC_SERVICE_NAME: string;
	export const GRADLE_USER_HOME: string;
	export const SHLVL: string;
	export const HOME: string;
	export const VSCODE_NLS_CONFIG: string;
	export const CI: string;
	export const HOMEBREW_PREFIX: string;
	export const PNPM_STORE_PATH: string;
	export const BUNDLE_PATH: string;
	export const TURBO_CACHE_DIR: string;
	export const NUGET_PACKAGES: string;
	export const NPM_CONFIG_CACHE: string;
	export const LOGNAME: string;
	export const CONDA_PYTHON_EXE: string;
	export const npm_lifecycle_script: string;
	export const GOCACHE: string;
	export const VSCODE_IPC_HOOK: string;
	export const VSCODE_CODE_CACHE_PATH: string;
	export const CONDA_DEFAULT_ENV: string;
	export const npm_config_user_agent: string;
	export const CARGO_TARGET_DIR: string;
	export const VSCODE_PID: string;
	export const INFOPATH: string;
	export const HOMEBREW_CELLAR: string;
	export const POETRY_CACHE_DIR: string;
	export const COMPOSER_HOME: string;
	export const VSCODE_CWD: string;
	export const UV_CACHE_DIR: string;
	export const npm_node_execpath: string;
	export const npm_config_prefix: string;
	export const NODE_ENV: string;
}

/**
 * Similar to [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private), except that it only includes environment variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Values are replaced statically at build time.
 * 
 * ```ts
 * import { PUBLIC_BASE_URL } from '$env/static/public';
 * ```
 */
declare module '$env/static/public' {
	
}

/**
 * This module provides access to runtime environment variables, as defined by the platform you're running on. For example if you're using [`adapter-node`](https://github.com/sveltejs/kit/tree/main/packages/adapter-node) (or running [`vite preview`](https://svelte.dev/docs/kit/cli)), this is equivalent to `process.env`. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured).
 * 
 * This module cannot be imported into client-side code.
 * 
 * Dynamic environment variables cannot be used during prerendering.
 * 
 * ```ts
 * import { env } from '$env/dynamic/private';
 * console.log(env.DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 * 
 * > In `dev`, `$env/dynamic` always includes environment variables from `.env`. In `prod`, this behavior will depend on your adapter.
 */
declare module '$env/dynamic/private' {
	export const env: {
		NVM_RC_VERSION: string;
		GOMODCACHE: string;
		_ZO_DOCTOR: string;
		PUPPETEER_CACHE_DIR: string;
		VSCODE_CRASH_REPORTER_PROCESS_TYPE: string;
		NODE: string;
		INIT_CWD: string;
		NVM_CD_FLAGS: string;
		TERM: string;
		SHELL: string;
		VSCODE_PROCESS_TITLE: string;
		TMPDIR: string;
		HOMEBREW_REPOSITORY: string;
		npm_config_global_prefix: string;
		CONDA_SHLVL: string;
		CONDA_PROMPT_MODIFIER: string;
		GSETTINGS_SCHEMA_DIR_CONDA_BACKUP: string;
		CURSOR_WORKSPACE_LABEL: string;
		PIP_CACHE_DIR: string;
		MallocNanoZone: string;
		CURSOR_TRACE_ID: string;
		COLOR: string;
		NO_COLOR: string;
		npm_config_noproxy: string;
		npm_config_local_prefix: string;
		NX_CACHE_DIRECTORY: string;
		CYPRESS_CACHE_FOLDER: string;
		USER: string;
		NVM_DIR: string;
		CCACHE_DIR: string;
		COMMAND_MODE: string;
		npm_config_globalconfig: string;
		YARN_CACHE_FOLDER: string;
		CONDA_EXE: string;
		SSH_AUTH_SOCK: string;
		__CF_USER_TEXT_ENCODING: string;
		npm_execpath: string;
		BUN_INSTALL_CACHE_DIR: string;
		HOMEBREW_CACHE: string;
		ELECTRON_RUN_AS_NODE: string;
		npm_config_devdir: string;
		PATH: string;
		GSETTINGS_SCHEMA_DIR: string;
		npm_package_json: string;
		_: string;
		npm_config_userconfig: string;
		npm_config_init_module: string;
		__CFBundleIdentifier: string;
		CONDA_PREFIX: string;
		npm_command: string;
		CP_HOME_DIR: string;
		PWD: string;
		VSCODE_HANDLES_UNCAUGHT_ERRORS: string;
		npm_lifecycle_event: string;
		EDITOR: string;
		VSCODE_ESM_ENTRYPOINT: string;
		npm_package_name: string;
		CONDA_PKGS_DIRS: string;
		CURSOR_AGENT: string;
		npm_config_npm_version: string;
		PLAYWRIGHT_BROWSERS_PATH: string;
		XPC_FLAGS: string;
		CURSOR_EXTENSION_HOST_ROLE: string;
		FORCE_COLOR: string;
		npm_config_node_gyp: string;
		GEM_SPEC_CACHE: string;
		npm_package_version: string;
		XPC_SERVICE_NAME: string;
		GRADLE_USER_HOME: string;
		SHLVL: string;
		HOME: string;
		VSCODE_NLS_CONFIG: string;
		CI: string;
		HOMEBREW_PREFIX: string;
		PNPM_STORE_PATH: string;
		BUNDLE_PATH: string;
		TURBO_CACHE_DIR: string;
		NUGET_PACKAGES: string;
		NPM_CONFIG_CACHE: string;
		LOGNAME: string;
		CONDA_PYTHON_EXE: string;
		npm_lifecycle_script: string;
		GOCACHE: string;
		VSCODE_IPC_HOOK: string;
		VSCODE_CODE_CACHE_PATH: string;
		CONDA_DEFAULT_ENV: string;
		npm_config_user_agent: string;
		CARGO_TARGET_DIR: string;
		VSCODE_PID: string;
		INFOPATH: string;
		HOMEBREW_CELLAR: string;
		POETRY_CACHE_DIR: string;
		COMPOSER_HOME: string;
		VSCODE_CWD: string;
		UV_CACHE_DIR: string;
		npm_node_execpath: string;
		npm_config_prefix: string;
		NODE_ENV: string;
		[key: `PUBLIC_${string}`]: undefined;
		[key: `${string}`]: string | undefined;
	}
}

/**
 * Similar to [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), but only includes variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Note that public dynamic environment variables must all be sent from the server to the client, causing larger network requests — when possible, use `$env/static/public` instead.
 * 
 * Dynamic environment variables cannot be used during prerendering.
 * 
 * ```ts
 * import { env } from '$env/dynamic/public';
 * console.log(env.PUBLIC_DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 */
declare module '$env/dynamic/public' {
	export const env: {
		[key: `PUBLIC_${string}`]: string | undefined;
	}
}
