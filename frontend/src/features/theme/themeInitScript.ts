/**
 * Runs in <head> before first paint so the saved theme is applied without a
 * flash. Static string (no user input). Mirrors THEME_STORAGE_KEY in theme.ts.
 */
/**
 * Critical canvas colors, inlined in <head> so the page background is always
 * the theme color — even before the main stylesheet loads, or if it is stale.
 * Values mirror --color-bg-base / --color-text-primary in globals.css.
 */
export const THEME_CRITICAL_CSS = `html{background-color:#FFFFFF;color:#24221F;color-scheme:light}html[data-theme="dark"]{background-color:#0B132B;color:#F5EEE4;color-scheme:dark}`;

export const THEME_INIT_SCRIPT = `(function(){try{var t=JSON.parse(localStorage.getItem('confyntra.theme'));document.documentElement.dataset.theme=(t==='dark'||t==='light')?t:'light';}catch(e){document.documentElement.dataset.theme='light';}})();`;
