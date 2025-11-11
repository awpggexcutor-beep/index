<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Lua Editor</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs/loader.min.js"></script>
  <style>
    :root {
      --radius: 6px;
      --tab: #202324;
      --tab-active: #2c2f30;
      --bg: #181a1b;
      --tab-anim-duration: 220ms;
      --tab-anim-easing: cubic-bezier(.2,.9,.2,1);
      --tab-anim-translate: -6px;
      --tab-anim-scale: 0.96;
    }
    body { margin: 0; background: var(--bg); font-family: sans-serif; height: 100vh; display: flex; flex-direction: column; }
    .tab-bar { display: flex; gap: 8px; padding: 6px; background: #1E1E1E; margin-bottom: 0px; }
    .tab { background: var(--tab); color: #e6e9eb; font-size: 13px; height: 28px; padding: 0 10px; border-radius: var(--radius); display: flex; align-items: center; gap: 6px; cursor: pointer; border: 1px solid rgba(255,255,255,0.03); transition: background 0.2s, filter 0.2s, transform var(--tab-anim-duration) var(--tab-anim-easing), opacity var(--tab-anim-duration) var(--tab-anim-easing); will-change: transform, opacity; opacity: 1; transform: translateY(0) scale(1); }
    .tab.active { background: var(--tab-active); box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
    .tab:hover { filter: brightness(1.08); }
    .tab .name { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .tab .close-btn { width: 16px; height: 16px; border-radius: 4px; display: grid; place-items: center; cursor: pointer; font-size: 12px; border: none; background: transparent; color: #bfc4c8; padding: 0; }
    .tab .close-btn:hover { background: rgba(255,255,255,0.05); }
    .add-btn { width: 28px; height: 28px; border-radius: var(--radius); border: 1px solid rgba(255,255,255,0.03); background: var(--tab); display: grid; place-items: center; cursor: pointer; font-size: 18px; color: #e6e9eb; transition: background 0.2s; }
    .add-btn:hover { background: var(--tab-active); }
    .editor-container { flex: 1; position: relative; }
    #container { position: absolute; top: 0; left: 0; right: 0; bottom: 0; }
    .monaco-editor, .monaco-editor .margin, .monaco-editor-background, .monaco-editor .monaco-editor-background { background-color: #1E1E1E !important; }
    .tab.leaving { opacity: 0; transform: translateY(var(--tab-anim-translate)) scale(var(--tab-anim-scale)); }
    .tab.entering { opacity: 0; transform: translateY(calc(var(--tab-anim-translate) * -1)) scale(0.98); }
    .ctx-menu { position: fixed; background: #1E1E1E; color: #e6e9eb; border-radius: 6px; padding: 6px; box-shadow: 0 8px 24px rgba(0,0,0,0.6); display: none; z-index: 50; }
    .ctx-menu button { display: block; background: transparent; border: none; color: inherit; padding: 8px 12px; text-align: left; width: 100%; cursor: pointer; border-radius: 4px; }
    .ctx-menu button:hover { background: rgba(255,255,255,0.05); }
    .rename-input { position: absolute; padding: 4px 6px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1); background: #0a0a0a; color: #e6e9eb; font-size: 13px; z-index: 100; }
  </style>
</head>
<body>
  <div class="tab-bar" id="tabs"></div>
  <div class="editor-container"><div id="container"></div></div>
  <div id="ctxMenu" class="ctx-menu"><button id="renameAction">Rename</button></div>

  <script>
    let tabs = [];
    let activeId = null;
    let editor = null;
    let counter = 0;
    const ctxMenu = document.getElementById("ctxMenu");
    const renameAction = document.getElementById("renameAction");

    require.config({ paths: { vs: "https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs" } });
    require(["vs/editor/editor.main"], function () {

      // 🎨 Add Lua syntax highlighting (Monarch tokenizer)
      monaco.languages.register({ id: 'lua' });
      monaco.languages.setMonarchTokensProvider('lua', {
        tokenizer: {
          root: [
            [/\b(function|local|end|if|then|elseif|else|for|in|while|do|repeat|until|return|break|and|or|not)\b/, 'keyword'],
            [/\b(true|false|nil)\b/, 'constant'],
            [/\b(print|require|pairs|ipairs|next|tonumber|tostring|type|string|table|math|os|io|coroutine|debug)\b/, 'predefined'],
            [/[A-Za-z_][A-Za-z0-9_]*/, 'identifier'],
            [/"([^"\\]|\\.)*$/, 'string.invalid'],
            [/"([^"\\]|\\.)*"/, 'string'],
            [/'([^'\\]|\\.)*$/, 'string.invalid'],
            [/'([^'\\]|\\.)*'/, 'string'],
            [/[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?/, 'number.float'],
            [/[0-9]+/, 'number'],
            [/--.*/, 'comment'],
          ],
        },
      });

      // 🎨 Optionally use VS-like dark theme
      monaco.editor.defineTheme('vs-lua-dark', {
        base: 'vs-dark',
        inherit: true,
        rules: [
          { token: 'keyword', foreground: '569CD6' },
          { token: 'number', foreground: 'B5CEA8' },
          { token: 'string', foreground: 'CE9178' },
          { token: 'comment', foreground: '6A9955' },
          { token: 'constant', foreground: '4EC9B0' },
          { token: 'predefined', foreground: 'DCDCAA' },
        ],
        colors: {
          'editor.background': '#1E1E1E'
        }
      });

      editor = monaco.editor.create(document.getElementById("container"), {
        value: "",
        language: "lua",
        theme: "vs-lua-dark",
        minimap: { enabled: false },
        automaticLayout: true
      });

      createNewTab("Script 1");
    });

    function renderTabs() {
      const tabsEl = document.getElementById("tabs");
      tabsEl.innerHTML = "";
      tabs.forEach(tab => {
        const wrap = document.createElement("div");
        wrap.className = "tab" + (tab.id === activeId ? " active" : "");
        wrap.dataset.id = tab.id;
        const name = document.createElement("div");
        name.className = "name";
        name.textContent = tab.name;
        const close = document.createElement("button");
        close.className = "close-btn";
        close.textContent = "×";
        close.onclick = (e) => { e.stopPropagation(); closeTab(tab.id); };
        wrap.appendChild(name); wrap.appendChild(close);
        wrap.onclick = () => setActiveTab(tab.id);
        wrap.addEventListener("contextmenu", e => { e.preventDefault(); showContextMenu(e.clientX, e.clientY, tab.id); });
        tabsEl.appendChild(wrap);
      });
      const addBtn = document.createElement("div");
      addBtn.className = "add-btn";
      addBtn.textContent = "+";
      addBtn.onclick = () => createNewTab("Script " + (counter+1));
      tabsEl.appendChild(addBtn);
    }

    function createNewTab(name, content) {
      counter++;
      const id = 'tab-' + Date.now() + '-' + Math.random().toString(36).slice(2,6);
      const model = monaco.editor.createModel(content || '', 'lua');
      const tab = {id, name: name || ('Script ' + counter), model};
      tabs.push(tab);
      activeId = id;
      setActiveTab(id);
      renderTabs();
      const el = document.querySelector(`[data-id='${id}']`);
      if(el){ el.classList.add("entering"); requestAnimationFrame(() => el.classList.remove("entering")); }
    }

    function closeTab(id) {
      const idx = tabs.findIndex(t => t.id === id);
      if(idx === -1) return;
      const el = document.querySelector(`[data-id='${id}']`);
      if(el){ el.classList.add("leaving"); setTimeout(() => { tabs[idx].model.dispose(); tabs.splice(idx,1); setActiveTab(tabs.length ? tabs[Math.max(0,idx-1)].id : createNewTab('Script 1')); renderTabs(); }, 220); }
      else { tabs[idx].model.dispose(); tabs.splice(idx,1); if(tabs.length){ setActiveTab(tabs[Math.max(0,idx-1)].id); } else { createNewTab('Script 1'); } renderTabs(); }
    }

    function setActiveTab(id) {
      activeId = id;
      const tab = tabs.find(t => t.id === id);
      if(tab){ editor.setModel(tab.model); }
      renderTabs();
    }

    function showContextMenu(x,y,id){ ctxMenu.style.left = x + "px"; ctxMenu.style.top = y + "px"; ctxMenu.style.display = "block"; ctxMenu.dataset.target = id; }
    function hideContextMenu(){ ctxMenu.style.display = "none"; ctxMenu.dataset.target = ""; }
    window.addEventListener("click", e => { if(!ctxMenu.contains(e.target)) hideContextMenu(); });
    renameAction.addEventListener("click", () => { const id = ctxMenu.dataset.target; hideContextMenu(); if(id) startRename(id); });
    function startRename(id){
      const tabEl = document.querySelector(`[data-id='${id}']`);
      if(!tabEl) return;
      const rect = tabEl.getBoundingClientRect();
      const input = document.createElement("input");
      input.value = tabs.find(t => t.id === id).name;
      input.className = "rename-input";
      input.style.left = rect.left + "px";
      input.style.top = rect.top + "px";
      input.style.width = rect.width + "px";
      document.body.appendChild(input);
      input.focus(); input.select();
      function commit(){ const v = input.value.trim() || "Untitled"; tabs = tabs.map(t => t.id === id ? {...t, name:v} : t); document.body.removeChild(input); renderTabs(); }
      input.addEventListener("blur", commit);
      input.addEventListener("keydown", e=>{ if(e.key==="Enter"){ commit(); } if(e.key==="Escape"){ document.body.removeChild(input); } });
    }

    // ========== HELPER FUNCTIONS FOR C# ==========
    window.getActiveTabText = () => {
      const tab = tabs.find(t => t.id === activeId);
      return tab ? tab.model.getValue() : '';
    };

    window.setActiveTabText = (text) => {
      const tab = tabs.find(t => t.id === activeId);
      if(tab && tab.model){ tab.model.setValue(text); }
    };

    window.clearActiveTab = () => { window.setActiveTabText(''); };



// Save all tabs to JSON (for C# to read)
window.getAllTabs = () => {
  return JSON.stringify(
    tabs.map(t => {
      // If this tab is active, copy editor content to its model
      if (t.id === activeId && editor && t.model) {
        t.model.setValue(editor.getValue());
      }
      return {
        id: t.id,
        name: t.name,
        content: t.model ? t.model.getValue() : ""
      };
    })
  );
};
// Load tabs from JSON (from C#)
window.loadTabs = (json) => {
  try {
    const data = JSON.parse(json);
    tabs.forEach(t => t.model.dispose());
    tabs = [];
    counter = 0;
    data.forEach(tab => {
      createNewTab(tab.name, tab.content);
    });
  } catch (err) {
    console.error("Failed to load tabs:", err);
  }
};


  </script>
</body>
</html>
