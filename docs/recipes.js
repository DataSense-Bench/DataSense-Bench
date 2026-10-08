(() => {
  const task = document.getElementById('recipe-task');
  const agent = document.getElementById('recipe-agent');
  const run = document.getElementById('recipe-run');
  const content = document.getElementById('recipe-content');
  const current = document.getElementById('recipe-current');
  const download = document.getElementById('recipe-download');
  let recipes = [], downloadUrl;
  function options(select, entries, preferred) {
    select.replaceChildren(...entries.map(([value, label]) => new Option(label, value)));
    if (entries.some(([value]) => String(value) === preferred)) select.value = preferred;
    select.disabled = entries.length === 0;
  }
  function showRecipe() {
    const recipe = recipes.find(r => r.task === task.value && r.agent === agent.value && String(r.run) === run.value);
    if (!recipe) {
      current.textContent = 'No recipe is available for this selection.';
      content.replaceChildren(); download.hidden = true; return;
    }
    current.textContent = `${task.value === 'bfcl' ? 'BFCL' : 'TBLite'} / ${recipe.label} / Run ${recipe.run}`;
    // HTML is escaped and allowlisted when producing the static recipe dataset.
    content.innerHTML = recipe.html;
    content.scrollTop = 0;
    if (downloadUrl) URL.revokeObjectURL(downloadUrl);
    downloadUrl = URL.createObjectURL(new Blob([recipe.markdown + '\n'], {type: 'text/markdown;charset=utf-8'}));
    download.href = downloadUrl;
    download.download = `${recipe.task}-${recipe.agent}-run${recipe.run}-recipe.md`;
    download.hidden = false;
  }
  function chooseAgent() {
    const oldRun = run.value;
    options(run, recipes.filter(r => r.task === task.value && r.agent === agent.value).map(r => [r.run, `Run ${r.run}`]), oldRun);
    showRecipe();
  }
  function chooseTask() {
    const oldAgent = agent.value;
    const agents = new Map(recipes.filter(r => r.task === task.value).map(r => [r.agent, r.label]));
    options(agent, [...agents.entries()], oldAgent);
    chooseAgent();
  }
  task.addEventListener('change', chooseTask);
  agent.addEventListener('change', chooseAgent);
  run.addEventListener('change', showRecipe);
  fetch('recipes.json?v=1').then(response => {
    if (!response.ok) throw new Error('Recipe request failed');
    return response.json();
  }).then(data => {
    recipes = data; chooseTask();
  }).catch(() => {
    current.textContent = 'Recipes could not be loaded. Please reload the page.';
    task.disabled = true;
  });
})();
