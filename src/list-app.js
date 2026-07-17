export function createApp({ adapter, interpreter, renderer, store, sync, ui }) {
  let state = null;
  let viewMode = 'list';

  function render() {
    renderer.render(state, viewMode);
  }

  function handleEffect(effect) {
    if (!effect) return;
    if (effect.type === 'modal') ui.openModal(effect.mode, effect.itemId, state);
  }

  async function dispatchUserInput(input) {
    const result = interpreter.execute(state, input);

    if (result.viewMode) {
      viewMode = result.viewMode;
      render();
      return;
    }

    if (result.effect) {
      handleEffect(result.effect);
      return;
    }

    if (result.patch?.length || result.actionLogEntry) {
      state = store.applyMutation(result);
      render();
      sync.enqueue(state, result.actionLogEntry);
    }
  }

  return {
    async init() {
      state = store.load();
      const backendState = await adapter.load();
      if (backendState) state = store.replaceState(backendState);

      ui.setDispatch(dispatchUserInput);
      ui.setGetState(() => state);
      ui.bindGlobal();
      render();
      sync.start?.(() => state);
    },
    getState() {
      return state;
    },
    onSyncedState(nextState) {
      state = nextState;
      render();
    }
  };
}
