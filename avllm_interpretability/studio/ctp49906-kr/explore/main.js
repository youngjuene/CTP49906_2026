(function () {
  "use strict";

  const shell = document.getElementById("app-shell");
  if (!shell) {
    return;
  }

  const panelButtons = Array.from(
    document.querySelectorAll("[data-panel-target]"),
  );
  const panels = Array.from(document.querySelectorAll(".panel[data-panel]"));
  const methodButtons = Array.from(
    document.querySelectorAll("[data-method-target]"),
  );
  const methodPanels = Array.from(
    document.querySelectorAll(".method-panel[data-method]"),
  );

  function setTabState(buttons, activeValue, attrName) {
    for (const button of buttons) {
      const isActive = button.dataset[attrName] === activeValue;
      button.setAttribute("aria-selected", String(isActive));
      button.tabIndex = isActive ? 0 : -1;
    }
  }

  function setPanel(panelName, focusPanel) {
    shell.dataset.panel = panelName;
    setTabState(panelButtons, panelName, "panelTarget");

    for (const panel of panels) {
      const isActive = panel.dataset.panel === panelName;
      panel.classList.toggle("is-active", isActive);
      panel.toggleAttribute("hidden", !isActive);
    }

    if (focusPanel) {
      const activePanel = panels.find((panel) => panel.dataset.panel === panelName);
      if (activePanel) {
        activePanel.scrollTop = 0;
      }
    }
  }

  function setMethod(methodName, focusMethod) {
    shell.dataset.method = methodName;
    setTabState(methodButtons, methodName, "methodTarget");

    for (const panel of methodPanels) {
      const isActive = panel.dataset.method === methodName;
      panel.classList.toggle("is-active", isActive);
      panel.toggleAttribute("hidden", !isActive);
    }

    if (focusMethod) {
      const explorePanel = document.getElementById("panel-explore");
      if (explorePanel) {
        explorePanel.scrollTop = 0;
      }
    }
  }

  function handleRovingKeys(event, buttons, attrName, activate) {
    const currentIndex = buttons.indexOf(event.currentTarget);
    if (currentIndex < 0) {
      return;
    }

    let nextIndex = currentIndex;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      nextIndex = (currentIndex + 1) % buttons.length;
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      nextIndex = (currentIndex - 1 + buttons.length) % buttons.length;
    } else if (event.key === "Home") {
      nextIndex = 0;
    } else if (event.key === "End") {
      nextIndex = buttons.length - 1;
    } else {
      return;
    }

    event.preventDefault();
    const nextButton = buttons[nextIndex];
    nextButton.focus();
    activate(nextButton.dataset[attrName], false);
  }

  for (const button of panelButtons) {
    button.addEventListener("click", () => {
      setPanel(button.dataset.panelTarget, true);
    });
    button.addEventListener("keydown", (event) => {
      handleRovingKeys(event, panelButtons, "panelTarget", setPanel);
    });
  }

  for (const button of methodButtons) {
    button.addEventListener("click", () => {
      setMethod(button.dataset.methodTarget, true);
    });
    button.addEventListener("keydown", (event) => {
      handleRovingKeys(event, methodButtons, "methodTarget", setMethod);
    });
  }

  setPanel(shell.dataset.panel || "explore", false);
  setMethod(shell.dataset.method || "diversity", false);
})();
