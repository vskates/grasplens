const copyButton = document.querySelector("[data-copy-command]");

if (copyButton) {
  const command = copyButton.parentElement?.querySelector("code")?.textContent ?? "";
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(command);
      copyButton.textContent = "Copied";
      window.setTimeout(() => {
        copyButton.textContent = "Copy";
      }, 1400);
    } catch {
      copyButton.textContent = "Select";
    }
  });
}

