/**
 * static/compare.js
 *
 * Frontend controller for the PicWise Food-Only Product Comparison (Step 5).
 * Strictly adheres to architectural constraints:
 * - 2 to 4 food products.
 * - Compares Food Safety, Allergy Risk, Nutrition, and Nutrition Score independently.
 * - NO overall winner.
 * - NO ranking.
 * - NO combined score.
 * - NO Food Safety score.
 * - NO Allergy score.
 * - Uses safe DOM APIs (textContent, createElement) for user/product-provided text.
 */

document.addEventListener("DOMContentLoaded", () => {
  const addSlotBtn = document.querySelector("#addSlotBtn");
  const loadSamplesBtn = document.querySelector("#loadSamplesBtn");
  const compareBtn = document.querySelector("#compareBtn");
  const productSlotsContainer = document.querySelector("#productSlotsContainer");
  const compareError = document.querySelector("#compareError");
  const comparisonResultsPanel = document.querySelector("#comparisonResultsPanel");
  const comparisonTableHeaderRow = document.querySelector("#comparisonTableHeaderRow");
  const comparisonTableBody = document.querySelector("#comparisonTableBody");

  const STATUS_EMOJI_MAP = {
    green: "🟢",
    yellow: "🟡",
    orange: "🟠",
    red: "🔴",
    unavailable: "⚪",
  };

  const MAX_SLOTS = 4;
  const MIN_SLOTS = 2;

  // Sample food analyses for quick testing & demonstration
  const SAMPLE_PRODUCTS = [
    {
      name: "Whole Wheat Bread",
      analysis: {
        category: "food",
        success: true,
        food_safety: {
          risk_class: "Safe",
          status: "success",
          presentation_status: "yellow",
        },
        allergy: {
          product_risk_level: "No Risk",
          product_ui_label: "Allergen-Free",
          status: "success",
          presentation_status: "green",
        },
        nutrition: {
          nutrition_score: 72.0,
          status: "scored",
          presentation_status: "yellow",
        },
        presentation: {
          food_safety: { status: "yellow", label: "Safe", risk_class: "Safe" },
          allergy: { status: "green", label: "Allergen-Free", risk_level: "No Risk" },
          nutrition: { status: "yellow", label: "Better Nutrition", score: 72.0 },
        },
      },
    },
    {
      name: "Peanut Butter Spread",
      analysis: {
        category: "food",
        success: true,
        food_safety: {
          risk_class: "Safe",
          status: "success",
          presentation_status: "yellow",
        },
        allergy: {
          product_risk_level: "High",
          product_ui_label: "High Allergy Risk",
          status: "success",
          presentation_status: "red",
        },
        nutrition: {
          nutrition_score: 54.0,
          status: "scored",
          presentation_status: "yellow",
        },
        presentation: {
          food_safety: { status: "yellow", label: "Safe", risk_class: "Safe" },
          allergy: { status: "red", label: "High Allergy Risk", risk_level: "High" },
          nutrition: { status: "yellow", label: "Better Nutrition", score: 54.0 },
        },
      },
    },
    {
      name: "Potato Chips",
      analysis: {
        category: "food",
        success: true,
        food_safety: {
          risk_class: "Moderate Risk",
          status: "success",
          presentation_status: "orange",
        },
        allergy: {
          product_risk_level: "No Risk",
          product_ui_label: "Allergen-Free",
          status: "success",
          presentation_status: "green",
        },
        nutrition: {
          nutrition_score: 18.0,
          status: "scored",
          presentation_status: "red",
        },
        presentation: {
          food_safety: { status: "orange", label: "Moderate Risk", risk_class: "Moderate Risk" },
          allergy: { status: "green", label: "Allergen-Free", risk_level: "No Risk" },
          nutrition: { status: "red", label: "Low Nutrition", score: 18.0 },
        },
      },
    },
  ];

  function getSlotCount() {
    return productSlotsContainer.querySelectorAll(".product-slot").length;
  }

  function updateSlotControls() {
    const slots = productSlotsContainer.querySelectorAll(".product-slot");
    const count = slots.length;

    slots.forEach((slot, idx) => {
      const badge = slot.querySelector(".slot-badge");
      if (badge) badge.textContent = `Product ${idx + 1}`;

      const removeBtn = slot.querySelector(".remove-slot-btn");
      if (removeBtn) {
        if (count > MIN_SLOTS) {
          removeBtn.classList.remove("hidden");
        } else {
          removeBtn.classList.add("hidden");
        }
      }
    });

    if (addSlotBtn) {
      addSlotBtn.disabled = count >= MAX_SLOTS;
    }
  }

  function createSlot(index) {
    const slot = document.createElement("div");
    slot.className = "product-slot card";
    slot.dataset.slot = String(index);

    const slotHeader = document.createElement("div");
    slotHeader.className = "slot-header";

    const badge = document.createElement("span");
    badge.className = "slot-badge";
    badge.textContent = `Product ${index}`;
    slotHeader.appendChild(badge);

    const removeBtn = document.createElement("button");
    removeBtn.className = "remove-slot-btn";
    removeBtn.type = "button";
    removeBtn.title = "Remove Product";
    removeBtn.textContent = "×";
    removeBtn.addEventListener("click", () => {
      if (getSlotCount() > MIN_SLOTS) {
        slot.remove();
        updateSlotControls();
      }
    });
    slotHeader.appendChild(removeBtn);
    slot.appendChild(slotHeader);

    const slotBody = document.createElement("div");
    slotBody.className = "slot-body";

    const nameLabel = document.createElement("label");
    nameLabel.className = "slot-label";
    nameLabel.textContent = "Product Name";
    slotBody.appendChild(nameLabel);

    const nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.className = "slot-input-name";
    nameInput.placeholder = "e.g. Whole Wheat Bread";
    nameInput.required = true;
    slotBody.appendChild(nameInput);

    const jsonLabel = document.createElement("label");
    jsonLabel.className = "slot-label";
    jsonLabel.textContent = "Analysis JSON";
    slotBody.appendChild(jsonLabel);

    const jsonArea = document.createElement("textarea");
    jsonArea.className = "slot-input-json";
    jsonArea.rows = 6;
    jsonArea.placeholder = "Paste /api/food/analyze JSON response here...";
    jsonArea.required = true;
    slotBody.appendChild(jsonArea);

    const statusIndicator = document.createElement("span");
    statusIndicator.className = "slot-status-indicator empty";
    statusIndicator.textContent = "No analysis loaded";
    slotBody.appendChild(statusIndicator);

    jsonArea.addEventListener("input", () => {
      try {
        const val = jsonArea.value.trim();
        if (!val) {
          statusIndicator.className = "slot-status-indicator empty";
          statusIndicator.textContent = "No analysis loaded";
        } else {
          const parsed = JSON.parse(val);
          if (parsed && typeof parsed === "object") {
            statusIndicator.className = "slot-status-indicator ready";
            statusIndicator.textContent = "✓ Valid analysis JSON";
          }
        }
      } catch {
        statusIndicator.className = "slot-status-indicator error";
        statusIndicator.textContent = "⚠ Invalid JSON syntax";
      }
    });

    slot.appendChild(slotBody);
    return slot;
  }

  // Hook up existing remove buttons
  productSlotsContainer.querySelectorAll(".remove-slot-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const slot = e.target.closest(".product-slot");
      if (slot && getSlotCount() > MIN_SLOTS) {
        slot.remove();
        updateSlotControls();
      }
    });
  });

  // Hook up existing jsonArea input events
  productSlotsContainer.querySelectorAll(".slot-input-json").forEach((textarea) => {
    const slot = textarea.closest(".product-slot");
    const statusIndicator = slot?.querySelector(".slot-status-indicator");
    if (statusIndicator) {
      textarea.addEventListener("input", () => {
        try {
          const val = textarea.value.trim();
          if (!val) {
            statusIndicator.className = "slot-status-indicator empty";
            statusIndicator.textContent = "No analysis loaded";
          } else {
            const parsed = JSON.parse(val);
            if (parsed && typeof parsed === "object") {
              statusIndicator.className = "slot-status-indicator ready";
              statusIndicator.textContent = "✓ Valid analysis JSON";
            }
          }
        } catch {
          statusIndicator.className = "slot-status-indicator error";
          statusIndicator.textContent = "⚠ Invalid JSON syntax";
        }
      });
    }
  });

  // Add product slot
  if (addSlotBtn) {
    addSlotBtn.addEventListener("click", () => {
      const count = getSlotCount();
      if (count < MAX_SLOTS) {
        const newSlot = createSlot(count + 1);
        productSlotsContainer.appendChild(newSlot);
        updateSlotControls();
      }
    });
  }

  // Load sample products
  if (loadSamplesBtn) {
    loadSamplesBtn.addEventListener("click", () => {
      // Ensure we have 3 slots for the 3 samples
      while (getSlotCount() < 3) {
        const newSlot = createSlot(getSlotCount() + 1);
        productSlotsContainer.appendChild(newSlot);
      }
      updateSlotControls();

      const slots = productSlotsContainer.querySelectorAll(".product-slot");
      SAMPLE_PRODUCTS.forEach((sample, idx) => {
        if (slots[idx]) {
          const nameInput = slots[idx].querySelector(".slot-input-name");
          const jsonArea = slots[idx].querySelector(".slot-input-json");
          const indicator = slots[idx].querySelector(".slot-status-indicator");

          if (nameInput) nameInput.value = sample.name;
          if (jsonArea) {
            jsonArea.value = JSON.stringify(sample.analysis, null, 2);
          }
          if (indicator) {
            indicator.className = "slot-status-indicator ready";
            indicator.textContent = "✓ Valid analysis JSON (Sample)";
          }
        }
      });

      showError("");
    });
  }

  function showError(msg) {
    if (!compareError) return;
    if (msg) {
      compareError.textContent = msg;
      compareError.classList.remove("hidden");
    } else {
      compareError.textContent = "";
      compareError.classList.add("hidden");
    }
  }

  // Compare button click handler
  if (compareBtn) {
    compareBtn.addEventListener("click", async () => {
      showError("");

      const slots = productSlotsContainer.querySelectorAll(".product-slot");
      const productsPayload = [];

      for (let i = 0; i < slots.length; i++) {
        const slot = slots[i];
        const nameInput = slot.querySelector(".slot-input-name");
        const jsonArea = slot.querySelector(".slot-input-json");

        const name = (nameInput?.value || "").trim();
        const jsonStr = (jsonArea?.value || "").trim();

        if (!name) {
          showError(`Product ${i + 1} is missing a name.`);
          return;
        }

        if (!jsonStr) {
          showError(`Product ${i + 1} (${name}) is missing analysis JSON.`);
          return;
        }

        let parsed;
        try {
          parsed = JSON.parse(jsonStr);
        } catch {
          showError(`Product ${i + 1} (${name}) has invalid JSON syntax.`);
          return;
        }

        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
          showError(`Product ${i + 1} (${name}) analysis must be a valid JSON object.`);
          return;
        }

        productsPayload.push({
          name: name,
          analysis: parsed,
        });
      }

      if (productsPayload.length < MIN_SLOTS) {
        showError(`At least ${MIN_SLOTS} products are required for comparison.`);
        return;
      }

      if (productsPayload.length > MAX_SLOTS) {
        showError(`At most ${MAX_SLOTS} products can be compared at once.`);
        return;
      }

      try {
        compareBtn.disabled = true;
        compareBtn.textContent = "Comparing...";

        const resp = await fetch("/api/compare", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ products: productsPayload }),
        });

        const data = await resp.json();

        if (!resp.ok) {
          showError(data.error || "An error occurred during product comparison.");
          return;
        }

        renderComparison(data);
      } catch (err) {
        showError("Network error: Could not reach the comparison service.");
      } finally {
        compareBtn.disabled = false;
        compareBtn.innerHTML = '<span class="button-icon">📊</span> Compare Products';
      }
    });
  }

  /**
   * Renders the side-by-side comparison table strictly using safe DOM APIs.
   * Products are columns, rows are:
   * 1. Food Safety
   * 2. Allergy Risk
   * 3. Nutrition
   * 4. Nutrition Score
   */
  function renderComparison(data) {
    if (!comparisonTableHeaderRow || !comparisonTableBody || !comparisonResultsPanel) return;

    // Reset table
    comparisonTableHeaderRow.innerHTML = "";
    comparisonTableBody.innerHTML = "";

    // 1. Header row
    const dimHeader = document.createElement("th");
    dimHeader.className = "dim-col-header";
    dimHeader.textContent = "Dimension";
    comparisonTableHeaderRow.appendChild(dimHeader);

    const products = data.products || [];
    products.forEach((p) => {
      const th = document.createElement("th");
      th.className = "product-col-header";
      // Safe textContent assignment
      th.textContent = p.name || "Product";
      comparisonTableHeaderRow.appendChild(th);
    });

    // 2. Row Definitions (Strictly independent rows)
    const rowConfigs = [
      {
        dimension: "Food Safety",
        key: "food_safety",
        renderCell: (p) => {
          const fs = p.food_safety || {};
          const status = fs.status || "unavailable";
          const label = fs.label || "Unavailable";
          const emoji = STATUS_EMOJI_MAP[status] || "⚪";

          const badge = document.createElement("span");
          badge.className = `status-pill ${status}`;
          badge.textContent = `${emoji} ${label}`;
          return badge;
        },
      },
      {
        dimension: "Allergy Risk",
        key: "allergy",
        renderCell: (p) => {
          const al = p.allergy || {};
          const status = al.status || "unavailable";
          const label = al.label || "Unavailable";
          const emoji = STATUS_EMOJI_MAP[status] || "⚪";

          const badge = document.createElement("span");
          badge.className = `status-pill ${status}`;
          badge.textContent = `${emoji} ${label}`;
          return badge;
        },
      },
      {
        dimension: "Nutrition",
        key: "nutrition",
        renderCell: (p) => {
          const nut = p.nutrition || {};
          const status = nut.status || "unavailable";
          const label = nut.label || "Unavailable";
          const emoji = STATUS_EMOJI_MAP[status] || "⚪";

          const badge = document.createElement("span");
          badge.className = `status-pill ${status}`;
          badge.textContent = `${emoji} ${label}`;
          return badge;
        },
      },
      {
        dimension: "Nutrition Score",
        key: "nutrition_score",
        renderCell: (p) => {
          const score = p.nutrition_score;
          const span = document.createElement("span");
          span.className = "summary-score-value";

          if (typeof score === "number" && !isNaN(score)) {
            const formatted = Number.isInteger(score) ? score : Math.round(score);
            span.textContent = `${formatted} / 100`;
          } else {
            span.textContent = "Unavailable";
            span.classList.add("text-muted");
          }
          return span;
        },
      },
    ];

    rowConfigs.forEach((cfg) => {
      const tr = document.createElement("tr");
      tr.className = `comparison-row row-${cfg.key}`;

      const dimCell = document.createElement("td");
      dimCell.className = "dim-label-cell";
      dimCell.textContent = cfg.dimension;
      tr.appendChild(dimCell);

      products.forEach((p) => {
        const td = document.createElement("td");
        td.className = "product-val-cell";
        td.appendChild(cfg.renderCell(p));
        tr.appendChild(td);
      });

      comparisonTableBody.appendChild(tr);
    });

    comparisonResultsPanel.classList.remove("hidden");
    comparisonResultsPanel.scrollIntoView({ behavior: "smooth" });
  }

  // Initial setup
  updateSlotControls();
});
