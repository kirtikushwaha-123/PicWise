const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
const maxFileSizeBytes = 16 * 1024 * 1024; // 16 MB
const unavailable = "Information not available";

const form = document.querySelector("#uploadForm");
const input = document.querySelector("#imageInput");
const dropZone = document.querySelector("#dropZone");
const fileError = document.querySelector("#fileError");
const previewWrap = document.querySelector("#previewWrap");
const previewImage = document.querySelector("#previewImage");
const removeImage = document.querySelector("#removeImage");
const analyzeButton = document.querySelector("#analyzeButton");
const loadingState = document.querySelector("#loadingState");
const loadingHeading = document.querySelector("#loadingHeading");
const loadingMessage = document.querySelector("#loadingMessage");
const resultsPanel = document.querySelector("#resultsPanel");

// OCR Review & Quality Gate Elements (STEP 9)
const qualityErrorPanel = document.querySelector("#qualityErrorPanel");
const uploadAnotherBtn = document.querySelector("#uploadAnotherBtn");
const ocrReviewPanel = document.querySelector("#ocrReviewPanel");
const ocrIngredientsInput = document.querySelector("#ocrIngredientsInput");
const ocrNutritionInput = document.querySelector("#ocrNutritionInput");
const ocrOtherInput = document.querySelector("#ocrOtherInput");
const rerunOcrBtn = document.querySelector("#rerunOcrBtn");
const confirmAnalyzeBtn = document.querySelector("#confirmAnalyzeBtn");

// Food-specific result elements
const foodResultsContainer = document.querySelector("#foodResultsContainer");
const personalCareResultsContainer = document.querySelector("#personalCareResultsContainer");
const foodWarningsBanner = document.querySelector("#foodWarningsBanner");
const foodWarningsList = document.querySelector("#foodWarningsList");

const foodSafetyStatusBadge = document.querySelector("#foodSafetyStatusBadge");
const foodSafetyCount = document.querySelector("#foodSafetyCount");
const foodSafetyIngredientsList = document.querySelector("#foodSafetyIngredientsList");

const nutritionStatusBadge = document.querySelector("#nutritionStatusBadge");
const nutritionScoreValue = document.querySelector("#nutritionScoreValue");
const nutritionScoreDenominator = document.querySelector("#nutritionScoreDenominator");
const nutritionScoreNote = document.querySelector("#nutritionScoreNote");
const nutritionNutrientsList = document.querySelector("#nutritionNutrientsList");

const allergyStatusBadge = document.querySelector("#allergyStatusBadge");
const allergyDetectedList = document.querySelector("#allergyDetectedList");
const allergyEmptyNotice = document.querySelector("#allergyEmptyNotice");

// Personal Care-specific result elements (Phase 10B)
const pcWarningsBanner = document.querySelector("#pcWarningsBanner");
const pcWarningsList = document.querySelector("#pcWarningsList");

const pcSafetyStatusBadge = document.querySelector("#pcSafetyStatusBadge");
const pcSafetyCount = document.querySelector("#pcSafetyCount");
const pcSafetyIngredientsList = document.querySelector("#pcSafetyIngredientsList");

const pcAllergyStatusBadge = document.querySelector("#pcAllergyStatusBadge");
const pcAllergyCount = document.querySelector("#pcAllergyCount");
const pcAllergyIngredientsList = document.querySelector("#pcAllergyIngredientsList");

const pcIrritationStatusBadge = document.querySelector("#pcIrritationStatusBadge");
const pcIrritationCount = document.querySelector("#pcIrritationCount");
const pcIrritationIngredientsList = document.querySelector("#pcIrritationIngredientsList");

let selectedFile = null;
let isAnalyzing = false;

// Event Listeners
input.addEventListener("change", () => {
  setSelectedFile(input.files[0]);
});

// Category Switch: Immediately clear stale results and errors
const categoryRadios = form.querySelectorAll("input[name='category']");
categoryRadios.forEach((radio) => {
  radio.addEventListener("change", () => {
    resetResults();
    fileError.textContent = "";
  });
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  if (event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files.length > 0) {
    setSelectedFile(event.dataTransfer.files[0]);
  }
});

removeImage.addEventListener("click", () => {
  selectedFile = null;
  input.value = "";
  previewImage.removeAttribute("src");
  previewWrap.classList.add("hidden");
  analyzeButton.disabled = true;
  fileError.textContent = "";
  resetResults();
});

if (uploadAnotherBtn) {
  uploadAnotherBtn.addEventListener("click", () => {
    removeImage.click();
  });
}

// Form Submission
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (isAnalyzing) {
    return;
  }

  if (!selectedFile) {
    fileError.textContent = "Please select an image before analyzing.";
    return;
  }

  const categoryInput = form.querySelector("input[name='category']:checked");
  const selectedCategory = categoryInput ? categoryInput.value : "";
  if (!selectedCategory) {
    fileError.textContent = "Please select a product category (Food or Personal Care).";
    return;
  }

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("category", selectedCategory);

  if (selectedCategory === "food") {
    setLoadingState(true, "Reading your product label...", "PicWise is extracting text from the uploaded label.");
  } else {
    setLoadingState(true, "Analyzing your product...", "PicWise is evaluating Personal Care safety, allergens, and irritation.");
  }
  resetResults();
  fileError.textContent = "";

  try {
    if (selectedCategory === "food") {
      console.log("[PicWise] Starting /api/food/ocr");
      // Step 1, 2, 3: Quality Check & OCR
      const response = await fetch("/api/food/ocr", {
        method: "POST",
        body: formData,
      });

      console.log("[PicWise] /api/food/ocr response: " + response.status);

      let data;
      try {
        data = await response.json();
      } catch (parseErr) {
        throw new Error("Received an invalid response from the server.");
      }

      console.log("[PicWise] OCR status: " + (data && data.status));

      if (!response.ok) {
        if (data && data.status === "unusable_image") {
          if (qualityErrorPanel) qualityErrorPanel.classList.remove("hidden");
          return;
        }
        const errMsg = (data && data.error) || (data && Array.isArray(data.errors) && data.errors[0]) || "OCR extraction failed.";
        throw new Error(errMsg);
      }

      // Step 4: Show OCR Review Screen
      console.log("[PicWise] Showing OCR Review");
      if (ocrIngredientsInput) {
        ocrIngredientsInput.value = (data.raw_text && data.raw_text.ingredients_text) || "";
      }
      if (ocrNutritionInput) {
        ocrNutritionInput.value = (data.raw_text && data.raw_text.nutrition_text) || "";
      }
      if (ocrOtherInput) {
        ocrOtherInput.value = (data.raw_text && data.raw_text.other_text) || "";
      }

      if (ocrReviewPanel) ocrReviewPanel.classList.remove("hidden");
    } else {
      // Personal care flow remains direct
      const response = await fetch("/api/personal-care/analyze", {
        method: "POST",
        body: formData,
      });

      let data;
      try {
        data = await response.json();
      } catch (parseErr) {
        throw new Error("Received an invalid response from the server.");
      }

      if (!response.ok) {
        const errMsg = (data && data.error) || (data && Array.isArray(data.errors) && data.errors[0]) || "Analysis failed.";
        throw new Error(errMsg);
      }

      renderPersonalCareAnalysis(data);
      resultsPanel.classList.remove("hidden");
    }
  } catch (error) {
    const networkMsg = "Unable to connect to the server. Please check that PicWise is running and try again.";
    fileError.textContent = (error.message === "Failed to fetch") ? networkMsg : (error.message || "An error occurred during analysis.");
  } finally {
    setLoadingState(false);
  }
});

// Re-run OCR Action
if (rerunOcrBtn) {
  rerunOcrBtn.addEventListener("click", () => {
    form.requestSubmit();
  });
}

// Confirm and Analyze Action (STEP 9)
if (confirmAnalyzeBtn) {
  confirmAnalyzeBtn.addEventListener("click", async () => {
    if (isAnalyzing) return;
    setLoadingState(true, "Analyzing your product...", "PicWise is evaluating Food Safety, Allergy Risk, and Nutrition.");
    fileError.textContent = "";

    try {
      const payload = {
        category: "food",
        ingredients_text: ocrIngredientsInput ? ocrIngredientsInput.value : "",
        nutrition_text: ocrNutritionInput ? ocrNutritionInput.value : "",
        all_text: ocrOtherInput ? ocrOtherInput.value : "",
      };

      const response = await fetch("/api/food/analyze-text", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      let data;
      try {
        data = await response.json();
      } catch (parseErr) {
        throw new Error("Received an invalid response from the server.");
      }

      if (!response.ok) {
        const errMsg = (data && data.error) || (data && Array.isArray(data.errors) && data.errors[0]) || "Analysis failed.";
        throw new Error(errMsg);
      }

      renderFoodAnalysis(data);
      resultsPanel.classList.remove("hidden");
      if (ocrReviewPanel) ocrReviewPanel.classList.add("hidden");
    } catch (err) {
      const networkMsg = "Unable to connect to the server. Please check that PicWise is running and try again.";
      fileError.textContent = (err.message === "Failed to fetch") ? networkMsg : (err.message || "An error occurred during analysis.");
    } finally {
      setLoadingState(false);
    }
  });
}

function setSelectedFile(file) {
  fileError.textContent = "";
  resetResults();

  if (!file) {
    return;
  }

  if (file.size === 0) {
    selectedFile = null;
    input.value = "";
    previewWrap.classList.add("hidden");
    analyzeButton.disabled = true;
    fileError.textContent = "Uploaded image file is empty.";
    return;
  }

  if (!allowedTypes.includes(file.type)) {
    selectedFile = null;
    input.value = "";
    previewWrap.classList.add("hidden");
    analyzeButton.disabled = true;
    fileError.textContent = "Only JPG, JPEG, PNG, and WEBP images are supported.";
    return;
  }

  if (file.size > maxFileSizeBytes) {
    selectedFile = null;
    input.value = "";
    previewWrap.classList.add("hidden");
    analyzeButton.disabled = true;
    fileError.textContent = "Image exceeds the maximum allowed size of 16MB.";
    return;
  }

  selectedFile = file;
  previewImage.src = URL.createObjectURL(file);
  previewWrap.classList.remove("hidden");
  analyzeButton.disabled = false;
}

function setLoadingState(isLoading, heading, message) {
  isAnalyzing = isLoading;
  if (isLoading) {
    if (loadingHeading) {
      loadingHeading.textContent = heading || "Reading your product label...";
    }
    if (loadingMessage) {
      loadingMessage.textContent = message || "PicWise is extracting text from the uploaded label.";
    }
    loadingState.classList.remove("hidden");
    analyzeButton.disabled = true;
    if (confirmAnalyzeBtn) confirmAnalyzeBtn.disabled = true;
    if (rerunOcrBtn) rerunOcrBtn.disabled = true;
  } else {
    loadingState.classList.add("hidden");
    analyzeButton.disabled = !selectedFile;
    if (confirmAnalyzeBtn) confirmAnalyzeBtn.disabled = false;
    if (rerunOcrBtn) rerunOcrBtn.disabled = false;
    if (loadingHeading) {
      loadingHeading.textContent = "Reading your product label...";
    }
    if (loadingMessage) {
      loadingMessage.textContent = "PicWise is extracting text from the uploaded label.";
    }
  }
}

function resetResults() {
  resultsPanel.classList.add("hidden");
  if (ocrReviewPanel) ocrReviewPanel.classList.add("hidden");
  if (qualityErrorPanel) qualityErrorPanel.classList.add("hidden");
  if (ocrIngredientsInput) ocrIngredientsInput.value = "";
  if (ocrNutritionInput) ocrNutritionInput.value = "";
  if (ocrOtherInput) ocrOtherInput.value = "";
  if (foodResultsContainer) foodResultsContainer.classList.add("hidden");
  if (personalCareResultsContainer) personalCareResultsContainer.classList.add("hidden");
  if (foodWarningsBanner) foodWarningsBanner.classList.add("hidden");
  if (pcWarningsBanner) pcWarningsBanner.classList.add("hidden");
  if (foodWarningsList) foodWarningsList.innerHTML = "";
  if (pcWarningsList) pcWarningsList.innerHTML = "";
  if (foodSafetyIngredientsList) foodSafetyIngredientsList.innerHTML = "";
  if (nutritionNutrientsList) nutritionNutrientsList.innerHTML = "";
  if (allergyDetectedList) allergyDetectedList.innerHTML = "";
  if (pcSafetyIngredientsList) pcSafetyIngredientsList.innerHTML = "";
  if (pcAllergyIngredientsList) pcAllergyIngredientsList.innerHTML = "";
  if (pcIrritationIngredientsList) pcIrritationIngredientsList.innerHTML = "";
}

/* ==========================================================================
   FOOD ANALYSIS RENDERING (Phase 9J)
   Consumes backend presentation object strictly.
   NO overall product health score, NO overall product color.
   ========================================================================== */
const STATUS_EMOJI_MAP = {
  green: "🟢",
  yellow: "🟡",
  orange: "🟠",
  red: "🔴",
  unavailable: "⚪"
};

function renderFoodAnalysis(data) {
  if (personalCareResultsContainer) personalCareResultsContainer.classList.add("hidden");
  if (foodResultsContainer) foodResultsContainer.classList.remove("hidden");

  const pres = data.presentation || {};

  renderFoodSafetyCard(data, pres.food_safety);
  renderAllergyCard(data, pres.allergy);
  renderNutritionCard(data, pres.nutrition);
  renderFoodWarnings(data.warnings);
}

/**
 * Renders the Food Safety row based strictly on backend presentation status.
 * Food Safety is ONE overall product-level categorical result.
 * No Food Safety score, no individual ingredient breakdowns, no confidence values.
 */
function renderFoodSafetyCard(data, fsPres) {
  fsPres = fsPres || {};
  const status = fsPres.color || fsPres.status || "unavailable";
  const label = fsPres.label || "Unavailable";
  const emoji = STATUS_EMOJI_MAP[status] || "⚪";

  if (foodSafetyStatusBadge) {
    foodSafetyStatusBadge.className = `status-pill ${escapeHtml(status)}`;
    foodSafetyStatusBadge.textContent = `${emoji} ${label}`;
  }

  if (foodSafetyCount) {
    const totalCount = data.food_safety?.total_ingredients || (data.food_safety?.ingredients ? data.food_safety.ingredients.length : 0);
    foodSafetyCount.textContent = String(totalCount);
  }

  // Individual ingredient breakdowns and confidences are NOT shown in this summary
  if (foodSafetyIngredientsList) {
    foodSafetyIngredientsList.innerHTML = "";
  }
}

/**
 * Renders the Allergy Risk row based strictly on backend presentation status.
 * No Allergy score and no individual breakdown in this summary.
 */
function renderAllergyCard(data, alPres) {
  alPres = alPres || {};
  const status = alPres.color || alPres.status || "unavailable";
  const label = alPres.label || "Unavailable";
  const emoji = STATUS_EMOJI_MAP[status] || "⚪";

  if (allergyStatusBadge) {
    allergyStatusBadge.className = `status-pill ${escapeHtml(status)}`;
    allergyStatusBadge.textContent = `${emoji} ${label}`;
  }

  // Preserve reference to data.allergy?.allergens_detected while keeping summary clean
  const allergensDetected = data.allergy?.allergens_detected;
  if (allergyDetectedList) {
    allergyDetectedList.innerHTML = "";
  }
}

/**
 * Renders the Nutrition row and numerical Nutrition Score based strictly on backend presentation status.
 * Nutrition Score remains the only numerical score displayed in this summary.
 */
function renderNutritionCard(data, nutPres) {
  nutPres = nutPres || {};
  const status = nutPres.color || nutPres.status || "unavailable";
  const label = nutPres.label || "Unavailable";
  const emoji = STATUS_EMOJI_MAP[status] || "⚪";

  if (nutritionStatusBadge) {
    nutritionStatusBadge.className = `status-pill ${escapeHtml(status)}`;
    nutritionStatusBadge.textContent = `${emoji} ${label}`;
  }

  const score = nutPres.score !== undefined && nutPres.score !== null
    ? nutPres.score
    : (data.nutrition && data.nutrition.nutrition_score !== undefined ? data.nutrition.nutrition_score : null);

  if (nutritionScoreValue) {
    if (typeof score === "number" && !isNaN(score)) {
      const formattedScore = Number.isInteger(score) ? score : Math.round(score);
      nutritionScoreValue.textContent = formattedScore;
      if (nutritionScoreDenominator) {
        nutritionScoreDenominator.textContent = " / 100";
        nutritionScoreDenominator.classList.remove("hidden");
      }
    } else {
      nutritionScoreValue.textContent = "Unavailable";
      if (nutritionScoreDenominator) {
        nutritionScoreDenominator.classList.add("hidden");
      }
    }
  }

  if (nutritionNutrientsList) {
    nutritionNutrientsList.innerHTML = "";

    const evaluated = data.nutrition?.nutrients_evaluated;
    if (Array.isArray(evaluated) && evaluated.length > 0) {
      const validNutrients = evaluated.filter(
        (item) => typeof item.amount_per_100g === "number" && !isNaN(item.amount_per_100g)
      );

      if (validNutrients.length > 0) {
        validNutrients.forEach((item) => {
          const row = document.createElement("div");
          row.className = "nutrient-item-row";

          const nutrientName = item.nutrient || "Unknown Nutrient";
          const val = Number.isInteger(item.amount_per_100g)
            ? item.amount_per_100g
            : Math.round(item.amount_per_100g * 10) / 10;
          const unit = item.unit || "g";
          const itemStatus = item.status || "Assessed";

          row.textContent = `${nutrientName}: ${val} ${unit} / 100 g · ${itemStatus}`;
          nutritionNutrientsList.appendChild(row);
        });
      } else {
        const emptyNotice = document.createElement("p");
        emptyNotice.className = "allergy-empty-notice";
        emptyNotice.textContent = "Detailed nutrient breakdown unavailable.";
        nutritionNutrientsList.appendChild(emptyNotice);
      }
    } else {
      const emptyNotice = document.createElement("p");
      emptyNotice.className = "allergy-empty-notice";
      emptyNotice.textContent = "Detailed nutrient breakdown unavailable.";
      nutritionNutrientsList.appendChild(emptyNotice);
    }
  }
}

/**
 * Renders non-fatal warnings or notices from the backend.
 */
function renderFoodWarnings(warnings) {
  if (Array.isArray(warnings) && warnings.length > 0) {
    foodWarningsList.innerHTML = "";
    warnings.forEach((warn) => {
      const li = document.createElement("li");
      li.textContent = warn;
      foodWarningsList.appendChild(li);
    });
    foodWarningsBanner.classList.remove("hidden");
  } else {
    foodWarningsBanner.classList.add("hidden");
  }
}

/* ==========================================================================
   PERSONAL CARE ANALYSIS RENDERING (Phase 10B)
   Consumes backend presentation object strictly across 3 independent dimensions.
   NO overall product health score, NO overall product color.
   ========================================================================== */
function renderPersonalCareAnalysis(data) {
  if (foodResultsContainer) foodResultsContainer.classList.add("hidden");
  if (personalCareResultsContainer) personalCareResultsContainer.classList.remove("hidden");

  const pres = data.presentation || {};
  const ingredients = data.personal_care?.ingredients || [];
  const recognizedCount = data.personal_care?.recognized_ingredients ?? ingredients.filter((i) => i.status === "success").length;
  const totalCount = data.personal_care?.total_ingredients ?? ingredients.length;
  const countDisplay = `${recognizedCount} of ${totalCount}`;

  // 1. Personal Care Safety Card
  renderPersonalCareDimensionCard({
    badgeElem: pcSafetyStatusBadge,
    countElem: pcSafetyCount,
    listElem: pcSafetyIngredientsList,
    dimPres: pres.personal_care_safety,
    dimensionKey: "safety",
    ingredients: ingredients,
    countText: countDisplay,
  });

  // 2. Personal Care Allergy Card
  renderPersonalCareDimensionCard({
    badgeElem: pcAllergyStatusBadge,
    countElem: pcAllergyCount,
    listElem: pcAllergyIngredientsList,
    dimPres: pres.allergy,
    dimensionKey: "allergy",
    ingredients: ingredients,
    countText: countDisplay,
  });

  // 3. Personal Care Irritation Card
  renderPersonalCareDimensionCard({
    badgeElem: pcIrritationStatusBadge,
    countElem: pcIrritationCount,
    listElem: pcIrritationIngredientsList,
    dimPres: pres.irritation,
    dimensionKey: "irritation",
    ingredients: ingredients,
    countText: countDisplay,
  });

  // 4. Warnings / Notices
  renderPersonalCareWarnings(data.warnings, data.ocr_quality_warning);
}

function renderPersonalCareDimensionCard({
  badgeElem,
  countElem,
  listElem,
  dimPres,
  dimensionKey,
  ingredients,
  countText,
}) {
  dimPres = dimPres || {};
  const status = dimPres.color || dimPres.status || "unavailable";
  const label = dimPres.label || "Unavailable";

  if (badgeElem) {
    badgeElem.className = `status-pill ${escapeHtml(status)}`;
    badgeElem.textContent = label;
  }

  if (countElem) {
    countElem.textContent = countText;
  }

  if (!listElem) return;
  listElem.innerHTML = "";

  if (!ingredients || ingredients.length === 0) {
    const emptyNotice = document.createElement("p");
    emptyNotice.className = "allergy-empty-notice";
    emptyNotice.textContent = "No ingredients detected by OCR.";
    listElem.appendChild(emptyNotice);
    return;
  }

  ingredients.forEach((item) => {
    const row = document.createElement("div");
    row.className = "ingredient-badge-row";

    const name = item.matched_name || item.raw_text || unavailable;
    const isSuccess = item.status === "success";

    let itemStatus = "unavailable";
    let itemLabel = "Unavailable";
    let confStr = "";

    if (isSuccess) {
      const dimData = item[dimensionKey] || {};
      const itemPres = (item.presentation && item.presentation[dimensionKey]) || {};
      itemStatus = itemPres.color || itemPres.status || "unavailable";
      itemLabel = itemPres.label || dimData.risk_class || "Assessed";
      if (Number.isFinite(dimData.confidence)) {
        confStr = `Confidence: ${Math.round(dimData.confidence * 100)}%`;
      }
    } else {
      itemStatus = "unavailable";
      if (item.status === "ingredient_not_recognized") {
        itemLabel = "Not Recognized";
        confStr = "Unrecognized ingredient";
      } else {
        itemLabel = "Unavailable";
        confStr = item.reason || "Analysis unavailable";
      }
    }

    row.innerHTML = `
      <div class="ingredient-info">
        <span class="ingredient-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
        ${confStr ? `<span class="ingredient-confidence">${escapeHtml(confStr)}</span>` : ""}
      </div>
      <span class="status-pill ${escapeHtml(itemStatus)}">${escapeHtml(itemLabel)}</span>
    `;
    listElem.appendChild(row);
  });
}

function renderPersonalCareWarnings(warnings, ocrQualityWarning) {
  const allWarns = Array.isArray(warnings) ? [...warnings] : [];
  if (ocrQualityWarning && !allWarns.includes(ocrQualityWarning)) {
    allWarns.unshift(ocrQualityWarning);
  }

  if (allWarns.length > 0) {
    if (pcWarningsList) {
      pcWarningsList.innerHTML = "";
      allWarns.forEach((warn) => {
        const li = document.createElement("li");
        li.textContent = warn;
        if (ocrQualityWarning && warn === ocrQualityWarning) {
          li.className = "quality-advisory-item";
          li.style.fontWeight = "600";
        }
        pcWarningsList.appendChild(li);
      });
    }
    if (pcWarningsBanner) pcWarningsBanner.classList.remove("hidden");
  } else {
    if (pcWarningsBanner) pcWarningsBanner.classList.add("hidden");
  }
}

function formatNutrientName(key) {
  return String(key)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function valueOrUnavailable(value) {
  return value === null || value === undefined || value === "" ? unavailable : value;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
