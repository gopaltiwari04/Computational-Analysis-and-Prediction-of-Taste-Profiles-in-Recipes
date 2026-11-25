const DIMS = ["sweet", "sour", "bitter", "salty", "umami"];
const DIM_COLORS = {
  sweet: "#f4a261",
  sour: "#d9b043",
  bitter: "#2a9d8f",
  salty: "#264653",
  umami: "#e76f51",
};

// Tab navigation
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

// Preset chips
document.querySelectorAll(".chip[data-preset]").forEach((chip) => {
  chip.addEventListener("click", () => {
    document.getElementById("ingredientsInput").value = chip.dataset.preset;
    runAnalyze();
  });
});

function renderRadarSVG(simplexMap) {
  const width = 360;
  const height = 290;
  const cx = width / 2;
  const cy = height / 2 + 6;
  const radius = 88;
  const n = DIMS.length;

  const angleFor = (i) => -Math.PI / 2 + (2 * Math.PI * i) / n;
  const pt = (i, r) => [cx + r * Math.cos(angleFor(i)), cy + r * Math.sin(angleFor(i))];

  let rings = "";
  [0.25, 0.5, 0.75, 1.0].forEach((lvl) => {
    const pts = DIMS.map((_, i) => pt(i, radius * lvl).join(",")).join(" ");
    rings += `<polygon points="${pts}" fill="none" stroke="#dbe2ea" stroke-width="1"/>`;
  });

  let axes = "";
  let labels = "";
  DIMS.forEach((d, i) => {
    const [x, y] = pt(i, radius);
    const [lx, ly] = pt(i, radius + 24);
    const val = Number(simplexMap[d] || 0);
    axes += `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="#cbd5e1" stroke-width="1"/>`;
    labels += `<text x="${lx}" y="${ly}" text-anchor="middle" dominant-baseline="middle" font-size="11" font-weight="700" fill="#1a2332">${d.toUpperCase()} (${(val * 100).toFixed(0)}%)</text>`;
  });

  const maxScale = Math.max(0.5, ...DIMS.map((d) => Number(simplexMap[d] || 0)));
  const polyPts = DIMS.map((d, i) => pt(i, radius * (Number(simplexMap[d] || 0) / maxScale)).join(",")).join(" ");

  return `<svg style="max-width:100%;height:auto;" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" aria-label="5D taste simplex radar">
    ${rings}${axes}
    <polygon points="${polyPts}" fill="rgba(29, 111, 99, 0.28)" stroke="#1d6f63" stroke-width="2.5"/>
    ${labels}
  </svg>`;
}

function renderProbBars(probs) {
  return DIMS.map((d) => {
    const p = Number((probs && probs[d]) || 0);
    return `<div class="prob-bar-row">
      <span>${d}</span>
      <div class="prob-track"><div class="prob-fill" style="width:${(p * 100).toFixed(1)}%;background:${DIM_COLORS[d]}"></div></div>
      <span>${(p * 100).toFixed(1)}%</span>
    </div>`;
  }).join("");
}

async function loadHealth() {
  try {
    const resp = await fetch("/api/health");
    const data = await resp.json();
    const c = data.corpus || {};
    const badge = document.getElementById("corpusBadge");
    if (c.is_synthetic) {
      badge.textContent = "SYNTHETIC SMOKE TEST CORPUS";
      badge.classList.add("synthetic");
    } else {
      badge.textContent = "REAL EMPIRICAL CORPUS";
    }
    document.getElementById("corpusStats").innerHTML = `
      <div><strong>${c.dataset_name || "Dataset"}</strong></div>
      <div>${Number(c.clean_recipe_count || 0).toLocaleString()} clean recipes · ${c.cuisines_represented || 0} cuisines</div>
      <div>${c.lexicon_entries_count || 0} FlavorDB-verified lexicon entries</div>
    `;
  } catch (err) {
    console.error(err);
  }
}

async function runAnalyze() {
  const rawText = document.getElementById("ingredientsInput").value;
  const ingredients = rawText.split(/[,\n;]+/).map((s) => s.trim()).filter(Boolean);
  const errBox = document.getElementById("analyzeError");
  const resGrid = document.getElementById("analyzeResults");
  errBox.classList.add("hidden");

  if (!ingredients.length) {
    errBox.textContent = "Please enter at least one ingredient.";
    errBox.classList.remove("hidden");
    return;
  }

  try {
    const resp = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ingredients }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "Analysis failed.");

    resGrid.classList.remove("hidden");
    const lp = data.lexicon_proxy;
    document.getElementById("radarContainer").innerHTML = renderRadarSVG(lp.normalized_simplex);
    document.getElementById("lexiconMetrics").innerHTML = `
      <div class="metric-pill"><div class="lbl">Dominant Proxy</div><div class="val">${lp.dominant_taste.toUpperCase()}</div></div>
      <div class="metric-pill"><div class="lbl">Entropy H (nats)</div><div class="val">${lp.shannon_entropy_nats.toFixed(3)}</div></div>
      <div class="metric-pill"><div class="lbl">Evenness J</div><div class="val">${lp.taste_evenness_j.toFixed(3)}</div></div>
    `;

    const ml = data.ml_predictions;
    if (ml && ml.available) {
      const s1 = ml.setting_1_full;
      const s2 = ml.setting_2_masked;
      document.getElementById("mlComparisonCards").innerHTML = `
        <div class="ml-box">
          <h4>Setting 1 (Full Ingredients — Rule Recovery): <strong>${s1.predicted_dominant_taste.toUpperCase()}</strong> <span class="tag tag-muted">${s1.status}</span></h4>
          ${s1.status === "success" ? renderProbBars(s1.probabilities) : "<div class='hint'>No recognized training tokens.</div>"}
        </div>
        <div class="ml-box">
          <h4>Setting 2 (Masked Anchors — Co-occurrence Context): <strong>${s2.predicted_dominant_taste.toUpperCase()}</strong> <span class="tag tag-muted">${s2.status}</span></h4>
          ${s2.status === "success" ? renderProbBars(s2.probabilities) : "<div class='hint'>Fewer than 1 non-anchor token remained in training vocabulary.</div>"}
        </div>
      `;
    } else {
      document.getElementById("mlComparisonCards").innerHTML = "<div class='hint'>Model artifact not loaded.</div>";
    }

    const tbody = document.querySelector("#ingredientAuditTable tbody");
    tbody.innerHTML = data.ingredients.map((ing) => {
      const statusBadge = ing.status === "matched"
        ? `<span class="tag tag-ok">matched</span>`
        : ing.status === "taste_unknown"
          ? `<span class="tag tag-warn">taste_unknown</span>`
          : `<span class="tag tag-muted">${ing.status}</span>`;
      const fdbLink = ing.flavordb_entity_id !== "NONE"
        ? `<a href="https://cosylab.iiitd.edu.in/flavordb/entity_details?id=${ing.flavordb_entity_id}" target="_blank" rel="noopener">FDB #${ing.flavordb_entity_id} (${ing.flavordb_entity_name})</a>`
        : "NONE";
      const chem = ing.chemesthetic_pungency !== "none"
        ? `<span class="tag tag-warn">${ing.chemesthetic_pungency}</span>`
        : "none";
      const mols = [ing.primary_aroma_compounds, ing.known_tastant_or_pungent_compounds]
        .filter((x) => x && x !== "NONE")
        .join(" | ") || "—";
      return `<tr>
        <td>${ing.raw}</td>
        <td><strong>${ing.normalized || "—"}</strong></td>
        <td>${statusBadge}</td>
        <td>${ing.is_masked_anchor ? "Yes (masked)" : "No"}</td>
        <td>${chem}</td>
        <td>${fdbLink}</td>
        <td>${mols}</td>
      </tr>`;
    }).join("");
  } catch (err) {
    errBox.textContent = err.message;
    errBox.classList.remove("hidden");
  }
}

document.getElementById("analyzeBtn").addEventListener("click", runAnalyze);

let allCuisines = [];
async function loadCuisines() {
  try {
    const resp = await fetch("/api/cuisines");
    if (!resp.ok) return;
    const data = await resp.json();
    allCuisines = data.cuisines || [];
    const selA = document.getElementById("cuisineSelectA");
    const selB = document.getElementById("cuisineSelectB");
    selA.innerHTML = allCuisines.map((c) => `<option value="${c.cuisine}">${c.cuisine}</option>`).join("");
    selB.innerHTML = allCuisines.map((c) => `<option value="${c.cuisine}">${c.cuisine}</option>`).join("");
    if (allCuisines.length > 1) {
      selA.value = allCuisines.find((c) => c.cuisine === "indian")?.cuisine || allCuisines[0].cuisine;
      selB.value = allCuisines.find((c) => c.cuisine === "chinese")?.cuisine || allCuisines[1].cuisine;
    }
    const renderCompare = () => {
      const a = allCuisines.find((c) => c.cuisine === selA.value);
      const b = allCuisines.find((c) => c.cuisine === selB.value);
      if (!a || !b) return;
      const toMap = (row) => Object.fromEntries(DIMS.map((d) => [d, row[`${d}_mean`]]));
      document.getElementById("cuisineCompareChart").innerHTML = `
        <div class="ml-box"><h4>${a.cuisine} (N=${a.n_recipes}) — Evenness J=${a.mean_taste_evenness}</h4><div class="radar-wrap">${renderRadarSVG(toMap(a))}</div></div>
        <div class="ml-box"><h4>${b.cuisine} (N=${b.n_recipes}) — Evenness J=${b.mean_taste_evenness}</h4><div class="radar-wrap">${renderRadarSVG(toMap(b))}</div></div>
      `;
    };
    selA.addEventListener("change", renderCompare);
    selB.addEventListener("change", renderCompare);
    renderCompare();

    document.querySelector("#cuisineTable tbody").innerHTML = allCuisines.map((c) => `
      <tr>
        <td><strong>${c.cuisine}</strong></td>
        <td>${Number(c.n_recipes).toLocaleString()}</td>
        <td>${Number(c.sweet_mean).toFixed(3)}</td>
        <td>${Number(c.sour_mean).toFixed(3)}</td>
        <td>${Number(c.bitter_mean).toFixed(3)}</td>
        <td>${Number(c.salty_mean).toFixed(3)}</td>
        <td>${Number(c.umami_mean).toFixed(3)}</td>
        <td>${c.most_distinctive_taste} (${Number(c.max_distinctiveness_z) >= 0 ? "+" : ""}${Number(c.max_distinctiveness_z).toFixed(2)})</td>
        <td>${Number(c.mean_taste_evenness).toFixed(3)}</td>
      </tr>
    `).join("");
  } catch (err) {
    console.error(err);
  }
}

async function loadSampleRecipes() {
  try {
    const resp = await fetch("/api/recipes/sample");
    const data = await resp.json();
    const wrap = document.getElementById("sampleRecipeChips");
    wrap.innerHTML = (data.samples || []).map((s) =>
      `<button class="chip" data-rid="${s.recipe_id}">#${s.recipe_id} (${s.cuisine}, ${s.dominant_taste})</button>`
    ).join("");
    wrap.querySelectorAll("button[data-rid]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.getElementById("recipeIdInput").value = btn.dataset.rid;
        runSimilar();
      });
    });
  } catch (err) {
    console.error(err);
  }
}

async function runSimilar() {
  const rid = document.getElementById("recipeIdInput").value;
  const topK = document.getElementById("topKInput").value;
  const errBox = document.getElementById("similarError");
  const out = document.getElementById("similarResults");
  errBox.classList.add("hidden");
  out.classList.add("hidden");

  try {
    const resp = await fetch(`/api/similar/${encodeURIComponent(rid)}?top_k=${encodeURIComponent(topK)}`);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "Lookup failed.");

    const q = data.query;
    out.classList.remove("hidden");
    out.innerHTML = `
      <div class="card">
        <h3>Query Recipe #${q.recipe_id} — ${q.cuisine} (Dominant: ${q.dominant_taste}, J=${q.taste_evenness})</h3>
        <p class="hint"><strong>Normalized Ingredients:</strong> ${q.ingredients.join(", ")}</p>
        <div class="table-scroll">
          <table>
            <thead>
              <tr><th>Neighbor ID</th><th>Cuisine</th><th>Dominant Taste</th><th>Evenness J</th><th>Cosine Similarity</th><th>Ingredients</th></tr>
            </thead>
            <tbody>
              ${data.neighbors.map((n) => `
                <tr>
                  <td><strong>#${n.neighbor_recipe_id}</strong></td>
                  <td>${n.cuisine}</td>
                  <td>${n.dominant_taste}</td>
                  <td>${n.taste_evenness}</td>
                  <td><strong>${Number(n.cosine_similarity).toFixed(4)}</strong></td>
                  <td>${(n.ingredients || []).join(", ")}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  } catch (err) {
    errBox.textContent = err.message;
    errBox.classList.remove("hidden");
  }
}
document.getElementById("similarBtn").addEventListener("click", runSimilar);

async function loadLexicon() {
  const q = document.getElementById("lexiconSearch").value;
  const status = document.getElementById("lexiconStatusFilter").value;
  try {
    const resp = await fetch(`/api/lexicon?q=${encodeURIComponent(q)}&status=${encodeURIComponent(status)}`);
    const data = await resp.json();
    const fmt = (v) => (v === null || v === undefined ? "<span class='tag tag-muted'>unknown</span>" : Number(v).toFixed(2));
    document.querySelector("#lexiconTable tbody").innerHTML = (data.entries || []).map((e) => `
      <tr>
        <td><strong>${e.ingredient_norm}</strong></td>
        <td>${e.annotation_status}</td>
        <td>${e.confidence_level}</td>
        <td>${fmt(e.sweet)}</td>
        <td>${fmt(e.sour)}</td>
        <td>${fmt(e.bitter)}</td>
        <td>${fmt(e.salty)}</td>
        <td>${fmt(e.umami)}</td>
        <td>${e.chemesthetic_pungency}</td>
        <td>${e.flavordb_entity_id !== "NONE" ? `${e.flavordb_entity_id} (${e.flavordb_entity_name})` : "NONE"}</td>
      </tr>
    `).join("");
  } catch (err) {
    console.error(err);
  }
}
document.getElementById("lexiconSearch").addEventListener("input", loadLexicon);
document.getElementById("lexiconStatusFilter").addEventListener("change", loadLexicon);

loadHealth();
loadCuisines();
loadSampleRecipes();
loadLexicon();
runAnalyze();
