const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Danny Plaza";
pres.title = "Predicción del Riesgo País de Ecuador mediante Machine Learning";

// ── Image paths ──
const IMG = "C:\\Users\\danny\\Maestría Ciencia de Datos USFQ\\tesis-riesgo-pais-ecuador\\notebooks\\";

// ── Color palette ──
const C = {
  navy:      "1B3A5C",
  navyDark:  "0F2440",
  gold:      "C89B3C",
  goldLight: "F5ECD7",
  white:     "FFFFFF",
  offWhite:  "F7F8FA",
  text:      "333333",
  textLight: "666666",
  gray:      "E0E0E0",
  grayMed:   "AAAAAA",
  green:     "2E7D32",
  red:       "C62828",
  blue:      "1565C0",
};

// ── Helpers ──
function addTopBar(slide) {
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.65, fill: { color: C.navy } });
}
function addFooter(slide) {
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.25, w: 10, h: 0.375, fill: { color: C.navy } });
  slide.addText("Danny Plaza — USFQ — Maestría en Ciencia de Datos", {
    x: 0.5, y: 5.28, w: 9, h: 0.3, fontSize: 8, color: C.white, fontFace: "Calibri", align: "right",
  });
}
function addTitle(slide, title) {
  addTopBar(slide);
  slide.addText(title, { x: 0.5, y: 0.08, w: 9, h: 0.5, fontSize: 22, fontFace: "Calibri", bold: true, color: C.white, margin: 0 });
}
function contentSlide(title) {
  const s = pres.addSlide();
  addTitle(s, title);
  addFooter(s);
  return s;
}

// ═══════════════════════════════════════════════════════════
// SLIDE 1 — PORTADA
// ═══════════════════════════════════════════════════════════
let s1 = pres.addSlide();
s1.background = { color: C.navy };
s1.addShape(pres.shapes.RECTANGLE, { x: 0.9, y: 1.1, w: 0.07, h: 2.6, fill: { color: C.gold } });
s1.addText("Predicción del Riesgo País\nde Ecuador mediante\nMachine Learning", {
  x: 1.2, y: 1.0, w: 7.5, h: 1.8, fontSize: 32, fontFace: "Calibri", bold: true, color: C.white, lineSpacingMultiple: 1.1,
});
s1.addText("con Variables de Análisis de Sentimiento de Noticias", {
  x: 1.2, y: 2.7, w: 7.5, h: 0.5, fontSize: 16, fontFace: "Calibri", italic: true, color: C.gold,
});
s1.addShape(pres.shapes.RECTANGLE, { x: 1.2, y: 3.4, w: 3, h: 0.02, fill: { color: C.gold } });
s1.addText([
  { text: "Danny Plaza", options: { fontSize: 16, bold: true, breakLine: true } },
  { text: "Pre-defensa de Tesis", options: { fontSize: 12, breakLine: true } },
  { text: "Maestría en Ciencia de Datos — USFQ", options: { fontSize: 12, breakLine: true } },
  { text: "Mayo 2025", options: { fontSize: 12 } },
], { x: 1.2, y: 3.6, w: 5, h: 1.5, fontFace: "Calibri", color: C.white, lineSpacingMultiple: 1.4 });


// ═══════════════════════════════════════════════════════════
// SLIDE 2 — EL RIESGO PAÍS
// ═══════════════════════════════════════════════════════════
let s2 = contentSlide("El Riesgo País de Ecuador");
s2.addText([
  { text: "El EMBI (Emerging Markets Bond Index) mide el spread soberano de Ecuador sobre bonos del Tesoro de EE.UU., expresado en puntos básicos.", options: { bullet: true, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 5 } },
  { text: "Afecta directamente: costo de financiamiento público, decisiones de inversión extranjera, planificación fiscal del gobierno.", options: { bullet: true, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 5 } },
  { text: "Ecuador: economía dolarizada, petro-dependiente, con historial de default soberano (2008, bajo Rafael Correa).", options: { bullet: true, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 5 } },
  { text: "Período de estudio: noviembre 2013 — diciembre 2025 (3,047 días de trading).", options: { bullet: true } },
], { x: 0.5, y: 0.85, w: 9, h: 3.5, fontSize: 12, fontFace: "Calibri", color: C.text, valign: "top", paraSpaceAfter: 4 });

// Key stat bar
s2.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.3, w: 9, h: 0.7, fill: { color: C.offWhite } });
s2.addText([
  { text: "3,047 ", options: { bold: true, fontSize: 18, color: C.navy } },
  { text: "días  ×  ", options: { fontSize: 12 } },
  { text: "22 ", options: { bold: true, fontSize: 18, color: C.navy } },
  { text: "variables  |  ", options: { fontSize: 12 } },
  { text: "5 ", options: { bold: true, fontSize: 18, color: C.gold } },
  { text: "fuentes de datos  |  ", options: { fontSize: 12 } },
  { text: "4 ", options: { bold: true, fontSize: 18, color: C.gold } },
  { text: "categorías de variables", options: { fontSize: 12 } },
], { x: 0.5, y: 4.3, w: 9, h: 0.7, fontFace: "Calibri", color: C.text, align: "center", valign: "middle" });


// ═══════════════════════════════════════════════════════════
// SLIDE 3 — PREGUNTAS DE INVESTIGACIÓN
// ═══════════════════════════════════════════════════════════
let s3 = contentSlide("Preguntas de Investigación");

// Question 1
s3.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.0, w: 9, h: 1.5, fill: { color: C.goldLight }, line: { color: C.gold, width: 1.5 } });
s3.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.0, w: 0.06, h: 1.5, fill: { color: C.gold } });
s3.addText("Pregunta 1", { x: 0.75, y: 1.05, w: 8.5, h: 0.3, fontSize: 11, fontFace: "Calibri", bold: true, color: C.gold, margin: 0 });
s3.addText("¿Puede el Machine Learning superar a los modelos econométricos clásicos (ARIMA) en la predicción del EMBI Ecuador?", {
  x: 0.75, y: 1.35, w: 8.5, h: 0.8, fontSize: 16, fontFace: "Calibri", color: C.navy, valign: "top", margin: 0,
});

// Question 2
s3.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 2.8, w: 9, h: 1.5, fill: { color: C.goldLight }, line: { color: C.gold, width: 1.5 } });
s3.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 2.8, w: 0.06, h: 1.5, fill: { color: C.gold } });
s3.addText("Pregunta 2", { x: 0.75, y: 2.85, w: 8.5, h: 0.3, fontSize: 11, fontFace: "Calibri", bold: true, color: C.gold, margin: 0 });
s3.addText("¿El análisis de sentimiento de noticias internacionales (NLP via GDELT v2) agrega valor predictivo incremental sobre las variables macroeconómicas tradicionales?", {
  x: 0.75, y: 3.15, w: 8.5, h: 0.8, fontSize: 16, fontFace: "Calibri", color: C.navy, valign: "top", margin: 0,
});


// ═══════════════════════════════════════════════════════════
// SLIDE 4 — FUENTES DE DATOS
// ═══════════════════════════════════════════════════════════
let s4 = contentSlide("Fuentes de Datos");

const srcRows = [
  [
    { text: "Fuente", options: { bold: true, fill: { color: C.navy }, color: C.white, fontSize: 11, align: "center" } },
    { text: "Variables", options: { bold: true, fill: { color: C.navy }, color: C.white, fontSize: 11, align: "center" } },
    { text: "Tipo", options: { bold: true, fill: { color: C.navy }, color: C.white, fontSize: 11, align: "center" } },
    { text: "Frecuencia", options: { bold: true, fill: { color: C.navy }, color: C.white, fontSize: 11, align: "center" } },
  ],
  [{ text: "Banco Central Ecuador", options: { fontSize: 11 } }, { text: "EMBI, tasas de interés", options: { fontSize: 11 } }, { text: "Macro", options: { fontSize: 11, align: "center" } }, { text: "Diaria", options: { fontSize: 11, align: "center" } }],
  [{ text: "BCE (complementario)", options: { fontSize: 11 } }, { text: "Desempleo, reservas internacionales", options: { fontSize: 11 } }, { text: "Macro", options: { fontSize: 11, align: "center" } }, { text: "Mensual/Trim.", options: { fontSize: 11, align: "center" } }],
  [{ text: "Yahoo Finance (yfinance)", options: { fontSize: 11 } }, { text: "WTI, VIX, DXY, oro, treasuries, ETF EMB/HYG", options: { fontSize: 11 } }, { text: "Financiero", options: { fontSize: 11, align: "center" } }, { text: "Diaria", options: { fontSize: 11, align: "center" } }],
  [{ text: "GDELT v2", options: { fontSize: 11 } }, { text: "Tono, Goldstein, volumen, cobertura por actor", options: { fontSize: 11 } }, { text: "NLP", options: { fontSize: 11, align: "center" } }, { text: "Diaria", options: { fontSize: 11, align: "center" } }],
];

s4.addTable(srcRows, {
  x: 0.5, y: 0.9, w: 9, colW: [2.2, 3.2, 1.5, 1.6],
  border: { pt: 0.5, color: C.gray }, rowH: [0.4, 0.4, 0.4, 0.4, 0.4],
  fontFace: "Calibri", color: C.text,
});

s4.addText([
  { text: "Integración: ", options: { bold: true } },
  { text: "merge por fecha → forward fill (datos mensuales a diarios) → filtro por calendario bursátil → ", options: {} },
  { text: "dataset_tesis_final.pkl", options: { bold: true, italic: true } },
], { x: 0.5, y: 3.3, w: 9, h: 0.5, fontSize: 11, fontFace: "Calibri", color: C.text });

// Category distribution
s4.addText("Distribución de las 60 features seleccionadas:", {
  x: 0.5, y: 3.9, w: 9, h: 0.3, fontSize: 12, fontFace: "Calibri", bold: true, color: C.navy, margin: 0,
});

const catData = [
  { label: "Financiero (externo)", n: 29, w: 4.35, color: C.blue },
  { label: "Macroeconómico (local)", n: 21, w: 3.15, color: C.gold },
  { label: "Autorregresivo (EMBI)", n: 6, w: 0.9, color: C.navy },
  { label: "NLP (GDELT)", n: 4, w: 0.6, color: "E65100" },
];
catData.forEach((cat, i) => {
  const yy = 4.3 + i * 0.22;
  s4.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: yy, w: cat.w, h: 0.18, fill: { color: cat.color } });
  s4.addText(`${cat.label} (${cat.n})`, { x: 5.0, y: yy - 0.02, w: 4.5, h: 0.22, fontSize: 9.5, fontFace: "Calibri", color: C.text, margin: 0 });
});


// ═══════════════════════════════════════════════════════════
// SLIDE 5 — PIPELINE
// ═══════════════════════════════════════════════════════════
let s5 = contentSlide("Pipeline Metodológico");

const steps = [
  { num: "1", title: "Recolección", detail: "5 fuentes → 22 variables × 3,047 días\nCSV, Excel, API yfinance, GDELT v2", color: C.navy },
  { num: "2", title: "Feature Engineering", detail: "Lags (1, 7, 30 días) + Rolling means (7, 30 días)\n22 variables → 132 features generados", color: C.blue },
  { num: "3", title: "Selección de Features", detail: "SelectKBest + Mutual Information\n132 → 60 features (ajustado solo en train)", color: C.gold },
  { num: "4", title: "Split Cronológico", detail: "Train 80% (2013-2022) / Val 10% (2022-2024) / Test 10% (2024-2025)\nSin aleatorización — orden temporal estricto", color: "E65100" },
  { num: "5", title: "Modelado + Validación", detail: "TimeSeriesSplit 10-fold + RandomizedSearchCV\nPrevención de data leakage en cada paso", color: C.green },
];

steps.forEach((step, i) => {
  const xL = 0.5;
  const yy = 0.85 + i * 0.88;
  // Number circle
  s5.addShape(pres.shapes.OVAL, { x: xL, y: yy, w: 0.45, h: 0.45, fill: { color: step.color } });
  s5.addText(step.num, { x: xL, y: yy, w: 0.45, h: 0.45, fontSize: 16, fontFace: "Calibri", bold: true, color: C.white, align: "center", valign: "middle" });
  // Connector
  if (i < steps.length - 1) {
    s5.addShape(pres.shapes.RECTANGLE, { x: xL + 0.215, y: yy + 0.45, w: 0.02, h: 0.43, fill: { color: C.gray } });
  }
  // Text
  s5.addText(step.title, { x: 1.15, y: yy, w: 2.5, h: 0.3, fontSize: 14, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });
  s5.addText(step.detail, { x: 1.15, y: yy + 0.28, w: 8.3, h: 0.55, fontSize: 10.5, fontFace: "Calibri", color: C.text, margin: 0, lineSpacingMultiple: 1.15 });
});


// ═══════════════════════════════════════════════════════════
// SLIDE 6 — MODELOS COMPARADOS
// ═══════════════════════════════════════════════════════════
let s6 = contentSlide("Modelos de Predicción");

const models = [
  { title: "ARIMA", sub: "Baseline Econométrico", color: C.grayMed, items: ["Modelo clásico de series de tiempo", "Auto-ARIMA (búsqueda p,d,q)", "Validación rolling, ventana 3 años"], rmse: "107.48 pb", badge: "" },
  { title: "Random Forest", sub: "Ganador", color: C.green, items: ["Ensemble de 891 árboles", "RandomizedSearchCV + 10-fold", "Hiperparámetros optimizados"], rmse: "64.16 pb", badge: "GANADOR" },
  { title: "XGBoost", sub: "Descartado", color: C.red, items: ["Gradient boosting", "Similar búsqueda", "Wilcoxon p=0.0645 (no sig.)"], rmse: "—", badge: "" },
];

models.forEach((m, i) => {
  const xP = 0.5 + i * 3.15;
  s6.addShape(pres.shapes.RECTANGLE, { x: xP, y: 0.85, w: 2.95, h: 3.7, fill: { color: C.offWhite }, shadow: { type: "outer", blur: 4, offset: 1, angle: 135, color: "000000", opacity: 0.08 } });
  s6.addShape(pres.shapes.RECTANGLE, { x: xP, y: 0.85, w: 2.95, h: 0.06, fill: { color: m.color } });
  s6.addText(m.title, { x: xP + 0.15, y: 1.05, w: 2.0, h: 0.35, fontSize: 16, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });
  s6.addText(m.sub, { x: xP + 0.15, y: 1.35, w: 2.65, h: 0.22, fontSize: 10, fontFace: "Calibri", italic: true, color: m.color, margin: 0 });
  const bullets = m.items.map((it, idx) => ({ text: it, options: { bullet: true, breakLine: idx < m.items.length - 1 } }));
  s6.addText(bullets, { x: xP + 0.15, y: 1.7, w: 2.65, h: 1.4, fontSize: 10.5, fontFace: "Calibri", color: C.text, valign: "top", paraSpaceAfter: 6 });
  if (m.rmse !== "—") {
    s6.addShape(pres.shapes.RECTANGLE, { x: xP + 0.15, y: 3.4, w: 2.65, h: 0.7, fill: { color: i === 1 ? "E8F5E9" : C.white } });
    s6.addText("RMSE Test", { x: xP + 0.15, y: 3.4, w: 2.65, h: 0.22, fontSize: 9, fontFace: "Calibri", color: C.textLight, align: "center", margin: 0 });
    s6.addText(m.rmse, { x: xP + 0.15, y: 3.6, w: 2.65, h: 0.4, fontSize: 22, fontFace: "Calibri", bold: true, color: i === 1 ? C.green : C.navy, align: "center", margin: 0 });
  } else {
    s6.addShape(pres.shapes.RECTANGLE, { x: xP + 0.15, y: 3.4, w: 2.65, h: 0.7, fill: { color: "FFEBEE" } });
    s6.addText("Descartado por\nprincipio de parsimonia", { x: xP + 0.15, y: 3.45, w: 2.65, h: 0.6, fontSize: 10, fontFace: "Calibri", color: C.red, align: "center", valign: "middle" });
  }
  if (m.badge === "GANADOR") {
    s6.addShape(pres.shapes.RECTANGLE, { x: xP + 1.7, y: 0.95, w: 1.1, h: 0.26, fill: { color: C.green } });
    s6.addText("GANADOR", { x: xP + 1.7, y: 0.95, w: 1.1, h: 0.26, fontSize: 9, fontFace: "Calibri", bold: true, color: C.white, align: "center", valign: "middle" });
  }
});

s6.addText("Selección: test de Wilcoxon Signed-Rank sobre 10 folds — si no hay diferencia significativa, se elige el modelo más simple.", {
  x: 0.5, y: 4.65, w: 9, h: 0.3, fontSize: 9, fontFace: "Calibri", italic: true, color: C.textLight,
});


// ═══════════════════════════════════════════════════════════
// SLIDE 7 — RESULTADO CENTRAL
// ═══════════════════════════════════════════════════════════
let s7 = contentSlide("Resultado Central: RF Supera al ARIMA en 40.3%");

// Big comparison boxes
s7.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 0.9, w: 3.8, h: 1.8, fill: { color: C.offWhite } });
s7.addText("ARIMA (Baseline)", { x: 0.5, y: 0.95, w: 3.8, h: 0.3, fontSize: 12, fontFace: "Calibri", color: C.textLight, align: "center", margin: 0 });
s7.addText("107.48 pb", { x: 0.5, y: 1.25, w: 3.8, h: 0.65, fontSize: 36, fontFace: "Calibri", bold: true, color: C.textLight, align: "center", margin: 0 });
s7.addShape(pres.shapes.RECTANGLE, { x: 1.2, y: 1.6, w: 2.4, h: 0.03, fill: { color: C.red } });

s7.addText("→", { x: 4.1, y: 1.1, w: 1.1, h: 0.7, fontSize: 40, fontFace: "Calibri", color: C.gold, align: "center", valign: "middle" });
s7.addText("−40.3%", { x: 4.1, y: 1.75, w: 1.1, h: 0.35, fontSize: 14, fontFace: "Calibri", bold: true, color: C.green, align: "center" });

s7.addShape(pres.shapes.RECTANGLE, { x: 5.0, y: 0.9, w: 4.5, h: 1.8, fill: { color: "E8F5E9" }, line: { color: C.green, width: 1.5 } });
s7.addText("Random Forest", { x: 5.0, y: 0.95, w: 4.5, h: 0.3, fontSize: 12, fontFace: "Calibri", color: C.green, align: "center", margin: 0, bold: true });
s7.addText("64.16 pb", { x: 5.0, y: 1.25, w: 4.5, h: 0.65, fontSize: 42, fontFace: "Calibri", bold: true, color: C.green, align: "center", margin: 0 });

// Metrics table
const mRows = [
  [{ text: "Métrica", options: { bold: true, fill: { color: C.navy }, color: C.white, fontSize: 11 } }, { text: "ARIMA", options: { bold: true, fill: { color: C.navy }, color: C.white, fontSize: 11, align: "center" } }, { text: "Random Forest", options: { bold: true, fill: { color: C.navy }, color: C.white, fontSize: 11, align: "center" } }],
  [{ text: "RMSE (pb)", options: { fontSize: 11 } }, { text: "107.48", options: { fontSize: 11, align: "center" } }, { text: "64.16", options: { fontSize: 11, bold: true, color: C.green, align: "center" } }],
  [{ text: "MAE (pb)", options: { fontSize: 11 } }, { text: "—", options: { fontSize: 11, align: "center" } }, { text: "34.90", options: { fontSize: 11, align: "center" } }],
  [{ text: "R²", options: { fontSize: 11 } }, { text: "—", options: { fontSize: 11, align: "center" } }, { text: "0.9555", options: { fontSize: 11, align: "center" } }],
  [{ text: "MAPE", options: { fontSize: 11 } }, { text: "—", options: { fontSize: 11, align: "center" } }, { text: "3.57%", options: { fontSize: 11, align: "center" } }],
];
s7.addTable(mRows, { x: 1.5, y: 3.0, w: 7, colW: [2.3, 2.35, 2.35], border: { pt: 0.5, color: C.gray }, rowH: [0.3, 0.3, 0.3, 0.3, 0.3], fontFace: "Calibri", color: C.text });
s7.addText("El modelo ML explica el 95.5% de la variabilidad del EMBI ecuatoriano", {
  x: 0.5, y: 4.7, w: 9, h: 0.3, fontSize: 11, fontFace: "Calibri", italic: true, color: C.navy, align: "center",
});


// ═══════════════════════════════════════════════════════════
// SLIDE 8 — SHAP BEESWARM (IMAGE)
// ═══════════════════════════════════════════════════════════
let s8 = contentSlide("Interpretabilidad: SHAP Beeswarm Plot");
s8.addImage({ path: IMG + "shap_beeswarm.png", x: 0.3, y: 0.75, w: 9.4, h: 4.35, sizing: { type: "contain", w: 9.4, h: 4.35 } });


// ═══════════════════════════════════════════════════════════
// SLIDE 9 — SHAP CATEGORÍAS (IMAGE)
// ═══════════════════════════════════════════════════════════
let s9 = contentSlide("SHAP: Contribución por Categoría de Variable");
s9.addImage({ path: IMG + "shap_categorias.png", x: 0.3, y: 0.75, w: 6.0, h: 4.35, sizing: { type: "contain", w: 6.0, h: 4.35 } });

// Right side: summary text
s9.addText("Resumen", { x: 6.5, y: 0.85, w: 3.2, h: 0.3, fontSize: 14, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });
const shapSummary = [
  { cat: "Autorregresivo", pct: "43.5%", detail: "Lags y rolling means del EMBI" },
  { cat: "Financiero", pct: "34.2%", detail: "Petróleo, dólar, VIX, treasuries" },
  { cat: "Macroeconómico", pct: "21.4%", detail: "Desempleo, reservas" },
  { cat: "NLP", pct: "0.9%", detail: "Volumen cobertura mediática" },
];
shapSummary.forEach((item, i) => {
  const yy = 1.3 + i * 0.8;
  s9.addText(item.pct, { x: 6.5, y: yy, w: 1.0, h: 0.3, fontSize: 18, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });
  s9.addText(item.cat, { x: 7.5, y: yy, w: 2.2, h: 0.22, fontSize: 11, fontFace: "Calibri", bold: true, color: C.text, margin: 0 });
  s9.addText(item.detail, { x: 7.5, y: yy + 0.22, w: 2.2, h: 0.22, fontSize: 9, fontFace: "Calibri", color: C.textLight, margin: 0 });
});


// ═══════════════════════════════════════════════════════════
// SLIDE 10 — ABLACIÓN RMSE (IMAGE)
// ═══════════════════════════════════════════════════════════
let s10 = contentSlide("Experimento de Ablación: Contribución del NLP");
s10.addImage({ path: IMG + "ablacion_rmse_comparacion.png", x: 0.2, y: 0.75, w: 6.2, h: 4.35, sizing: { type: "contain", w: 6.2, h: 4.35 } });

// Right: table + finding
const ablRows = [
  [{ text: "Grupo", options: { bold: true, fill: { color: C.navy }, color: C.white, fontSize: 10 } }, { text: "Feat.", options: { bold: true, fill: { color: C.navy }, color: C.white, fontSize: 10, align: "center" } }, { text: "RMSE", options: { bold: true, fill: { color: C.navy }, color: C.white, fontSize: 10, align: "center" } }],
  [{ text: "A — Solo AR", options: { fontSize: 10 } }, { text: "6", options: { fontSize: 10, align: "center" } }, { text: "51.31", options: { fontSize: 10, align: "center" } }],
  [{ text: "B — AR + Macro", options: { fontSize: 10 } }, { text: "56", options: { fontSize: 10, align: "center" } }, { text: "60.07", options: { fontSize: 10, align: "center" } }],
  [{ text: "C — AR + NLP", options: { fontSize: 10, bold: true } }, { text: "10", options: { fontSize: 10, align: "center", bold: true } }, { text: "49.01", options: { fontSize: 10, align: "center", bold: true, color: C.green } }],
  [{ text: "D — Todos", options: { fontSize: 10 } }, { text: "60", options: { fontSize: 10, align: "center" } }, { text: "57.48", options: { fontSize: 10, align: "center" } }],
];
s10.addTable(ablRows, { x: 6.5, y: 0.9, w: 3.2, colW: [1.4, 0.6, 0.8], border: { pt: 0.5, color: C.gray }, rowH: [0.3, 0.3, 0.3, 0.35, 0.3], fontFace: "Calibri", color: C.text });

s10.addShape(pres.shapes.RECTANGLE, { x: 6.5, y: 2.8, w: 3.2, h: 1.2, fill: { color: C.goldLight }, line: { color: C.gold, width: 1 } });
s10.addText([
  { text: "NLP mejora +4.5%", options: { fontSize: 14, bold: true, breakLine: true, color: C.navy } },
  { text: "(A→C: 51.31 → 49.01 pb)", options: { fontSize: 10, breakLine: true } },
  { text: "", options: { fontSize: 5, breakLine: true } },
  { text: "4 variables NLP = VOLUMEN de cobertura mediática, no sentimiento", options: { fontSize: 9, italic: true } },
], { x: 6.6, y: 2.85, w: 3.0, h: 1.1, fontFace: "Calibri", color: C.text, valign: "top" });


// ═══════════════════════════════════════════════════════════
// SLIDE 11 — ABLACIÓN PREDICCIONES B vs D (IMAGE)
// ═══════════════════════════════════════════════════════════
let s11 = contentSlide("Ablación: Predicciones B vs D en Test Set");
s11.addImage({ path: IMG + "ablacion_predicciones_BD.png", x: 0.2, y: 0.75, w: 9.6, h: 4.35, sizing: { type: "contain", w: 9.6, h: 4.35 } });


// ═══════════════════════════════════════════════════════════
// SLIDE 12 — GRANGER (IMAGE)
// ═══════════════════════════════════════════════════════════
let s12 = contentSlide("Causalidad de Granger: ¿Anticipa el NLP al EMBI?");
s12.addImage({ path: IMG + "granger_heatmap.png", x: 0.2, y: 0.75, w: 9.6, h: 3.4, sizing: { type: "contain", w: 9.6, h: 3.4 } });

// Bottom findings
s12.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.2, w: 4.2, h: 0.8, fill: { color: "FFEBEE" } });
s12.addText([
  { text: "NLP → EMBI: ", options: { bold: true } },
  { text: "0 lags significativos\nLas noticias NO anticipan el riesgo país", options: {} },
], { x: 0.6, y: 4.25, w: 4.0, h: 0.7, fontSize: 10, fontFace: "Calibri", color: C.text, valign: "middle" });

s12.addShape(pres.shapes.RECTANGLE, { x: 5.3, y: 4.2, w: 4.2, h: 0.8, fill: { color: "E8F5E9" } });
s12.addText([
  { text: "EMBI → NLP: ", options: { bold: true } },
  { text: "tono (lags 1-4) y Goldstein (lag 1)\nEl EMBI alto atrae cobertura más negativa", options: {} },
], { x: 5.4, y: 4.25, w: 4.0, h: 0.7, fontSize: 10, fontFace: "Calibri", color: C.text, valign: "middle" });


// ═══════════════════════════════════════════════════════════
// SLIDE 13 — REGÍMENES (IMAGE)
// ═══════════════════════════════════════════════════════════
let s13 = contentSlide("Regímenes Estructurales: Quiebres del EMBI");
s13.addImage({ path: IMG + "chow_cusum_analysis.png", x: 0.2, y: 0.75, w: 6.3, h: 4.35, sizing: { type: "contain", w: 6.3, h: 4.35 } });

// Right: Chow results
s13.addText("Tests de Quiebre", { x: 6.7, y: 0.85, w: 3, h: 0.3, fontSize: 13, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });

const breaks = [
  { date: "Marzo 2020", f: "F = 5.52", p: "p = 0.0002", event: "Inicio COVID-19" },
  { date: "Enero 2022", f: "F = 11.61", p: "p < 0.0001", event: "Tightening de la Fed" },
];
breaks.forEach((b, i) => {
  const yy = 1.3 + i * 1.2;
  s13.addShape(pres.shapes.OVAL, { x: 6.7, y: yy, w: 0.3, h: 0.3, fill: { color: i === 0 ? C.red : C.green } });
  s13.addText(b.date, { x: 7.1, y: yy, w: 2.6, h: 0.25, fontSize: 12, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });
  s13.addText(`${b.f}  |  ${b.p}`, { x: 7.1, y: yy + 0.25, w: 2.6, h: 0.2, fontSize: 10, fontFace: "Calibri", color: C.text, margin: 0 });
  s13.addText(b.event, { x: 7.1, y: yy + 0.45, w: 2.6, h: 0.2, fontSize: 10, fontFace: "Calibri", italic: true, color: C.textLight, margin: 0 });
});

s13.addShape(pres.shapes.RECTANGLE, { x: 6.7, y: 3.8, w: 3, h: 0.8, fill: { color: C.offWhite } });
s13.addText("CUSUM: p = 0.47\n→ Sin drift acumulado\n→ Quiebres son eventos discretos", {
  x: 6.8, y: 3.85, w: 2.8, h: 0.7, fontSize: 10, fontFace: "Calibri", color: C.text, valign: "middle",
});


// ═══════════════════════════════════════════════════════════
// SLIDE 14 — SHAP POR RÉGIMEN (IMAGE)
// ═══════════════════════════════════════════════════════════
let s14 = contentSlide("SHAP por Régimen: ¿Qué Explica el EMBI en Cada Período?");
s14.addImage({ path: IMG + "shap_regimenes_heatmap.png", x: 0.2, y: 0.75, w: 9.6, h: 4.35, sizing: { type: "contain", w: 9.6, h: 4.35 } });


// ═══════════════════════════════════════════════════════════
// SLIDE 15 — AUTOENCODER (IMAGE)
// ═══════════════════════════════════════════════════════════
let s15 = contentSlide("Autoencoder NLP: ¿Estructura No Lineal?");
s15.addImage({ path: IMG + "autoencoder_comparativa_rmse.png", x: 0.2, y: 0.75, w: 5.5, h: 4.0, sizing: { type: "contain", w: 5.5, h: 4.0 } });

// Right: conclusion
s15.addText("Resultados", { x: 5.9, y: 0.85, w: 3.8, h: 0.3, fontSize: 14, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });

const aeResults = [
  { label: "Original (4 NLP rolling)", rmse: "64.16 pb", highlight: true },
  { label: "PCA-3 (lineal)", rmse: "66.46 pb", highlight: false },
  { label: "Autoencoder-3 (no lineal)", rmse: "66.49 pb", highlight: false },
];
aeResults.forEach((r, i) => {
  const yy = 1.3 + i * 0.6;
  s15.addShape(pres.shapes.RECTANGLE, { x: 5.9, y: yy, w: 3.8, h: 0.5, fill: { color: r.highlight ? "E8F5E9" : C.offWhite } });
  s15.addText(r.label, { x: 6.0, y: yy + 0.02, w: 2.5, h: 0.22, fontSize: 10, fontFace: "Calibri", color: C.text, margin: 0 });
  s15.addText(r.rmse, { x: 8.2, y: yy + 0.02, w: 1.4, h: 0.22, fontSize: 11, fontFace: "Calibri", bold: true, color: r.highlight ? C.green : C.navy, align: "right", margin: 0 });
});

s15.addShape(pres.shapes.RECTANGLE, { x: 5.9, y: 3.3, w: 3.8, h: 1.4, fill: { color: C.goldLight }, line: { color: C.gold, width: 1 } });
s15.addText([
  { text: "Hallazgo clave:", options: { bold: true, breakLine: true, fontSize: 11 } },
  { text: "", options: { fontSize: 4, breakLine: true } },
  { text: "PCA ≈ Autoencoder (Δ = 0.03 pb)\n→ Estructura NLP es lineal\n\nLo que importa: rolling mean 30 días\n(agregación temporal, no compresión)", options: { fontSize: 10 } },
], { x: 6.0, y: 3.35, w: 3.6, h: 1.3, fontFace: "Calibri", color: C.text, valign: "top" });


// ═══════════════════════════════════════════════════════════
// SLIDE 16 — CONCLUSIONES
// ═══════════════════════════════════════════════════════════
let s16 = contentSlide("Conclusiones");

const conclusions = [
  { title: "ML supera a ARIMA en 40.3%", body: "Random Forest (RMSE=64.16 pb) mejora significativamente sobre ARIMA (107.48 pb). El modelo explica 95.5% de la variabilidad del EMBI.", accent: C.green },
  { title: "NLP aporta señal real pero modesta (~4.5%)", body: "El volumen de cobertura mediática internacional mejora la predicción. Opera por acumulación temporal (30 días), no por sentimiento.", accent: C.gold },
  { title: "El riesgo país cambia de régimen", body: "Quiebres en 2020 y 2022. Variables dominantes: commodities → desempleo → política monetaria de la Fed.", accent: C.blue },
  { title: "La señal NLP es lineal y contemporánea", body: "Autoencoder = PCA (0.03 pb). Granger: 0 lags significativos. Noticias y EMBI reaccionan al mismo evento.", accent: "E65100" },
];

conclusions.forEach((c, i) => {
  const yy = 0.85 + i * 1.05;
  // Card
  s16.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: yy, w: 9, h: 0.95, fill: { color: C.offWhite }, shadow: { type: "outer", blur: 3, offset: 1, angle: 135, color: "000000", opacity: 0.06 } });
  s16.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: yy, w: 0.06, h: 0.95, fill: { color: c.accent } });
  s16.addText(c.title, { x: 0.7, y: yy + 0.08, w: 8.6, h: 0.3, fontSize: 13, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });
  s16.addText(c.body, { x: 0.7, y: yy + 0.4, w: 8.6, h: 0.45, fontSize: 10.5, fontFace: "Calibri", color: C.text, margin: 0, lineSpacingMultiple: 1.15 });
});


// ═══════════════════════════════════════════════════════════
// SLIDE 17 — LIMITACIONES Y TRABAJO FUTURO
// ═══════════════════════════════════════════════════════════
let s17 = contentSlide("Limitaciones y Trabajo Futuro");

// Left card
s17.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 0.9, w: 4.3, h: 3.8, fill: { color: C.offWhite }, shadow: { type: "outer", blur: 4, offset: 1, angle: 135, color: "000000", opacity: 0.06 } });
s17.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 0.9, w: 4.3, h: 0.05, fill: { color: C.red } });
s17.addText("Limitaciones", { x: 0.7, y: 1.05, w: 3.9, h: 0.35, fontSize: 14, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });
s17.addText([
  { text: "GDELT v2 disponible solo desde 2013 — no cubre el default soberano de 2008", options: { bullet: true, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 5 } },
  { text: "Variables NLP son métricas agregadas diarias, no texto crudo de noticias", options: { bullet: true, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 5 } },
  { text: "Test de Wilcoxon con 5 folds tiene poder estadístico limitado (p mínimo posible = 0.0625)", options: { bullet: true, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 5 } },
  { text: "Modelo evaluado en período post-COVID; robustez ante futuras crisis no verificada", options: { bullet: true } },
], { x: 0.7, y: 1.5, w: 3.9, h: 3.0, fontSize: 10.5, fontFace: "Calibri", color: C.text, valign: "top", paraSpaceAfter: 4 });

// Right card
s17.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 0.9, w: 4.3, h: 3.8, fill: { color: C.offWhite }, shadow: { type: "outer", blur: 4, offset: 1, angle: 135, color: "000000", opacity: 0.06 } });
s17.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 0.9, w: 4.3, h: 0.05, fill: { color: C.green } });
s17.addText("Trabajo Futuro", { x: 5.4, y: 1.05, w: 3.9, h: 0.35, fontSize: 14, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });
s17.addText([
  { text: "Embeddings de texto con FinBERT o BERT-multilingual sobre titulares GDELT", options: { bullet: true, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 5 } },
  { text: "Extensión a otros EMBI de la región (Colombia, Perú, Argentina)", options: { bullet: true, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 5 } },
  { text: "Modelos Transformer para capturar dependencias temporales de largo plazo", options: { bullet: true, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 5 } },
  { text: "Incorporación de redes sociales (Twitter/X) como fuente NLP complementaria", options: { bullet: true } },
], { x: 5.4, y: 1.5, w: 3.9, h: 3.0, fontSize: 10.5, fontFace: "Calibri", color: C.text, valign: "top", paraSpaceAfter: 4 });


// ═══════════════════════════════════════════════════════════
// SLIDE 18 — CIERRE
// ═══════════════════════════════════════════════════════════
let s18 = pres.addSlide();
s18.background = { color: C.navy };
s18.addShape(pres.shapes.RECTANGLE, { x: 2.8, y: 1.4, w: 0.07, h: 2.4, fill: { color: C.gold } });
s18.addText("Gracias", { x: 3.1, y: 1.3, w: 5, h: 1, fontSize: 44, fontFace: "Calibri", bold: true, color: C.white });
s18.addText("¿Preguntas?", { x: 3.1, y: 2.2, w: 5, h: 0.6, fontSize: 24, fontFace: "Calibri", color: C.gold });
s18.addShape(pres.shapes.RECTANGLE, { x: 3.1, y: 3.0, w: 2.5, h: 0.02, fill: { color: C.gold } });
s18.addText([
  { text: "Danny Plaza", options: { fontSize: 14, bold: true, breakLine: true } },
  { text: "danny.plaza24@gmail.com", options: { fontSize: 11, breakLine: true } },
  { text: "", options: { fontSize: 8, breakLine: true } },
  { text: "Maestría en Ciencia de Datos — USFQ — 2025", options: { fontSize: 11 } },
], { x: 3.1, y: 3.2, w: 5, h: 1.5, fontFace: "Calibri", color: C.white, lineSpacingMultiple: 1.4 });


// ═══════════════════════════════════════════════════════════
// SAVE
// ═══════════════════════════════════════════════════════════
const outPath = process.argv[2] || "presentacion_predefensa.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("Presentation saved: " + outPath);
  console.log("Slides: 18 | Images embedded: 8");
}).catch(err => { console.error("Error:", err); });
