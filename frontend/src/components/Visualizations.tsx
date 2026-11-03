import { lazy, Suspense, useEffect, useId, useState } from "react";
import * as Slider from "@radix-ui/react-slider";
import { all, create } from "mathjs";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import { Check, ChevronDown, FunctionSquare, MoveUpRight } from "lucide-react";
import type { FunctionPlotConfig, PlotParameter, VisualizationSpec } from "../types";

const math = create(all, { number: "number", precision: 12 });
const Plot = lazy(() => import("react-plotly.js"));
const permittedTokens = /^(?:x|a|b|c|sin|cos|tan|sqrt|abs|exp|log|pi|\d+(?:\.\d*)?|\.\d+|[+\-*/^().,]|\s)+$/;

function sample(expression: string, domain: [number, number], params: Record<string, number> = {}) {
  if (!permittedTokens.test(expression)) throw new Error("Unsupported plot expression");
  const compiled = math.compile(expression);
  const x: number[] = [];
  const y: number[] = [];
  const points = 180;
  for (let index = 0; index <= points; index += 1) {
    const value = domain[0] + ((domain[1] - domain[0]) * index) / points;
    x.push(value);
    try {
      const result = Number(compiled.evaluate({ x: value, a: params.a ?? 1, b: params.b ?? 0, c: params.c ?? 0 }));
      y.push(Number.isFinite(result) && Math.abs(result) < 1e5 ? result : Number.NaN);
    } catch {
      y.push(Number.NaN);
    }
  }
  return { x, y };
}

function PlotFrame({ children, title, eyebrow = "INTERACTIVE MODEL" }: { children: React.ReactNode; title: string; eyebrow?: string }) {
  return <section className="visual-card"><div className="visual-heading"><div><span className="visual-eyebrow">{eyebrow}</span><h3>{title}</h3></div><span className="plot-mark"><FunctionSquare size={18} /></span></div><Suspense fallback={<div className="plot-placeholder">Preparing interactive plot…</div>}>{children}</Suspense></section>;
}

function PlotChart({ config, extra = [] }: { config: FunctionPlotConfig; extra?: Plotly.Data[] }) {
  const main = sample(config.expression, config.domain);
  const data: Plotly.Data[] = [{ x: main.x, y: main.y, type: "scatter", mode: "lines", name: `y = ${config.expression}`, line: { color: "#16785c", width: 3 }, hovertemplate: "x = %{x:.2f}<br>y = %{y:.2f}<extra></extra>" }];
  if (config.second_expression) {
    const second = sample(config.second_expression, config.domain);
    data.push({ x: second.x, y: second.y, type: "scatter", mode: "lines", name: config.second_label ?? `y = ${config.second_expression}`, line: { color: "#e16d54", width: 2.5, dash: "dash" }, hovertemplate: "x = %{x:.2f}<br>y = %{y:.2f}<extra></extra>" });
  }
  data.push(...extra);
  return <Plot data={data} layout={{ autosize: true, height: 290, margin: { l: 42, r: 12, t: 12, b: 38 }, paper_bgcolor: "transparent", plot_bgcolor: "#fbfcfa", showlegend: true, legend: { orientation: "h", y: 1.14, x: 0, font: { size: 11, color: "#60716a" } }, xaxis: { title: { text: "x", font: { size: 12 } }, gridcolor: "#e7ede8", zerolinecolor: "#aab8af" }, yaxis: { title: { text: "y", font: { size: 12 } }, gridcolor: "#e7ede8", zerolinecolor: "#aab8af" }, hovermode: "closest", dragmode: "pan" }} config={{ responsive: true, displaylogo: false, scrollZoom: true, modeBarButtonsToRemove: ["select2d", "lasso2d"] }} useResizeHandler className="plotly-chart" />;
}

export function FunctionPlot({ spec }: { spec: Extract<VisualizationSpec, { type: "function_plot" }> }) {
  return <PlotFrame title={spec.title}><PlotChart config={spec.config} /></PlotFrame>;
}

export function ParameterExplorer({ spec }: { spec: Extract<VisualizationSpec, { type: "parameter_explorer" }> }) {
  const parameters = spec.config.parameters ?? [];
  const [values, setValues] = useState<Record<string, number>>(() => Object.fromEntries(parameters.map((item) => [item.name, item.default])));
  const handleChange = (parameter: PlotParameter, next: number[]) => setValues((previous) => ({ ...previous, [parameter.name]: next[0] }));
  const plotted = sample(spec.config.expression, spec.config.domain, values);
  return (
    <PlotFrame title={spec.title}>
      <div className="parameter-readout">y = {spec.config.expression.replace(/\b(a|b|c)\b/g, (_, key: string) => String(values[key] ?? 0))}</div>
      <div className="parameter-controls">
        {parameters.map((parameter) => <label className="parameter-control" key={parameter.name}>
          <span><b>{parameter.label}</b><output>{(values[parameter.name] ?? parameter.default).toFixed(1)}</output></span>
          <Slider.Root className="range-root" min={parameter.minimum} max={parameter.maximum} step={parameter.step} value={[values[parameter.name] ?? parameter.default]} onValueChange={(next) => handleChange(parameter, next)} aria-label={`Adjust ${parameter.name}`}>
            <Slider.Track className="range-track"><Slider.Range className="range-fill" /></Slider.Track><Slider.Thumb className="range-thumb" aria-label={`Adjust ${parameter.name}`} />
          </Slider.Root>
        </label>)}
      </div>
      <Plot data={[{ x: plotted.x, y: plotted.y, type: "scatter", mode: "lines", name: "f(x)", line: { color: "#16785c", width: 3 }, hovertemplate: "x = %{x:.2f}<br>y = %{y:.2f}<extra></extra>" }]} layout={{ autosize: true, height: 260, margin: { l: 42, r: 12, t: 8, b: 36 }, paper_bgcolor: "transparent", plot_bgcolor: "#fbfcfa", xaxis: { gridcolor: "#e7ede8", zerolinecolor: "#aab8af" }, yaxis: { gridcolor: "#e7ede8", zerolinecolor: "#aab8af" }, hovermode: "closest", dragmode: "pan" }} config={{ responsive: true, displaylogo: false, scrollZoom: true }} useResizeHandler className="plotly-chart" />
    </PlotFrame>
  );
}

export function DerivativePlot({ spec }: { spec: Extract<VisualizationSpec, { type: "derivative_plot" }> }) {
  const { x, y } = sample(spec.config.derivative_expression, spec.config.domain);
  const [point, setPoint] = useState(1);
  const tangentValue = Number(math.evaluate(spec.config.expression, { x: point }));
  const slope = Number(math.evaluate(spec.config.derivative_expression, { x: point }));
  const tangentY = x.map((value) => tangentValue + slope * (value - point));
  return <PlotFrame title={spec.title}><div className="plot-legend"><span><i className="legend-dot green" />f(x) = {spec.config.expression}</span><span><i className="legend-dot coral" />f′(x) = {spec.config.derivative_expression}</span></div><Plot data={[{ ...sample(spec.config.expression, spec.config.domain), type: "scatter", mode: "lines", name: "f(x)", line: { color: "#16785c", width: 3 } }, { x, y, type: "scatter", mode: "lines", name: "f′(x)", line: { color: "#e66b51", width: 2, dash: "dot" } }, { x, y: tangentY, type: "scatter", mode: "lines", name: "Tangent", line: { color: "#b28a3d", width: 1.5, dash: "dash" } }, { x: [point], y: [tangentValue], type: "scatter", mode: "markers", name: "Point", marker: { color: "#b28a3d", size: 9 } }]} layout={{ autosize: true, height: 290, margin: { l: 42, r: 12, t: 10, b: 36 }, paper_bgcolor: "transparent", plot_bgcolor: "#fbfcfa", showlegend: false, xaxis: { gridcolor: "#e7ede8", zerolinecolor: "#aab8af" }, yaxis: { gridcolor: "#e7ede8", zerolinecolor: "#aab8af" }, hovermode: "closest", dragmode: "pan" }} config={{ responsive: true, displaylogo: false, scrollZoom: true }} useResizeHandler className="plotly-chart" /><label className="point-slider"><span>Tangent point <b>x = {point.toFixed(1)}</b></span><input type="range" min={spec.config.domain[0]} max={spec.config.domain[1]} step="0.1" value={point} onChange={(event) => setPoint(Number(event.target.value))} /></label></PlotFrame>;
}

export function IntegralPlot({ spec }: { spec: Extract<VisualizationSpec, { type: "integral_plot" }> }) {
  const { x, y } = sample(spec.config.expression, spec.config.domain);
  const [lower, upper] = spec.config.bounds;
  const indices = x.map((value, index) => value >= lower && value <= upper ? index : -1).filter((index) => index >= 0);
  return <PlotFrame title={spec.title}><div className="plot-legend"><span><i className="legend-dot green" />f(x) = {spec.config.expression}</span><span><i className="legend-dot wash" />shaded area</span></div><Plot data={[{ x, y, type: "scatter", mode: "lines", name: "f(x)", line: { color: "#16785c", width: 3 } }, { x: indices.map((index) => x[index]), y: indices.map((index) => y[index]), type: "scatter", mode: "lines", name: "Area", fill: "tozeroy", fillcolor: "rgba(36, 151, 111, .19)", line: { color: "rgba(36, 151, 111, .65)", width: 1 } }]} layout={{ autosize: true, height: 290, margin: { l: 42, r: 12, t: 10, b: 36 }, paper_bgcolor: "transparent", plot_bgcolor: "#fbfcfa", showlegend: false, xaxis: { gridcolor: "#e7ede8", zerolinecolor: "#aab8af" }, yaxis: { gridcolor: "#e7ede8", zerolinecolor: "#aab8af" }, hovermode: "closest", dragmode: "pan" }} config={{ responsive: true, displaylogo: false, scrollZoom: true }} useResizeHandler className="plotly-chart" /><p className="visual-caption">Shaded from x = {lower} to x = {upper}</p></PlotFrame>;
}

function TriangleDiagram({ points }: { points: [number, number][] }) {
  const titleId = useId();
  const vertices = points.length >= 3 ? points.slice(0, 3) : [[0, 0], [3, 0], [3, 4]];
  const xs = vertices.map(([x]) => x), ys = vertices.map(([, y]) => y);
  const width = Math.max(...xs) - Math.min(...xs), height = Math.max(...ys) - Math.min(...ys);
  const scale = Math.min(240 / Math.max(width, 1), 150 / Math.max(height, 1));
  const mapped = vertices.map(([x, y]) => [60 + (x - Math.min(...xs)) * scale, 190 - (y - Math.min(...ys)) * scale]);
  return <figure className="triangle-diagram">
    <svg viewBox="0 0 360 240" role="img" aria-labelledby={titleId}>
      <title id={titleId}>Triangle diagram with labeled vertices and side lengths</title>
      <polygon points={mapped.map((p) => p.join(",")).join(" ")} fill="currentColor" fillOpacity="0.08" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" />
      {mapped.map(([x, y], i) => {
        const [px, py] = mapped[(i + 1) % 3], [qx, qy] = mapped[(i + 2) % 3];
        const u = [px - x, py - y], v = [qx - x, qy - y];
        const ul = Math.hypot(...u), vl = Math.hypot(...v);
        if (!ul || !vl || Math.abs(u[0] * v[0] + u[1] * v[1]) > ul * vl * 0.001) return null;
        const ux = u[0] / ul * 12, uy = u[1] / ul * 12, vx = v[0] / vl * 12, vy = v[1] / vl * 12;
        return <polyline key={`angle-${i}`} points={`${x + ux},${y + uy} ${x + ux + vx},${y + uy + vy} ${x + vx},${y + vy}`} fill="none" stroke="currentColor" strokeWidth="1.5" />;
      })}
      {mapped.map(([x, y], i) => <g key={i}><circle cx={x} cy={y} r="4" fill="currentColor" /><text x={x - 15} y={y - 10} className="diagram-vertex">{String.fromCharCode(65 + i)}</text></g>)}
      {mapped.map(([x, y], i) => {
        const next = (i + 1) % 3, [nx, ny] = mapped[next];
        const length = Math.hypot(vertices[i][0] - vertices[next][0], vertices[i][1] - vertices[next][1]);
        return <text key={i} x={(x + nx) / 2 + (Math.abs(x - nx) < 1 ? 18 : 0)} y={(y + ny) / 2 + (Math.abs(y - ny) < 1 ? 24 : -10)} textAnchor="middle" className="diagram-length">{Number(length.toFixed(2))}</text>;
      })}
    </svg>
    <figcaption>Starting diagram · side lengths in the same units</figcaption>
  </figure>;
}

export function GeometryCanvas({ spec }: { spec: Extract<VisualizationSpec, { type: "geometry" }> }) {
  const id = `geometry-${useId().replace(/:/g, "")}`;
  const [radius, setRadius] = useState(spec.config.radius || 2);
  useEffect(() => {
    let cancelled = false;
    let disposeBoard: (() => void) | undefined;
    void import("jsxgraph").then(({ default: JXG }) => {
      if (cancelled) return;
      let boundingbox: [number, number, number, number] = [-5, 5, 5, -5];
      if (spec.config.points.length) {
        const xs = spec.config.points.map(([x]) => x), ys = spec.config.points.map(([, y]) => y);
        const padding = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 1) * 0.2;
        boundingbox = [Math.min(...xs) - padding, Math.max(...ys) + padding, Math.max(...xs) + padding, Math.min(...ys) - padding];
      }
      const board = JXG.JSXGraph.initBoard(id, { boundingbox, keepaspectratio: true, axis: true, showNavigation: true, showCopyright: false, pan: { enabled: true }, zoom: { wheel: true } });
      if (spec.config.shape === "circle") {
        board.create("circle", [[0, 0], radius], { strokeColor: "#16785c", strokeWidth: 3, fillColor: "#a9d8c5", fillOpacity: 0.3, fixed: false });
        board.create("point", [0, 0], { name: "O", size: 3, color: "#e66b51", fixed: true });
      } else if (spec.config.shape === "triangle") {
        const points = spec.config.points.length >= 3 ? spec.config.points : [[0, 0], [3, 0], [3, 4]];
        const first = board.create("point", points[0], { name: "A", size: 4, color: "#16785c" });
        const second = board.create("point", points[1], { name: "B", size: 4, color: "#e66b51" });
        const third = board.create("point", points[2], { name: "C", size: 4, color: "#b28a3d" });
        board.create("polygon", [first, second, third], { borders: { strokeColor: "#16785c", strokeWidth: 2 }, fillColor: "#a9d8c5", fillOpacity: 0.24, vertices: { size: 4 } });
      } else {
        const points = spec.config.points.length ? spec.config.points : [[-2, 0], [0, 2], [2, 0]];
        points.forEach((point, index) => board.create("point", point, { name: String.fromCharCode(65 + index), size: 4, color: index === 1 ? "#e66b51" : "#16785c" }));
        if (spec.config.shape === "angle") board.create("angle", [[-2, 0], [0, 0], [2, 0]], { radius: 0.7, fillColor: "#a9d8c5", fillOpacity: 0.45, strokeColor: "#16785c" });
      }
      disposeBoard = () => JXG.JSXGraph.freeBoard(board);
    });
    return () => {
      cancelled = true;
      disposeBoard?.();
    };
  }, [id, radius, spec.config.points, spec.config.shape]);

  return <PlotFrame title={spec.title} eyebrow="GEOMETRY LAB">{spec.config.shape === "triangle" && <TriangleDiagram points={spec.config.points} />}<p className="visual-caption">Drag the points below to explore the shape.</p><div id={id} className="jxgbox geometry-board" role="img" aria-label={`Interactive ${spec.config.shape} diagram`} />{spec.config.shape === "circle" && <label className="point-slider"><span>Radius <b>r = {radius.toFixed(1)}</b></span><input type="range" min="0.5" max="4.5" step="0.1" value={radius} onChange={(event) => setRadius(Number(event.target.value))} /><span className="area-equation">Area = πr² = {(Math.PI * radius ** 2).toFixed(2)}</span></label>}</PlotFrame>;
}

export function StepByStepSolution({ spec }: { spec: Extract<VisualizationSpec, { type: "step_by_step" }> }) {
  return <section className="steps-card"><div className="steps-header"><div><span className="visual-eyebrow">SOLUTION PATH</span><h3>{spec.title}</h3></div><span className="steps-count">{spec.config.steps.length} steps</span></div><div className="steps-list">{spec.config.steps.map((step, index) => <details className="step-row" key={`${index}-${step}`} open={index === 0}><summary><span className="step-number">{String(index + 1).padStart(2, "0")}</span><span className="step-title">Step {index + 1}</span><ChevronDown size={15} className="step-chevron" /></summary><div className="step-content"><MathFormula text={step} /></div></details>)}</div><div className="solution-result"><span><Check size={16} /> Final result</span><strong>{spec.config.result || "See the explanation above"}</strong><small>Verification: {spec.config.verification}</small></div></section>;
}

export function ConceptCard({ spec }: { spec: Extract<VisualizationSpec, { type: "concept_card" }> }) {
  return <section className="concept-card"><span className="visual-eyebrow">CORE IDEA</span><h3>{spec.title}</h3><div className="concept-copy"><MathFormula text={spec.config.explanation} /></div>{spec.config.formula && <div className="formula-box"><MathFormula text={`$${spec.config.formula}$`} /></div>}{spec.config.example && <div className="concept-example"><MoveUpRight size={15} /><MathFormula text={spec.config.example} /></div>}</section>;
}

export function MathFormula({ text }: { text: string }) {
  return <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>{text}</ReactMarkdown>;
}

export function VisualizationRenderer({ spec }: { spec?: VisualizationSpec }) {
  if (!spec || spec.type === "none") return null;
  switch (spec.type) {
    case "function_plot": return <FunctionPlot spec={spec} />;
    case "derivative_plot": return <DerivativePlot spec={spec} />;
    case "integral_plot": return <IntegralPlot spec={spec} />;
    case "geometry": return <GeometryCanvas spec={spec} />;
    case "parameter_explorer": return <ParameterExplorer spec={spec} />;
    case "step_by_step": return <StepByStepSolution spec={spec} />;
    case "concept_card": return <ConceptCard spec={spec} />;
    default: return null;
  }
}

export function ExpandableMessage({ children }: { children: React.ReactNode }) {
  return <details className="answer-details" open><summary><span>Explanation</span><ChevronDown size={14} /></summary>{children}</details>;
}
