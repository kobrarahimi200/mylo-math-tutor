import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { VisualizationSpec } from "../types";
import { MessageBubble } from "./ChatMessages";
import { FunctionPlot, GeometryCanvas, ParameterExplorer, StepByStepSolution } from "./Visualizations";

const { board, initBoard, freeBoard } = vi.hoisted(() => {
  const mockBoard = { create: vi.fn(), update: vi.fn() };
  return { board: mockBoard, initBoard: vi.fn(() => mockBoard), freeBoard: vi.fn() };
});

vi.mock("react-plotly.js", () => ({ default: () => <div data-testid="plot" /> }));
vi.mock("jsxgraph", () => ({ default: { JSXGraph: { initBoard, freeBoard } } }));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("generative math components", () => {
  it("renders an interactive function plot from a typed spec", async () => {
    const spec: Extract<VisualizationSpec, { type: "function_plot" }> = {
      type: "function_plot", title: "Function plot", config: { expression: "x^2", domain: [-10, 10] },
    };
    render(<FunctionPlot spec={spec} />);
    expect(screen.getByRole("heading", { name: "Function plot" })).toBeInTheDocument();
    expect(await screen.findByTestId("plot")).toBeInTheDocument();
  });

  it("renders separate expandable solution steps and verification", () => {
    const spec: Extract<VisualizationSpec, { type: "step_by_step" }> = {
      type: "step_by_step", title: "Worked solution", config: { steps: ["Subtract 5", "Divide by 2"], result: "x = 5", verification: "verified" },
    };
    render(<StepByStepSolution spec={spec} />);
    expect(screen.getByText("Subtract 5")).toBeInTheDocument();
    expect(screen.getByText("Divide by 2")).toBeInTheDocument();
    expect(screen.getByText("x = 5")).toBeInTheDocument();
    expect(screen.getByText("Verification: verified")).toBeInTheDocument();
  });

  it("updates the parameter explorer immediately from its slider", () => {
    const spec: Extract<VisualizationSpec, { type: "parameter_explorer" }> = {
      type: "parameter_explorer", title: "Adjust the function", config: {
        expression: "a*x^2", domain: [-10, 10], parameters: [{ name: "a", label: "a", minimum: -5, maximum: 5, step: 0.1, default: 1 }],
      },
    };
    render(<ParameterExplorer spec={spec} />);
    fireEvent.keyDown(screen.getByRole("slider", { name: "Adjust a" }), { key: "ArrowRight" });
    expect(screen.getByText("1.1")).toBeInTheDocument();
  });

  it("initializes a JSXGraph board for the requested geometry", async () => {
    const spec: Extract<VisualizationSpec, { type: "geometry" }> = {
      type: "geometry", title: "Explore a circle", config: { shape: "circle", radius: 2, points: [] },
    };
    render(<GeometryCanvas spec={spec} />);
    await waitFor(() => expect(initBoard).toHaveBeenCalled());
    expect(board.create).toHaveBeenCalledWith("circle", [[0, 0], 2], expect.any(Object));
    expect(screen.getByLabelText("Interactive circle diagram")).toBeInTheDocument();
  });

  it("uses supplied triangle coordinates and labels the diagram", async () => {
    render(<GeometryCanvas spec={{ type: "geometry", title: "3–4–5 triangle", config: { shape: "triangle", radius: 2, points: [[0, 0], [3, 0], [3, 4]] } }} />);
    expect(screen.getByRole("img", { name: "Triangle diagram with labeled vertices and side lengths" })).toBeInTheDocument();
    await waitFor(() => expect(board.create).toHaveBeenCalledWith("point", [3, 4], expect.any(Object)));
    expect(screen.getByText("5")).toBeInTheDocument();
  });

  it("marks Persian answers as right-to-left while keeping math rendering enabled", () => {
    render(<MessageBubble message={{ id: "fa", role: "assistant", content: "مشتق چیست؟ $f'(x)=2x$", language: "fa" }} />);
    expect(screen.getByText(/مشتق چیست/).closest(".message-content")).toHaveAttribute("dir", "rtl");
  });
});