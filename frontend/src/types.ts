export type Level = "beginner" | "intermediate" | "advanced";

export type PlotParameter = {
  name: "a" | "b" | "c";
  label: string;
  minimum: number;
  maximum: number;
  step: number;
  default: number;
};

export type FunctionPlotConfig = {
  expression: string;
  second_expression?: string | null;
  second_label?: string | null;
  domain: [number, number];
  parameters?: PlotParameter[];
};

export type VisualizationSpec =
  | { type: "function_plot"; title: string; config: FunctionPlotConfig }
  | { type: "derivative_plot"; title: string; config: FunctionPlotConfig & { derivative_expression: string } }
  | { type: "integral_plot"; title: string; config: FunctionPlotConfig & { bounds: [number, number] } }
  | { type: "geometry"; title: string; config: { shape: "circle" | "triangle" | "angle" | "coordinates"; radius: number; points: [number, number][] } }
  | { type: "step_by_step"; title: string; config: { steps: string[]; result: string; verification: string } }
  | { type: "concept_card"; title: string; config: { explanation: string; formula: string; example: string } }
  | { type: "parameter_explorer"; title: string; config: FunctionPlotConfig }
  | { type: "none"; title: string; config: Record<string, never> };

export type ChatResponse = {
  answer: string;
  conversationId: string;
  visualizations: VisualizationSpec[];
  steps: string[];
  metadata: {
    llm?: { configuredProvider: string; provider: string; model: string; available: boolean; used: boolean; error: string };
    explanationLevel?: Level;
    intent: string;
    language: string;
    verification: string;
    conversationMemory: boolean;
  };
};

export type Message = {
  id: string;
  role: "assistant" | "user";
  content: string;
  language?: string;
  visualization?: VisualizationSpec;
  metadata?: ChatResponse["metadata"];
};
export type ConversationSummary = { id: string; title: string; updatedAt: string; level: Level };
export type SavedConversation = ConversationSummary & { messages: Message[] };
