"use client";

import Editor, { type OnChange, type OnMount } from "@monaco-editor/react";

export interface GraderEditorProps {
  value: string;
  onChange: (value: string) => void;
}

export function GraderEditor({ value, onChange }: GraderEditorProps) {
  const handleChange: OnChange = (next) => {
    onChange(next ?? "");
  };

  const handleMount: OnMount = (editor) => {
    editor.layout();
  };

  return (
    <div style={{ flex: 1, minHeight: 0, display: "flex" }}>
      <Editor
        height="100%"
        defaultLanguage="python"
        language="python"
        theme="vs-dark"
        value={value}
        onChange={handleChange}
        onMount={handleMount}
        options={{
          minimap: { enabled: false },
          fontSize: 14,
          tabSize: 4,
          automaticLayout: true,
        }}
      />
    </div>
  );
}

export default GraderEditor;
