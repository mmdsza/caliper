import { GraderWorkspace } from "../components/GraderWorkspace";
import { sampleGrader } from "../lib/sample-grader";

export default function HomePage() {
  return (
    <GraderWorkspace
      graderName={sampleGrader.name}
      graderVersion={sampleGrader.version}
      initialSource={sampleGrader.source}
    />
  );
}
