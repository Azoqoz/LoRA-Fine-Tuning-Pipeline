import experiment from "../data/experiment.json";
import Experiment from "../components/experiment";

export default function Page() {
  return <Experiment data={experiment} />;
}
